#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mediapy as media
import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - dependency guard
    cv2 = None


DEFAULT_LABEL_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    1: (0, 255, 0),  # hand
    2: (255, 0, 0),  # object
    3: (0, 0, 255),  # object
    4: (255, 255, 0),  # object
    5: (255, 0, 255),  # object
}

VIEWS: Tuple[int, int] = (0, 1)


def _parse_episode_id_tokens(tokens: Sequence[str] | None) -> set[int]:
    if tokens is None:
        return set()
    out: set[int] = set()
    for token in tokens:
        tok = token.strip()
        if not tok:
            continue
        if "-" in tok:
            start_s, end_s = tok.split("-", 1)
            if not start_s or not end_s:
                raise ValueError(f"Invalid episode id range token: {token!r}")
            start = int(start_s)
            end = int(end_s)
            lo, hi = (start, end) if start <= end else (end, start)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    return out


def _parse_label_color(raw_value: str) -> Tuple[int, Tuple[int, int, int]]:
    token = raw_value.strip()
    if ":" not in token:
        raise argparse.ArgumentTypeError(
            f"Invalid --label-color {raw_value!r}. Expected LABEL:R,G,B"
        )
    label_str, rgb_str = token.split(":", 1)
    try:
        label_id = int(label_str.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid label id in --label-color {raw_value!r}"
        ) from exc
    if label_id <= 0:
        raise argparse.ArgumentTypeError(
            f"Label id must be > 0 in --label-color {raw_value!r}"
        )

    rgb_parts = [part.strip() for part in rgb_str.split(",")]
    if len(rgb_parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid RGB in --label-color {raw_value!r}. Expected LABEL:R,G,B"
        )
    try:
        r, g, b = [int(part) for part in rgb_parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid RGB values in --label-color {raw_value!r}"
        ) from exc
    for channel in (r, g, b):
        if channel < 0 or channel > 255:
            raise argparse.ArgumentTypeError(
                f"RGB channels must be in [0,255] in --label-color {raw_value!r}"
            )
    return label_id, (r, g, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a dataset root and recolor existing per-episode segmentation videos "
            "for views 0 and 1 using segmentation label maps."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Source dataset root containing annotation/ and videos/.",
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        required=True,
        help="Destination dataset root. Must not already exist.",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        nargs="+",
        default=None,
        help="Optional episode ids and ranges, e.g. 1 3 5-8. Defaults to all annotation ids.",
    )
    parser.add_argument(
        "--label-color",
        type=_parse_label_color,
        action="append",
        default=[],
        help="Override label color as LABEL:R,G,B (repeatable), e.g. --label-color 1:0,255,0",
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=5.0,
        help="Fallback FPS if source segmentation video FPS cannot be read.",
    )
    return parser.parse_args()


def discover_episode_ids(annotation_dir: Path) -> List[int]:
    annotation_files = sorted(annotation_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")
    episode_ids: List[int] = []
    for annotation_path in annotation_files:
        try:
            episode_ids.append(int(annotation_path.stem))
        except ValueError:
            print(
                f"[warning] Non-numeric annotation filename; skipping: {annotation_path.name}"
            )
    if not episode_ids:
        raise RuntimeError(
            f"No numeric episode ids discovered from annotation filenames in {annotation_dir}"
        )
    return sorted(set(episode_ids))


def _load_label_maps(mask_path: Path) -> np.ndarray:
    label_maps = np.load(mask_path, allow_pickle=False)
    if label_maps.ndim == 2:
        label_maps = label_maps[None, ...]
    if label_maps.ndim != 3:
        raise ValueError(
            f"Expected mask array with shape (T,H,W) or (H,W), got {label_maps.shape} "
            f"for {mask_path}"
        )
    return np.asarray(label_maps)


def _validate_labels_mapped(
    label_maps: np.ndarray,
    label_colors_rgb: Dict[int, Tuple[int, int, int]],
    episode_id: int,
    view_index: int,
) -> List[int]:
    labels = sorted(int(v) for v in np.unique(label_maps).tolist() if int(v) != 0)
    missing = [label for label in labels if label not in label_colors_rgb]
    if missing:
        raise ValueError(
            f"Unmapped labels found in episode={episode_id} view={view_index}: {missing}. "
            f"Known labels: {sorted(label_colors_rgb.keys())}. "
            "Add overrides via --label-color LABEL:R,G,B."
        )
    return labels


def _render_segmentation_frame(
    label_map: np.ndarray, label_colors_rgb: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    frame_out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label_id, rgb in label_colors_rgb.items():
        frame_out[label_map == label_id] = np.asarray(rgb, dtype=np.uint8)
    return frame_out


def _read_video_fps(video_path: Path, default_fps: float) -> float:
    if default_fps <= 0:
        raise ValueError(f"--default-fps must be > 0, got {default_fps}")
    if cv2 is None:
        return float(default_fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return float(default_fps)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        return float(default_fps)
    return fps


def recolor_segmentation_video(
    mask_path: Path,
    segmentation_video_path: Path,
    label_colors_rgb: Dict[int, Tuple[int, int, int]],
    default_fps: float,
    episode_id: int,
    view_index: int,
) -> Tuple[int, float, List[int]]:
    label_maps = _load_label_maps(mask_path=mask_path)
    labels = _validate_labels_mapped(
        label_maps=label_maps,
        label_colors_rgb=label_colors_rgb,
        episode_id=episode_id,
        view_index=view_index,
    )
    output_frames = [
        _render_segmentation_frame(label_map=label_maps[frame_idx], label_colors_rgb=label_colors_rgb)
        for frame_idx in range(label_maps.shape[0])
    ]
    fps = _read_video_fps(video_path=segmentation_video_path, default_fps=default_fps)
    media.write_video(str(segmentation_video_path), output_frames, fps=fps)
    return int(label_maps.shape[0]), float(fps), labels


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    output_dataset_root = args.output_dataset_root
    annotation_dir = dataset_root / "annotation"
    videos_dir = dataset_root / "videos"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")
    if output_dataset_root.exists():
        raise FileExistsError(
            f"Output dataset root already exists: {output_dataset_root}. "
            "Please provide a new path."
        )

    discovered_episode_ids = discover_episode_ids(annotation_dir=annotation_dir)
    if args.episode_ids is None:
        selected_episode_ids = discovered_episode_ids
    else:
        selected_episode_ids = sorted(_parse_episode_id_tokens(args.episode_ids))

    if not selected_episode_ids:
        raise ValueError("No episode ids selected for processing.")

    label_colors_rgb = dict(DEFAULT_LABEL_COLORS_RGB)
    for label_id, rgb in args.label_color:
        label_colors_rgb[int(label_id)] = tuple(int(v) for v in rgb)

    print(f"[copy] src={dataset_root}")
    print(f"[copy] dst={output_dataset_root}")
    shutil.copytree(dataset_root, output_dataset_root, dirs_exist_ok=False)

    output_annotation_dir = output_dataset_root / "annotation"
    output_videos_dir = output_dataset_root / "videos"
    output_available_episode_ids = set(discover_episode_ids(annotation_dir=output_annotation_dir))

    processed = 0
    skipped = 0
    errors = 0
    error_messages: List[str] = []

    print(
        f"[start] episodes={len(selected_episode_ids)} views={list(VIEWS)} "
        f"label_colors={label_colors_rgb}"
    )

    for episode_id in selected_episode_ids:
        if episode_id not in output_available_episode_ids:
            print(
                f"[warning] Episode id {episode_id} not present in copied annotation; "
                "skipping both views."
            )
            skipped += len(VIEWS)
            continue

        episode_dir = output_videos_dir / str(episode_id)
        if not episode_dir.exists():
            print(f"[warning] Missing episode directory in copied dataset: {episode_dir}")
            skipped += len(VIEWS)
            continue

        for view_index in VIEWS:
            mask_path = episode_dir / f"{view_index}_segmentation_mask.npy"
            segmentation_video_path = episode_dir / f"{view_index}_segmentation.mp4"
            if not mask_path.exists():
                print(f"[warning] Missing mask file; skipping: {mask_path}")
                skipped += 1
                continue
            if not segmentation_video_path.exists():
                print(
                    "[warning] Missing segmentation video for FPS/reference; "
                    f"skipping: {segmentation_video_path}"
                )
                skipped += 1
                continue

            print(
                f"[process] episode={episode_id} view={view_index} "
                f"mask={mask_path.name} video={segmentation_video_path.name}"
            )
            try:
                frame_count, fps, labels = recolor_segmentation_video(
                    mask_path=mask_path,
                    segmentation_video_path=segmentation_video_path,
                    label_colors_rgb=label_colors_rgb,
                    default_fps=float(args.default_fps),
                    episode_id=episode_id,
                    view_index=view_index,
                )
            except Exception as exc:
                errors += 1
                message = f"episode={episode_id} view={view_index} failed: {exc}"
                error_messages.append(message)
                print(f"[error] {message}")
                continue

            processed += 1
            print(
                f"[done] episode={episode_id} view={view_index} "
                f"frames={frame_count} fps={fps:.3f} labels={labels}"
            )

    print(
        f"[summary] processed={processed} skipped={skipped} errors={errors} "
        f"output_root={output_dataset_root}"
    )
    if errors > 0:
        first_error = error_messages[0] if error_messages else "unknown error"
        raise RuntimeError(
            f"Completed with {errors} errors. First error: {first_error}"
        )


if __name__ == "__main__":
    main()
