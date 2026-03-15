#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import mediapy as media
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model

DATASET_ROOT = Path("/data/Ctrl-World/datasets/large_real_dataset")
TARGET_EPISODE_IDS = "51"
WRIST_VIEW_INDEX = 1
PRIMING_VIDEO_PATH = "/data/Ctrl-World/datasets/initial_hand_motion/videos/6/1_rgb.mp4"
OUTPUT_VIDEO_PATH = Path(__file__).with_suffix(".mp4")

# Bounding box on the prompt frame as [x_min, y_min, x_max, y_max].
# Values can be absolute pixels or normalized [0, 1].
PROMPT_BOUNDING_BOX_XYXY: list[float] | None = [8.0, 136.0, 538.0, 585.0]

PROMPT_OBJ_ID = 0

# If True, priming frames are reversed before being prepended to target frames.
REVERSE_PRIMING_VIDEO = True

FPS = 5
MASK_COLOR_RGB = (0, 255, 0)
MASK_ALPHA = 0.45
FILL_HOLE_AREA = 0
MORPH_CLOSE_RADIUS = 3

def _resize_frames(frames: Iterable[np.ndarray], target_hw: tuple[int, int]) -> list[np.ndarray]:
    target_h, target_w = target_hw
    resized: list[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] == (target_h, target_w):
            resized.append(np.asarray(frame, dtype=np.uint8))
            continue
        pil_frame = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        pil_frame = pil_frame.resize((target_w, target_h), resample=Image.BILINEAR)
        resized.append(np.asarray(pil_frame, dtype=np.uint8))
    return resized


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    return arr > 0.0


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = np.asarray(frame, dtype=np.float32).copy()
    color = np.asarray(color_rgb, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _morphological_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary morphological closing: dilation then erosion."""
    if radius <= 0:
        return mask

    kernel_size = 2 * int(radius) + 1
    x = torch.from_numpy(mask.astype(np.float32))[None, None, ...]

    # Dilation via max-pooling.
    dilated = torch.nn.functional.max_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=radius
    )
    # Erosion via dilation on the complement.
    closed = 1.0 - torch.nn.functional.max_pool2d(
        1.0 - dilated, kernel_size=kernel_size, stride=1, padding=radius
    )
    return (closed[0, 0] > 0.5).cpu().numpy()


def _resolve_prompt_frame_idx(priming_len: int, reverse_priming_video: bool) -> int:
    if priming_len <= 0:
        raise ValueError("priming_len must be > 0")
    return priming_len - 1 if reverse_priming_video else 0


def _normalize_box_xyxy(box_xyxy: list[float], width: int, height: int) -> list[float]:
    box = np.asarray(box_xyxy, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"Expected box as [x_min, y_min, x_max, y_max], got: {box_xyxy}")

    if ((box >= 0.0) & (box <= 1.0)).all():
        normalized = box
    else:
        normalized = np.array(
            [
                box[0] / float(width),
                box[1] / float(height),
                box[2] / float(width),
                box[3] / float(height),
            ],
            dtype=np.float32,
        )

    x1, y1, x2, y2 = [float(v) for v in normalized]
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    normalized = np.array([x1, y1, x2, y2], dtype=np.float32)

    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        raise ValueError(
            "Bounding box is empty after clipping to frame bounds. "
            f"Got normalized box={normalized.tolist()} for width={width}, height={height}."
        )

    return normalized.tolist()


def _resolve_wrist_view_video_path(
    dataset_root: Path, episode_id: int, view_index: int = WRIST_VIEW_INDEX
) -> Path:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory under dataset root: {videos_dir}")
    video_path = videos_dir / str(int(episode_id)) / f"{int(view_index)}_rgb.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing wrist-view video for episode {episode_id}: {video_path}")
    return video_path


def _parse_episode_ids_spec(episode_ids_spec: str) -> set[int]:
    spec = str(episode_ids_spec).strip()
    if not spec:
        raise ValueError("TARGET_EPISODE_IDS cannot be empty. Use 'all' or ids like '1-3,8,10'.")
    if spec.lower() == "all":
        return set()

    tokens = spec.replace(",", " ").split()
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


def _discover_episode_ids(dataset_root: Path, view_index: int = WRIST_VIEW_INDEX) -> list[int]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory under dataset root: {videos_dir}")
    episode_ids: list[int] = []
    for episode_dir in sorted(videos_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        if not episode_dir.name.isdigit():
            continue
        video_path = episode_dir / f"{int(view_index)}_rgb.mp4"
        if video_path.exists():
            episode_ids.append(int(episode_dir.name))
    if not episode_ids:
        raise FileNotFoundError(
            f"No episode folders with '{int(view_index)}_rgb.mp4' found under: {videos_dir}"
        )
    return sorted(set(episode_ids))


def _resolve_target_episode_ids(
    dataset_root: Path, episode_ids_spec: str, view_index: int = WRIST_VIEW_INDEX
) -> list[int]:
    parsed_ids = _parse_episode_ids_spec(episode_ids_spec)
    if not parsed_ids:
        return _discover_episode_ids(dataset_root=dataset_root, view_index=view_index)
    return sorted(parsed_ids)


def _output_video_path_for_episode(episode_id: int) -> Path:
    return OUTPUT_VIDEO_PATH.with_name(
        f"{OUTPUT_VIDEO_PATH.stem}_episode_{int(episode_id)}{OUTPUT_VIDEO_PATH.suffix}"
    )


def main() -> None:
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.fill_hole_area = FILL_HOLE_AREA
    predictor.backbone = sam3_model.detector.backbone

    episode_ids = _resolve_target_episode_ids(
        dataset_root=DATASET_ROOT,
        episode_ids_spec=TARGET_EPISODE_IDS,
        view_index=WRIST_VIEW_INDEX,
    )
    priming_raw_frames = [np.rot90(frame, k=1) for frame in media.read_video(PRIMING_VIDEO_PATH)]
    if not priming_raw_frames:
        raise RuntimeError("Priming video must contain at least one frame.")

    for episode_id in episode_ids:
        try:
            target_video_path = _resolve_wrist_view_video_path(
                dataset_root=DATASET_ROOT,
                episode_id=episode_id,
            )
        except FileNotFoundError as exc:
            print(f"[warning] {exc}")
            continue

        target_video_frames = [
            np.rot90(frame, k=-1) for frame in media.read_video(str(target_video_path))
        ]
        if not target_video_frames:
            print(f"[warning] No frames in target video: {target_video_path}")
            continue

        target_h, target_w = target_video_frames[0].shape[:2]
        priming_video_frames = _resize_frames(priming_raw_frames, (target_h, target_w))
        if REVERSE_PRIMING_VIDEO:
            priming_video_frames = list(reversed(priming_video_frames))
        combined_video_frames = [*priming_video_frames, *target_video_frames]

        with tempfile.TemporaryDirectory(prefix="combined_video_") as tmp_dir:
            combined_video_path = Path(tmp_dir) / "combined.mp4"
            media.write_video(str(combined_video_path), combined_video_frames, fps=FPS)
            inference_state = predictor.init_state(video_path=str(combined_video_path))

            prompt_frame_idx = _resolve_prompt_frame_idx(
                priming_len=len(priming_video_frames),
                reverse_priming_video=REVERSE_PRIMING_VIDEO,
            )

            def propagate_from_current_prompts() -> dict[int, np.ndarray]:
                masks: dict[int, np.ndarray] = {}
                for (
                    frame_idx,
                    obj_ids,
                    _low_res_masks,
                    video_res_masks,
                    obj_scores,
                ) in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(combined_video_frames),
                    reverse=False,
                    propagate_preflight=True,
                ):
                    if video_res_masks is None:
                        continue

                    target_mask: np.ndarray | None = None
                    num_masks = int(video_res_masks.shape[0])
                    if num_masks == 0:
                        continue

                    best_mask_idx = 0
                    if obj_scores is not None:
                        score_tensor = obj_scores.detach().reshape(-1)
                        if score_tensor.numel() >= num_masks:
                            best_mask_idx = int(torch.argmax(score_tensor[:num_masks]).item())

                    target_mask = _to_binary_mask(
                        video_res_masks[best_mask_idx].detach().cpu().numpy()
                    )
                    target_mask = _morphological_close(
                        target_mask, radius=MORPH_CLOSE_RADIUS
                    )
                    if target_mask is not None:
                        masks[int(frame_idx)] = target_mask
                return masks

            # 1) Initial prompt only.
            prompt_frame_h, prompt_frame_w = combined_video_frames[prompt_frame_idx].shape[:2]
            normalized_xyxy = _normalize_box_xyxy(
                box_xyxy=PROMPT_BOUNDING_BOX_XYXY,
                width=prompt_frame_w,
                height=prompt_frame_h,
            )
            print(f"normalized_xyxy: {normalized_xyxy}")
            _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=PROMPT_OBJ_ID,
                box=np.asarray([normalized_xyxy], dtype=np.float32),
            )
            masks_by_frame = propagate_from_current_prompts()

        rendered_frames: list[np.ndarray] = []
        for frame_idx, frame in enumerate(combined_video_frames):
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            mask = masks_by_frame.get(frame_idx)
            if mask is None:
                rendered_frames.append(frame_uint8)
            else:
                rendered_frames.append(_overlay_mask(frame_uint8, mask, MASK_COLOR_RGB, MASK_ALPHA))

        output_path = _output_video_path_for_episode(episode_id)
        media.write_video(str(output_path), rendered_frames, fps=FPS)
        print(f"[episode {episode_id}] Saved segmentation video to: {output_path}")


if __name__ == "__main__":
    main()
