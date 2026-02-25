#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - dependency guard
    cv2 = None

from sam3.model_builder import build_sam3_video_predictor


DEFAULT_DATASET_ROOT = Path(
    "/data/sam3_based_labeling_pipeline/assets/test_le_robot_dataset"
)

# View index -> prompt list. obj_id maps directly to mask label id.
VIEW_PROMPTS = {
    0: [
        {"obj_id": 1, "text": "robot hand"},
        {"obj_id": 2, "text": "red dice"},
    ],
    1: [
        {"obj_id": 1, "text": "robot hand on the left"},
        {"obj_id": 2, "text": "red dice"},
    ],
}

# Label id -> RGB color.
LABEL_COLORS_RGB = {
    1: (0, 255, 0),  # robot hand
    2: (255, 0, 0),  # red dice
}


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SAM3 segmentation for LeRobot videos and save per-view "
            "segmentation videos + label-map masks."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="LeRobot dataset root containing annotation/ and videos/.",
    )
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional episode ids to process. Defaults to all annotation files.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum frames to track per video.",
    )
    return parser.parse_args()


def _to_binary_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_arr = np.asarray(mask)
    while mask_arr.ndim > 2:
        mask_arr = mask_arr[0]
    return mask_arr > 0.0


def _ensure_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected frame with 2 or 3 dims, got shape={arr.shape}")
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] == 4:
        return arr[:, :, :3]
    raise ValueError(f"Unsupported channel count: {arr.shape[2]}")


def _resize_longest_side(frame: np.ndarray, target_longest: int = 1008) -> np.ndarray:
    frame_rgb = _ensure_rgb_uint8(frame)
    h, w = frame_rgb.shape[:2]
    longest = max(h, w)
    if longest == target_longest:
        return frame_rgb
    scale = float(target_longest) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Install it with: pip install opencv-python"
        )
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return _ensure_rgb_uint8(resized)


def _open_video_info(video_path: Path) -> VideoInfo:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Install it with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video shape for: {video_path}")
    if fps <= 0:
        fps = 30.0
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame_count for: {video_path}")

    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def _create_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    assert cv2 is not None
    candidates = [
        (path, "mp4v"),
        (path, "avc1"),
        (path.with_suffix(".avi"), "MJPG"),
    ]
    for out_path, codec in candidates:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*codec),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if writer.isOpened():
            if out_path != path:
                print(
                    f"[warning] Preferred {path.name} unavailable with codec {codec}; "
                    f"writing {out_path.name}."
                )
            return writer
    raise RuntimeError(f"Could not open video writer for output: {path}")


def discover_episode_ids(annotation_dir: Path) -> List[int]:
    annotation_files = sorted(annotation_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")

    episode_ids: List[int] = []
    for annotation_path in annotation_files:
        try:
            with annotation_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            episode_id = int(data.get("episode_id", int(annotation_path.stem)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            episode_id = int(annotation_path.stem)
        episode_ids.append(episode_id)
    return sorted(set(episode_ids))


def _run_propagation_for_prompt(
    predictor,
    session_id: str,
    obj_id: int,
    max_frame_num_to_track: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    outputs_per_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for stream_response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=0,
            max_frame_num_to_track=max_frame_num_to_track,
        )
    ):
        frame_idx = int(stream_response["frame_index"])
        out = stream_response["outputs"]

        scores = np.asarray(out["out_probs"], dtype=np.float32)
        masks = np.asarray(out["out_binary_masks"])
        if len(scores) == 0 or len(masks) == 0:
            continue

        best_idx = int(np.argmax(scores))
        frame_store = outputs_per_frame.setdefault(frame_idx, {})
        frame_store[obj_id] = np.asarray(masks[best_idx])

    return outputs_per_frame


def _merge_outputs(
    dst: Dict[int, Dict[int, np.ndarray]],
    src: Dict[int, Dict[int, np.ndarray]],
) -> None:
    for frame_idx, src_obj_map in src.items():
        frame_map = dst.setdefault(frame_idx, {})
        for obj_id, mask in src_obj_map.items():
            frame_map[int(obj_id)] = np.asarray(mask)


def _resize_mask_if_needed(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    bool_mask = _to_binary_mask(mask)
    if bool_mask.shape == (height, width):
        return bool_mask

    assert cv2 is not None
    resized = cv2.resize(
        bool_mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > 0


def _build_label_map(
    frame_obj_masks: Dict[int, np.ndarray] | None,
    width: int,
    height: int,
) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    if not frame_obj_masks:
        return label_map

    # Draw in ascending label order to keep deterministic overlap handling.
    for obj_id in sorted(frame_obj_masks.keys()):
        if obj_id not in LABEL_COLORS_RGB:
            continue
        mask = _resize_mask_if_needed(frame_obj_masks[obj_id], width=width, height=height)
        label_map[mask] = np.uint8(obj_id)
    return label_map


def _render_segmentation_frame(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label, rgb in LABEL_COLORS_RGB.items():
        out[label_map == label] = np.asarray(rgb, dtype=np.uint8)
    return out


def process_video(
    predictor,
    video_path: Path,
    output_video_path: Path,
    output_mask_path: Path,
    prompts: Iterable[Dict[str, str | int]],
    max_frames: int | None,
) -> None:
    info = _open_video_info(video_path)
    max_frame_num_to_track = min(max_frames, info.frame_count) if max_frames else info.frame_count
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Install it with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for preprocessing: {video_path}")
    original_frames: List[np.ndarray] = []
    for _frame_idx in range(max_frame_num_to_track):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_frames.append(_ensure_rgb_uint8(frame_rgb))
    cap.release()
    if not original_frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    max_frame_num_to_track = len(original_frames)
    preprocessed_frames = [
        _resize_longest_side(frame, target_longest=1008) for frame in original_frames
    ]
    preprocessed_dir = Path(tempfile.mkdtemp(prefix="sam3_preprocessed_frames_"))
    for idx, frame_rgb in enumerate(preprocessed_frames):
        Image.fromarray(frame_rgb).save(preprocessed_dir / f"{idx:06d}.jpg", quality=95)

    start_response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(preprocessed_dir))
    )
    session_id = start_response["session_id"]

    merged_outputs: Dict[int, Dict[int, np.ndarray]] = {}
    try:
        for prompt_spec in prompts:
            obj_id = int(prompt_spec["obj_id"])
            text = str(prompt_spec["text"])

            _ = predictor.handle_request(
                request=dict(type="reset_session", session_id=session_id)
            )
            _ = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    obj_id=obj_id,
                    text=text,
                )
            )

            prompt_outputs = _run_propagation_for_prompt(
                predictor=predictor,
                session_id=session_id,
                obj_id=obj_id,
                max_frame_num_to_track=max_frame_num_to_track,
            )
            _merge_outputs(merged_outputs, prompt_outputs)
    finally:
        try:
            _ = predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        finally:
            for jpg_path in preprocessed_dir.glob("*.jpg"):
                jpg_path.unlink(missing_ok=True)
            preprocessed_dir.rmdir()

    label_maps = np.zeros(
        (max_frame_num_to_track, info.height, info.width),
        dtype=np.uint8,
    )
    for frame_idx in range(max_frame_num_to_track):
        frame_obj_masks = merged_outputs.get(frame_idx)
        label_maps[frame_idx] = _build_label_map(
            frame_obj_masks=frame_obj_masks,
            width=info.width,
            height=info.height,
        )

    np.save(output_mask_path, label_maps)

    writer = _create_video_writer(
        path=output_video_path,
        width=info.width,
        height=info.height,
        fps=info.fps,
    )
    try:
        for frame_idx in range(max_frame_num_to_track):
            frame_rgb = _render_segmentation_frame(label_maps[frame_idx])
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"[done] video={video_path}")
    print(f"[done] segmentation video: {output_video_path}")
    print(f"[done] segmentation mask: {output_mask_path}")
    print(
        f"[done] frames={max_frame_num_to_track} "
        f"labels={np.unique(label_maps).tolist()}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    annotation_dir = dataset_root / "annotation"
    videos_dir = dataset_root / "videos"

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")

    if args.episode_ids is None:
        episode_ids = discover_episode_ids(annotation_dir=annotation_dir)
    else:
        episode_ids = sorted(set(int(x) for x in args.episode_ids))

    predictor = build_sam3_video_predictor()
    try:
        for episode_id in episode_ids:
            episode_dir = videos_dir / str(episode_id)
            if not episode_dir.exists():
                print(f"[warning] Missing episode directory: {episode_dir}")
                continue

            for view_index in (0, 1):
                input_video_path = episode_dir / f"{view_index}_rgb.mp4"
                if not input_video_path.exists():
                    print(f"[warning] Missing input video: {input_video_path}")
                    continue

                output_video_path = episode_dir / f"{view_index}_segmentation.mp4"
                output_mask_path = episode_dir / f"{view_index}_segmentation_mask.npy"
                prompts = VIEW_PROMPTS[view_index]

                print(
                    f"[episode {episode_id}] processing view={view_index} "
                    f"video={input_video_path.name}"
                )
                process_video(
                    predictor=predictor,
                    video_path=input_video_path,
                    output_video_path=output_video_path,
                    output_mask_path=output_mask_path,
                    prompts=prompts,
                    max_frames=args.max_frames,
                )
    finally:
        predictor.shutdown()


if __name__ == "__main__":
    main()
