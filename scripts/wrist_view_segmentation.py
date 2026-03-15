#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
import mediapy as media

from sam3.model_builder import build_sam3_video_predictor

TARGET_VIDEO_PATH = "/data/Ctrl-World/datasets/large_real_dataset/videos/0/1_rgb.mp4"
PRIMING_VIDEO_PATH = "/data/Ctrl-World/datasets/initial_hand_motion/videos/6/1_rgb.mp4"
OUTPUT_VIDEO_PATH = Path(__file__).with_suffix(".mp4")
PROMPT_TEXT = "wrist-view hand"
FPS = 5
MASK_COLOR_RGB = (0, 255, 0)
MASK_ALPHA = 0.45


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


def abs_to_rel_coords(coords, img_width, img_height, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)."""
    if coord_type == "point":
        return [[x / img_width, y / img_height] for x, y in coords]
    if coord_type == "box":
        return [
            [x / img_width, y / img_height, w / img_width, h / img_height]
            for x, y, w, h in coords
        ]
    raise ValueError(f"Unknown coord_type: {coord_type}")


def main() -> None:
    predictor = build_sam3_video_predictor()
    session_id: str | None = None
    try:
        target_video_frames = [np.rot90(frame, k=-1) for frame in media.read_video(TARGET_VIDEO_PATH)]
        priming_video_frames = reversed([np.rot90(frame, k=1) for frame in media.read_video(PRIMING_VIDEO_PATH)])

        if not target_video_frames or not priming_video_frames:
            raise RuntimeError("Both target and priming videos must contain at least one frame.")

        target_h, target_w = target_video_frames[0].shape[:2]
        priming_video_frames = _resize_frames(priming_video_frames, (target_h, target_w))
        combined_video_frames = [*priming_video_frames, *target_video_frames]

        with tempfile.TemporaryDirectory(prefix="combined_video_") as tmp_dir:
            combined_video_path = Path(tmp_dir)
            for i, frame in enumerate(combined_video_frames):
                Image.fromarray(frame).save(combined_video_path / f"{i:06d}.jpg")

            start_response = predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=str(combined_video_path),
                )
            )
            session_id = start_response["session_id"]

            prompt_frame_idx = 0

            points_abs = np.array(
                [
                    [270, 20],  # [x, y] measured from top-left corner
                ]
            )
            # positive clicks have label 1, while negative clicks have label 0
            click_x, click_y = int(points_abs[0][0]), int(points_abs[0][1])

            def propagate_from_current_prompts() -> dict[int, np.ndarray]:
                masks: dict[int, np.ndarray] = {}
                for response in predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        start_frame_index=prompt_frame_idx,
                        max_frame_num_to_track=len(combined_video_frames),
                        propagation_direction="forward",
                    )
                ):
                    frame_idx = int(response["frame_index"])
                    out = response["outputs"]
                    out_masks = np.asarray(out.get("out_binary_masks", []))
                    if out_masks.size == 0:
                        continue

                    selected_mask: np.ndarray | None = None
                    for raw_mask in out_masks:
                        candidate_mask = _to_binary_mask(raw_mask)
                        h, w = candidate_mask.shape[:2]
                        if 0 <= click_x < w and 0 <= click_y < h and candidate_mask[click_y, click_x]:
                            selected_mask = candidate_mask
                            break
                    if selected_mask is not None:
                        masks[frame_idx] = selected_mask
                return masks

            # Pass 1: text-only prompt propagation.
            _ = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=prompt_frame_idx,
                    text=PROMPT_TEXT,
                    obj_id=0,
                )
            )
            text_masks_by_frame = propagate_from_current_prompts()

            masks_by_frame = text_masks_by_frame
            print(
                "Pass stats - text-only frames:"
                f" {len(text_masks_by_frame)}"
            )
        rendered_frames: list[np.ndarray] = []
        for frame_idx, frame in enumerate(combined_video_frames):
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            mask = masks_by_frame.get(frame_idx)
            if mask is None:
                rendered_frames.append(frame_uint8)
            else:
                rendered_frames.append(_overlay_mask(frame_uint8, mask, MASK_COLOR_RGB, MASK_ALPHA))

        media.write_video(str(OUTPUT_VIDEO_PATH), rendered_frames, fps=FPS)
        print(f"Saved segmentation video to: {OUTPUT_VIDEO_PATH}")
    finally:
        if session_id is not None:
            _ = predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        predictor.shutdown()


if __name__ == "__main__":
    main()