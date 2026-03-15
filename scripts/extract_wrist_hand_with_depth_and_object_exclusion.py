#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

try:
    import cv2  # pylint: disable=no-member
except ImportError:  # pragma: no cover - dependency guard
    cv2 = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - dependency guard
    torch = None
    F = None

DEFAULT_OUTPUT_FPS = 5.0
DEFAULT_NEAR_QUANTILE = 0.90
DEFAULT_OBJECT_PROMPT = "red cube"
DEFAULT_FALLBACK_PROMPT = "red dice"


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract wrist-view hand masks by combining SAM3 object masks with "
            "DepthAnythingV2 near-camera depth."
        )
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root. Default is <video_dir>/<video_stem>_wrist_hand_depth.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for number of frames to process.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=DEFAULT_OUTPUT_FPS,
        help="FPS for written preview and overlay videos.",
    )
    parser.add_argument(
        "--object-prompt",
        type=str,
        default=DEFAULT_OBJECT_PROMPT,
        help="Primary SAM3 prompt used to segment the held object.",
    )
    parser.add_argument(
        "--fallback-object-prompt",
        type=str,
        default=DEFAULT_FALLBACK_PROMPT,
        help="Fallback prompt used if the primary prompt is weak.",
    )
    parser.add_argument(
        "--prompt-prob-fallback-threshold",
        type=float,
        default=0.30,
        help="Use fallback prompt if mean primary prompt probability is below this.",
    )
    parser.add_argument(
        "--object-min-area-px",
        type=int,
        default=120,
        help="Minimum area for an object mask to be considered visible.",
    )
    parser.add_argument(
        "--object-mask-threshold",
        type=float,
        default=0.0,
        help="Threshold applied to SAM3 mask logits before binarization.",
    )
    parser.add_argument(
        "--depth-model-id",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face model id for DepthAnythingV2.",
    )
    parser.add_argument(
        "--depth-device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for depth model inference.",
    )
    parser.add_argument(
        "--depth-batch-size",
        type=int,
        default=4,
        help="Batch size for depth inference.",
    )
    parser.add_argument(
        "--near-quantile",
        type=float,
        default=DEFAULT_NEAR_QUANTILE,
        help="Quantile used for near-camera thresholding.",
    )
    parser.add_argument(
        "--near-depth-direction",
        type=str,
        default="higher_is_near",
        choices=["higher_is_near", "lower_is_near"],
        help="Interpretation of depth output. DepthAnythingV2 usually uses higher_is_near.",
    )
    parser.add_argument(
        "--near-margin",
        type=float,
        default=0.0,
        help=(
            "Margin added to near threshold. Positive margin expands near mask for "
            "higher_is_near and shrinks it for lower_is_near."
        ),
    )
    parser.add_argument(
        "--object-visible-min-area",
        type=int,
        default=120,
        help="Subtract object mask only when object area is at least this many pixels.",
    )
    parser.add_argument(
        "--morph-open-kernel",
        type=int,
        default=3,
        help="Kernel size for morphological opening (0 to disable).",
    )
    parser.add_argument(
        "--morph-close-kernel",
        type=int,
        default=5,
        help="Kernel size for morphological closing (0 to disable).",
    )
    parser.add_argument(
        "--keep-largest-components",
        type=int,
        default=1,
        help="How many largest connected components to keep in hand mask (0 keeps all).",
    )
    parser.add_argument(
        "--hand-overlay-color-rgb",
        type=int,
        nargs=3,
        default=[0, 255, 0],
        metavar=("R", "G", "B"),
        help="Hand mask overlay color.",
    )
    parser.add_argument(
        "--hand-overlay-alpha",
        type=float,
        default=0.35,
        help="Alpha for hand mask overlay.",
    )
    return parser.parse_args()


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
    assert cv2 is not None
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return _ensure_rgb_uint8(resized)


def _open_video_info(video_path: Path) -> VideoInfo:
    assert cv2 is not None
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
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame_count for: {video_path}")
    return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)


def _read_video_frames(video_path: Path, max_frames: int | None) -> tuple[VideoInfo, List[np.ndarray]]:
    info = _open_video_info(video_path)
    assert cv2 is not None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for decoding: {video_path}")
    limit = info.frame_count if max_frames is None else min(max_frames, info.frame_count)
    frames: List[np.ndarray] = []
    for _ in range(limit):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(_ensure_rgb_uint8(frame_rgb))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return info, frames


def _create_video_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    assert cv2 is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {path}")
    return writer


def _to_binary_mask(mask: np.ndarray | object, threshold: float = 0.0) -> np.ndarray:
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_arr = np.asarray(mask)
    while mask_arr.ndim > 2:
        mask_arr = mask_arr[0]
    return mask_arr > float(threshold)


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


def _overlay_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: Sequence[int],
    alpha: float,
) -> np.ndarray:
    out = np.asarray(frame_rgb, dtype=np.uint8).copy()
    mask_bool = _to_binary_mask(mask)
    if mask_bool.shape != out.shape[:2]:
        mask_bool = _resize_mask_if_needed(mask_bool, width=out.shape[1], height=out.shape[0])
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask_bool] = np.asarray(color_rgb, dtype=np.uint8)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0.0)


def _prepare_preprocessed_frames(frames: List[np.ndarray]) -> Path:
    preprocessed_frames = [_resize_longest_side(frame, target_longest=1008) for frame in frames]
    preprocessed_dir = Path(tempfile.mkdtemp(prefix="sam3_preprocessed_frames_"))
    for idx, frame_rgb in enumerate(preprocessed_frames):
        Image.fromarray(frame_rgb).save(preprocessed_dir / f"{idx:06d}.jpg", quality=95)
    return preprocessed_dir


def _run_sam3_prompt(
    predictor,
    session_id: str,
    prompt_text: str,
    frame_count: int,
) -> tuple[Dict[int, np.ndarray], Dict[int, float]]:
    _ = predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    _ = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            obj_id=0,
            text=prompt_text,
        )
    )

    mask_by_frame: Dict[int, np.ndarray] = {}
    prob_by_frame: Dict[int, float] = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=0,
            max_frame_num_to_track=frame_count,
            propagation_direction="both",
        )
    ):
        frame_idx = int(response["frame_index"])
        out = response["outputs"]
        probs = np.asarray(out["out_probs"], dtype=np.float32).reshape(-1)
        masks = np.asarray(out["out_binary_masks"])
        if probs.size == 0 or len(masks) == 0:
            continue
        best_idx = int(np.argmax(probs))
        mask_by_frame[frame_idx] = np.asarray(masks[best_idx])
        prob_by_frame[frame_idx] = float(probs[best_idx])
    return mask_by_frame, prob_by_frame


def _extract_object_masks(
    predictor,
    frames: List[np.ndarray],
    prompt_text: str,
    fallback_prompt_text: str,
    fallback_threshold: float,
    object_mask_threshold: float,
    object_min_area_px: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    preprocessed_dir = _prepare_preprocessed_frames(frames)
    frame_count = len(frames)
    h, w = frames[0].shape[:2]

    start_response = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(preprocessed_dir))
    )
    session_id = start_response["session_id"]
    best_prompt = prompt_text
    run_summaries: List[Dict[str, object]] = []

    try:
        primary_masks, primary_probs = _run_sam3_prompt(
            predictor=predictor,
            session_id=session_id,
            prompt_text=prompt_text,
            frame_count=frame_count,
        )
        primary_mean_prob = float(np.mean(list(primary_probs.values()))) if primary_probs else 0.0
        run_summaries.append(
            {
                "prompt": prompt_text,
                "mean_prob": primary_mean_prob,
                "frames_with_candidates": int(len(primary_probs)),
            }
        )
        chosen_masks = primary_masks
        chosen_probs = primary_probs

        if fallback_prompt_text and primary_mean_prob < float(fallback_threshold):
            fallback_masks, fallback_probs = _run_sam3_prompt(
                predictor=predictor,
                session_id=session_id,
                prompt_text=fallback_prompt_text,
                frame_count=frame_count,
            )
            fallback_mean_prob = (
                float(np.mean(list(fallback_probs.values()))) if fallback_probs else 0.0
            )
            run_summaries.append(
                {
                    "prompt": fallback_prompt_text,
                    "mean_prob": fallback_mean_prob,
                    "frames_with_candidates": int(len(fallback_probs)),
                }
            )
            if fallback_mean_prob > primary_mean_prob:
                chosen_masks = fallback_masks
                chosen_probs = fallback_probs
                best_prompt = fallback_prompt_text
    finally:
        _ = predictor.handle_request(request=dict(type="close_session", session_id=session_id))

    object_mask = np.zeros((frame_count, h, w), dtype=bool)
    object_area_px: List[int] = []
    object_prob: List[float] = []
    for frame_idx in range(frame_count):
        raw_mask = chosen_masks.get(frame_idx)
        if raw_mask is None:
            object_area_px.append(0)
            object_prob.append(float("nan"))
            continue
        mask = _to_binary_mask(raw_mask, threshold=object_mask_threshold)
        mask = _resize_mask_if_needed(mask, width=w, height=h)
        if int(mask.sum()) < int(object_min_area_px):
            mask = np.zeros((h, w), dtype=bool)
        object_mask[frame_idx] = mask
        object_area_px.append(int(mask.sum()))
        object_prob.append(float(chosen_probs.get(frame_idx, float("nan"))))

    info = {
        "selected_prompt": best_prompt,
        "prompt_runs": run_summaries,
        "object_area_px": object_area_px,
        "object_prob": object_prob,
    }
    return object_mask, info


def _load_depthanything_v2(model_id: str, device: str):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required. Install it with: pip install torch")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ModuleNotFoundError(
            "Transformers is required for DepthAnythingV2. "
            "Install with: pip install transformers"
        ) from exc

    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    requested_device = device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    model = model.to(requested_device)
    model.eval()
    return model, processor, requested_device


def _estimate_depth_maps(
    frames: List[np.ndarray],
    model_id: str,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    model, processor, resolved_device = _load_depthanything_v2(model_id=model_id, device=device)
    if torch is None or F is None:
        raise ModuleNotFoundError("PyTorch is required for depth inference.")
    if batch_size <= 0:
        raise ValueError("--depth-batch-size must be >= 1")

    h, w = frames[0].shape[:2]
    all_depth: List[np.ndarray] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start : start + batch_size]
        pil_images = [Image.fromarray(frame) for frame in batch_frames]
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {
            key: value.to(resolved_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs)
            predicted = outputs.predicted_depth
            predicted = predicted.unsqueeze(1)
            predicted = F.interpolate(
                predicted,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            predicted = predicted[:, 0, :, :].detach().cpu().numpy().astype(np.float32)
        all_depth.extend(list(predicted))
    depth_maps = np.asarray(all_depth, dtype=np.float32)
    if depth_maps.shape != (len(frames), h, w):
        raise RuntimeError(
            f"Unexpected depth shape {depth_maps.shape}, expected {(len(frames), h, w)}"
        )
    info = {
        "depth_model_id": model_id,
        "depth_device_requested": device,
        "depth_device_resolved": resolved_device,
        "depth_batch_size": int(batch_size),
    }
    return depth_maps, info


def _keep_largest_components(mask: np.ndarray, keep_n: int) -> np.ndarray:
    if keep_n <= 0:
        return mask
    assert cv2 is not None
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    keep_ids = set(int(idx + 1) for idx in order[:keep_n])
    out = np.zeros_like(mask, dtype=bool)
    for label_id in keep_ids:
        out |= labels == label_id
    return out


def _clean_mask(
    mask: np.ndarray,
    open_kernel: int,
    close_kernel: int,
    keep_largest_components: int,
) -> np.ndarray:
    assert cv2 is not None
    out = mask.astype(np.uint8)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    out_bool = out > 0
    out_bool = _keep_largest_components(out_bool, keep_n=keep_largest_components)
    return out_bool


def _make_depth_preview_frame(depth_map: np.ndarray) -> np.ndarray:
    assert cv2 is not None
    depth = np.asarray(depth_map, dtype=np.float32)
    min_v = float(np.min(depth))
    max_v = float(np.max(depth))
    if max_v - min_v < 1e-8:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm_f = (depth - min_v) / (max_v - min_v)
        norm = np.clip(norm_f * 255.0, 0.0, 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def _compute_hand_masks(
    depth_maps: np.ndarray,
    object_masks: np.ndarray,
    near_quantile: float,
    near_depth_direction: str,
    near_margin: float,
    object_visible_min_area: int,
    morph_open_kernel: int,
    morph_close_kernel: int,
    keep_largest_components: int,
) -> tuple[np.ndarray, Dict[str, List[float | int | bool]]]:
    frame_count, h, w = depth_maps.shape
    hand_masks = np.zeros((frame_count, h, w), dtype=bool)
    near_thresholds: List[float] = []
    near_area_px: List[int] = []
    hand_area_px: List[int] = []
    object_area_px: List[int] = []
    subtraction_applied: List[bool] = []

    q = float(np.clip(near_quantile, 0.0, 1.0))
    for i in range(frame_count):
        depth = depth_maps[i]
        threshold = float(np.quantile(depth, q))
        if near_depth_direction == "higher_is_near":
            threshold = threshold - float(near_margin)
            near_mask = depth >= threshold
        else:
            threshold = threshold + float(near_margin)
            near_mask = depth <= threshold

        obj_mask = object_masks[i].astype(bool)
        obj_area = int(obj_mask.sum())
        apply_subtraction = obj_area >= int(object_visible_min_area)
        hand_mask = near_mask & (~obj_mask if apply_subtraction else np.ones_like(obj_mask, dtype=bool))
        hand_mask = _clean_mask(
            hand_mask,
            open_kernel=int(morph_open_kernel),
            close_kernel=int(morph_close_kernel),
            keep_largest_components=int(keep_largest_components),
        )

        hand_masks[i] = hand_mask
        near_thresholds.append(threshold)
        near_area_px.append(int(near_mask.sum()))
        hand_area_px.append(int(hand_mask.sum()))
        object_area_px.append(obj_area)
        subtraction_applied.append(bool(apply_subtraction))

    stats = {
        "near_threshold": near_thresholds,
        "near_area_px": near_area_px,
        "object_area_px": object_area_px,
        "hand_area_px": hand_area_px,
        "subtraction_applied": subtraction_applied,
    }
    return hand_masks, stats


def _resolve_output_dir(video_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return video_path.parent / f"{video_path.stem}_wrist_hand_depth"


def run_extraction(args: argparse.Namespace) -> None:
    from sam3.model_builder import build_sam3_video_predictor

    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Install it with: pip install opencv-python"
        )
    if not args.video_path.exists():
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not (0.0 <= float(args.near_quantile) <= 1.0):
        raise ValueError("--near-quantile must be in [0, 1]")
    if args.object_min_area_px < 0:
        raise ValueError("--object-min-area-px must be >= 0")
    if args.object_visible_min_area < 0:
        raise ValueError("--object-visible-min-area must be >= 0")

    output_root = _resolve_output_dir(args.video_path, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    info, frames = _read_video_frames(video_path=args.video_path, max_frames=args.max_frames)
    fps = float(args.output_fps) if float(args.output_fps) > 0 else (info.fps or DEFAULT_OUTPUT_FPS)
    frame_count = len(frames)
    h, w = frames[0].shape[:2]

    print(f"[info] frames={frame_count} shape=({h}, {w}) fps_out={fps:.3f}")

    predictor = build_sam3_video_predictor()
    object_masks, object_info = _extract_object_masks(
        predictor=predictor,
        frames=frames,
        prompt_text=str(args.object_prompt),
        fallback_prompt_text=str(args.fallback_object_prompt),
        fallback_threshold=float(args.prompt_prob_fallback_threshold),
        object_mask_threshold=float(args.object_mask_threshold),
        object_min_area_px=int(args.object_min_area_px),
    )
    print(
        "[info] object prompt selected:",
        object_info.get("selected_prompt", str(args.object_prompt)),
    )

    depth_maps, depth_info = _estimate_depth_maps(
        frames=frames,
        model_id=str(args.depth_model_id),
        device=str(args.depth_device),
        batch_size=int(args.depth_batch_size),
    )
    print(
        f"[info] depth model={depth_info['depth_model_id']} "
        f"device={depth_info['depth_device_resolved']}"
    )

    hand_masks, hand_stats = _compute_hand_masks(
        depth_maps=depth_maps,
        object_masks=object_masks,
        near_quantile=float(args.near_quantile),
        near_depth_direction=str(args.near_depth_direction),
        near_margin=float(args.near_margin),
        object_visible_min_area=int(args.object_visible_min_area),
        morph_open_kernel=int(args.morph_open_kernel),
        morph_close_kernel=int(args.morph_close_kernel),
        keep_largest_components=int(args.keep_largest_components),
    )

    depth_npy_path = output_root / "depth_float32.npy"
    object_npy_path = output_root / "object_mask.npy"
    hand_npy_path = output_root / "hand_mask.npy"
    depth_preview_path = output_root / "depth_preview.mp4"
    hand_overlay_path = output_root / "hand_overlay.mp4"
    metadata_path = output_root / "metadata.json"

    np.save(depth_npy_path, depth_maps.astype(np.float32))
    np.save(object_npy_path, object_masks.astype(np.uint8))
    np.save(hand_npy_path, hand_masks.astype(np.uint8))

    depth_writer = _create_video_writer(depth_preview_path, width=w, height=h, fps=fps)
    overlay_writer = _create_video_writer(hand_overlay_path, width=w, height=h, fps=fps)
    try:
        for i in range(frame_count):
            depth_vis = _make_depth_preview_frame(depth_maps[i])
            depth_writer.write(cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

            overlay = _overlay_mask(
                frame_rgb=frames[i],
                mask=hand_masks[i],
                color_rgb=args.hand_overlay_color_rgb,
                alpha=float(args.hand_overlay_alpha),
            )
            overlay_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        depth_writer.release()
        overlay_writer.release()

    metadata = {
        "video_path": str(args.video_path),
        "frame_count": int(frame_count),
        "frame_shape_hw": [int(h), int(w)],
        "output_fps": float(fps),
        "sam3": object_info,
        "depth": depth_info,
        "hand_mask": {
            "near_quantile": float(args.near_quantile),
            "near_depth_direction": str(args.near_depth_direction),
            "near_margin": float(args.near_margin),
            "object_visible_min_area": int(args.object_visible_min_area),
            "morph_open_kernel": int(args.morph_open_kernel),
            "morph_close_kernel": int(args.morph_close_kernel),
            "keep_largest_components": int(args.keep_largest_components),
            "per_frame_stats": hand_stats,
        },
        "outputs": {
            "depth_float32_npy": str(depth_npy_path),
            "object_mask_npy": str(object_npy_path),
            "hand_mask_npy": str(hand_npy_path),
            "depth_preview_mp4": str(depth_preview_path),
            "hand_overlay_mp4": str(hand_overlay_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[done] wrote outputs to: {output_root}")


def main() -> None:
    args = parse_args()
    run_extraction(args)


if __name__ == "__main__":
    main()
