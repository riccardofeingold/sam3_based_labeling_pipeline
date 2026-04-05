#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import mediapy as media
import numpy as np
import torch
from PIL import Image

from dotenv import load_dotenv
load_dotenv(override=True)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
if not DISCORD_WEBHOOK_URL:
    raise ValueError("DISCORD_WEBHOOK_URL is not set")

try:
    import cv2
except ImportError:  # pragma: no cover - dependency guard
    cv2 = None

from sam3.model_builder import build_sam3_video_model, build_sam3_video_predictor


DEFAULT_DATASET_ROOT = Path("datasets/converted_processed_OOD/2026-04-04T12-25-46/processed_OOD")
DEFAULT_CALIBRATION_DIR = Path("/home/riccardo/sam3_based_labeling_pipeline/assets/calibration_params_08_03_26")
DEFAULT_WRIST_PRIMING_VIDEO_PATH = Path(
    "/home/riccardo/sam3_based_labeling_pipeline/assets/initial_hand_motion/videos/6/1_rgb.mp4"
)
WRIST_VIEW_INDEX = 1
THIRD_VIEW_INDEX = 0

# View index -> prompt list.
# Prompt fields:
# - text: run with build_sam3_video_predictor
# - box: run with build_sam3_video_model().tracker
# - box_from_projection: derive box from calibration and axis-aligned ROI
# - use_priming_video: prepend reversed priming video to each chunk
# - reprompt_every: for text prompts, re-add prompt every N frames
VIEW_PROMPTS = {
    0: [
        {
            "obj_id": 0,
            "label_id": 1,
            "text": "the hand",
            "select_mask_based_on_projected_ee": True,
            "reprompt_every": 100,
        },
        {"obj_id": 1, "label_id": 2, "text": "red dice"},
        {"obj_id": 2, "label_id": 3, "text": "blue dice"},
        {"obj_id": 3, "label_id": 4, "text": "yellow dice"},
        {"obj_id": 4, "label_id": 5, "text": "yellow duck"}
    ],
    1: [
        {
            "obj_id": 0,
            "label_id": 1,
            "box": (8.0, 136.0, 538.0, 585.0),  # xmin, ymin, xmax, ymax
            "use_priming_video": True,
        },
        {"obj_id": 1, "label_id": 2, "text": "red dice"},
        {"obj_id": 2, "label_id": 3, "text": "blue dice"},
        {"obj_id": 3, "label_id": 4, "text": "yellow dice"},
        {"obj_id": 4, "label_id": 5, "text": "duck"}
    ],
}

LABEL_COLORS_RGB = {
    1: (0, 255, 0),  # hand
    2: (255, 0, 0),  # object
    3: (0, 0, 255),  # object
    4: (255, 255, 0),  # object
    5: (255, 0, 255),  # object
}

VIDEO_INDEX_TO_CAMERA = {
    0: "oakd_side_view",
    1: "oakd_wrist_view",
}

CAMERA_EXTRINSIC_MODE = {
    "oakd_side_view": "base_camera",
    "oakd_wrist_view": "ee_camera",
}

EE_TRANSLATION_OFFSET = np.array([0.13, 0.0, 0.07], dtype=np.float64)
DEFAULT_MORPH_CLOSE_RADIUS = 3


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    T_ee_camera: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SAM3 segmentation masks with mixed prompt types "
            "(text / box / projection-box)."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--episode-ids",
        type=str,
        nargs="+",
        default=None,
        help="Episode ids and ranges, e.g. 1 3 5-10. Defaults to all.",
    )
    parser.add_argument(
        "--rotate-view1-episode-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional episode ids for which videos/<episode>/1_rgb.mp4 is rotated by 180 "
            "degrees in-place before SAM3 processing. Supports ranges like '1-10'."
        ),
    )
    parser.add_argument(
        "--view-mode",
        type=str,
        choices=["wrist", "third", "both"],
        default="both",
        help="Process only wrist view, third view, or both.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--chunk-max-frames",
        type=int,
        default=1000,
        help="Max target frames per chunk. <=0 disables chunking.",
    )
    parser.add_argument("--downscaled-long-side-px", type=int, default=None)
    parser.add_argument("--output-fps", type=float, default=5.0)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument(
        "--wrist-priming-video-path",
        type=Path,
        default=DEFAULT_WRIST_PRIMING_VIDEO_PATH,
    )
    parser.add_argument(
        "--quaternion-order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
    )
    parser.add_argument("--axis-length-m", type=float, default=0.05)
    parser.add_argument("--roi-neg-primary-m", type=float, default=0.05)
    parser.add_argument("--roi-pos-primary-m", type=float, default=0.12)
    parser.add_argument("--roi-neg-secondary-m", type=float, default=0.12)
    parser.add_argument("--roi-pos-secondary-m", type=float, default=0.12)
    parser.add_argument(
        "--reprompt-min-border-px",
        type=int,
        default=0,
        help=(
            "Minimum distance (in pixels) that the projected EE point must be from "
            "any frame border to trigger a reprompt. 0 disables the check (always reprompt)."
        ),
    )
    parser.add_argument(
        "--morph-close-radius",
        type=int,
        default=DEFAULT_MORPH_CLOSE_RADIUS,
        help="Morphological close radius. Use <=0 to disable.",
    )
    parser.add_argument(
        "--morph-close-views",
        type=str,
        choices=["wrist", "third", "both", "none"],
        default="wrist",
        help="Which view(s) should receive morphological close post-processing.",
    )
    parser.add_argument(
        "--discord-webhook-url",
        type=str,
        default=os.environ.get("DISCORD_WEBHOOK_URL"),
        help=(
            "Discord webhook URL for completion notifications. "
            "Defaults to DISCORD_WEBHOOK_URL env var."
        ),
    )
    parser.add_argument(
        "--discord-mention",
        type=str,
        default="",
        help="Optional mention prefix in Discord message, e.g. '@here' or '<@123>'.",
    )
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=20,
        help=(
            "Minimum free VRAM (in GB) required on at least one GPU before loading models. "
            "Script will poll all GPUs and wait until the condition is met. "
            "If not set, no VRAM check is performed."
        ),
    )
    parser.add_argument(
        "--vram-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between VRAM availability checks (default: 30).",
    )
    return parser.parse_args()


def _parse_episode_id_tokens(tokens: List[str] | None) -> set[int]:
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


def _selected_views(view_mode: str) -> set[int]:
    if view_mode == "wrist":
        return {WRIST_VIEW_INDEX}
    if view_mode == "third":
        return {THIRD_VIEW_INDEX}
    return {WRIST_VIEW_INDEX, THIRD_VIEW_INDEX}


def _morph_close_enabled_for_view(view_index: int, morph_close_views: str) -> bool:
    if morph_close_views == "none":
        return False
    if morph_close_views == "both":
        return True
    if morph_close_views == "wrist":
        return view_index == WRIST_VIEW_INDEX
    if morph_close_views == "third":
        return view_index == THIRD_VIEW_INDEX
    return False


def _to_binary_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    return arr > 0.0


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


def _resize_frames_to_shape(
    frames: Iterable[np.ndarray], target_height: int, target_width: int
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for frame in frames:
        frame_rgb = _ensure_rgb_uint8(frame)
        if frame_rgb.shape[:2] == (target_height, target_width):
            out.append(frame_rgb)
            continue
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame = pil_frame.resize((target_width, target_height), resample=Image.BILINEAR)
        out.append(_ensure_rgb_uint8(np.asarray(pil_frame)))
    return out


def _compute_scaled_shape(
    height: int, width: int, downscaled_long_side_px: int | None
) -> Tuple[int, int]:
    if downscaled_long_side_px is None:
        return height, width
    if downscaled_long_side_px <= 0:
        raise ValueError(
            f"Expected --downscaled-long-side-px > 0, got {downscaled_long_side_px}"
        )
    long_side = max(height, width)
    if long_side <= downscaled_long_side_px:
        return height, width
    scale = float(downscaled_long_side_px) / float(long_side)
    return (
        max(1, int(round(float(height) * scale))),
        max(1, int(round(float(width) * scale))),
    )


def _iter_chunk_bounds(total_frames: int, chunk_max_frames: int) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []
    if chunk_max_frames <= 0:
        return [(0, total_frames)]
    out: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_max_frames)
        out.append((start, end))
        start = end
    return out


def _discord(webhook_url: str | None, mention: str, content: str) -> None:
    """Send *content* to Discord if *webhook_url* is set, prefixing with *mention*."""
    if not webhook_url:
        return
    prefix = f"{mention.strip()} " if mention.strip() else ""
    _send_discord_message(webhook_url, f"{prefix}{content}")


def _send_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib_request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "curl/8.5.0",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"Discord webhook returned status {status}")
    except (urllib_error.URLError, RuntimeError) as exc:
        print(f"[warning] Failed to send Discord notification: {exc}")


def load_calibration_map(calibration_dir: Path) -> Dict[str, Calibration]:
    import pickle as pkl

    path_intrinsics = calibration_dir / "camera_intrinsics.pkl"
    path_extrinsics = calibration_dir / "transformations.pkl"
    if not path_intrinsics.exists() or not path_extrinsics.exists():
        raise FileNotFoundError(
            f"Missing calibration files under {calibration_dir}: "
            f"{path_intrinsics.name}, {path_extrinsics.name}"
        )

    with path_intrinsics.open("rb") as f:
        intr_data = pkl.load(f)
    with path_extrinsics.open("rb") as f:
        extr_data = pkl.load(f)

    extr_map: Dict[str, np.ndarray] = {}
    if isinstance(extr_data, list):
        for item in extr_data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                T = np.asarray(item[1], dtype=np.float64)
                if T.shape == (4, 4):
                    extr_map[name] = T

    calibs: Dict[str, Calibration] = {}
    for cam_name, intr_pair in intr_data.items():
        if cam_name not in extr_map:
            continue
        K, _dist = intr_pair
        calibs[cam_name] = Calibration(
            K=np.asarray(K, dtype=np.float64),
            dist=np.zeros((5,), dtype=np.float64),
            T_ee_camera=np.asarray(extr_map[cam_name], dtype=np.float64),
        )
    if not calibs:
        raise RuntimeError(f"No valid camera calibrations loaded from {calibration_dir}")
    return calibs


def quat_to_rotmat(quat: Sequence[float], order: str) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    if order == "xyzw":
        x, y, z, w = q
    else:
        w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_prev_joint_pose_from_ee(
    ee_pos: np.ndarray, ee_quat: np.ndarray, quaternion_order: str
) -> Tuple[np.ndarray, np.ndarray]:
    R_base_ee = quat_to_rotmat(ee_quat, order=quaternion_order)
    p_base_ee = np.asarray(ee_pos, dtype=np.float64)
    p_base_prev = p_base_ee - (R_base_ee @ EE_TRANSLATION_OFFSET)
    return R_base_ee, p_base_prev


def make_transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def project_points_camera_frame(
    points_camera: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pts_cam = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    visible = pts_cam[:, 2] > 1e-6
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img_pts, _ = cv2.projectPoints(pts_cam, rvec, tvec, K, dist)
    return img_pts.reshape(-1, 2), visible


def score_ee_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_ee_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int = 20,
) -> Tuple[int, int, int]:
    n = min(len(ee_pos_rows), len(ee_quat_rows), sample_n)
    in_frame = 0
    visible = 0
    for i in range(n):
        R_base_prev, p_base_prev = compute_prev_joint_pose_from_ee(
            ee_pos_rows[i], ee_quat_rows[i], quaternion_order
        )
        T_base_prev = make_transform(R_base_prev, p_base_prev)
        T_base_cam = T_base_prev @ T_ee_cam
        T_cam_base = np.linalg.inv(T_base_cam)
        p_ee_cam = (T_cam_base @ np.r_[ee_pos_rows[i], 1.0])[:3]
        img_pts, vis = project_points_camera_frame(np.asarray([p_ee_cam]), K, dist)
        if bool(vis[0]):
            visible += 1
            u, v = float(img_pts[0, 0]), float(img_pts[0, 1])
            if 0 <= u < width and 0 <= v < height:
                in_frame += 1
    return in_frame, visible, n


def score_base_cam_direction(
    ee_pos_rows: np.ndarray,
    T_base_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    sample_n: int = 20,
) -> Tuple[int, int, int]:
    n = min(len(ee_pos_rows), sample_n)
    T_cam_base = np.linalg.inv(T_base_cam)
    in_frame = 0
    visible = 0
    for i in range(n):
        p_ee_cam = (T_cam_base @ np.r_[ee_pos_rows[i], 1.0])[:3]
        img_pts, vis = project_points_camera_frame(np.asarray([p_ee_cam]), K, dist)
        if bool(vis[0]):
            visible += 1
            u, v = float(img_pts[0, 0]), float(img_pts[0, 1])
            if 0 <= u < width and 0 <= v < height:
                in_frame += 1
    return in_frame, visible, n


def choose_base_cam_direction(
    ee_pos_rows: np.ndarray,
    T_base_cam_raw: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    sample_n: int = 20,
) -> Tuple[np.ndarray, str]:
    raw_in, raw_vis, n = score_base_cam_direction(
        ee_pos_rows, T_base_cam_raw, K, dist, width, height, sample_n
    )
    inv_T = np.linalg.inv(T_base_cam_raw)
    inv_in, inv_vis, _ = score_base_cam_direction(
        ee_pos_rows, inv_T, K, dist, width, height, sample_n
    )
    if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
        return T_base_cam_raw, f"raw(T_base_camera) in_frame={raw_in}/{n}"
    return inv_T, f"inverse(T_base_camera) in_frame={inv_in}/{n}"


def choose_ee_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_ee_cam_raw: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int = 20,
) -> Tuple[np.ndarray, str]:
    raw_in, raw_vis, n = score_ee_cam_direction(
        ee_pos_rows,
        ee_quat_rows,
        T_ee_cam_raw,
        K,
        dist,
        width,
        height,
        quaternion_order,
        sample_n,
    )
    inv_T = np.linalg.inv(T_ee_cam_raw)
    inv_in, inv_vis, _ = score_ee_cam_direction(
        ee_pos_rows,
        ee_quat_rows,
        inv_T,
        K,
        dist,
        width,
        height,
        quaternion_order,
        sample_n,
    )
    if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
        return T_ee_cam_raw, f"raw(T_ee_camera) in_frame={raw_in}/{n}"
    return inv_T, f"inverse(T_ee_camera) in_frame={inv_in}/{n}"


def _transform_base_points_to_camera(
    points_base: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    camera_mode: str,
    quaternion_order: str,
    T_ee_cam: np.ndarray | None,
    T_cam_base_static: np.ndarray | None,
) -> np.ndarray:
    if camera_mode == "base_camera":
        assert T_cam_base_static is not None
        return (T_cam_base_static @ np.c_[points_base, np.ones((len(points_base), 1))].T).T[:, :3]

    assert T_ee_cam is not None
    R_base_prev, p_base_prev = compute_prev_joint_pose_from_ee(
        ee_pos, ee_quat, quaternion_order
    )
    T_base_prev = make_transform(R_base_prev, p_base_prev)
    T_base_cam = T_base_prev @ T_ee_cam
    T_cam_base = np.linalg.inv(T_base_cam)
    return (T_cam_base @ np.c_[points_base, np.ones((len(points_base), 1))].T).T[:, :3]


def project_ee_and_axes(
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    calib: Calibration,
    camera_mode: str,
    quaternion_order: str,
    axis_length_m: float,
    T_ee_cam: np.ndarray | None,
    T_cam_base_static: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_base_ee = quat_to_rotmat(ee_quat, order=quaternion_order)
    origin_base = np.asarray(ee_pos, dtype=np.float64)
    axis_dirs_base = R_base_ee @ np.eye(3, dtype=np.float64)
    axis_endpoints_base = origin_base[None, :] + axis_dirs_base.T * float(axis_length_m)
    world_points_base = np.vstack([origin_base[None, :], axis_endpoints_base])
    cam_points = _transform_base_points_to_camera(
        points_base=world_points_base,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        camera_mode=camera_mode,
        quaternion_order=quaternion_order,
        T_ee_cam=T_ee_cam,
        T_cam_base_static=T_cam_base_static,
    )
    pix, vis = project_points_camera_frame(cam_points, calib.K, calib.dist)
    return pix[0], pix[1:], vis


def project_ee_axis_aligned_roi_polygon(
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    calib: Calibration,
    camera_mode: str,
    quaternion_order: str,
    T_ee_cam: np.ndarray | None,
    T_cam_base_static: np.ndarray | None,
    ee_pix: np.ndarray,
    axis_pix: np.ndarray,
    vis: np.ndarray,
    neg_primary_m: float,
    pos_primary_m: float,
    neg_secondary_m: float,
    pos_secondary_m: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    axis_strength = np.full((3,), -1.0, dtype=np.float64)
    for axis_idx in range(3):
        if bool(vis[0]) and bool(vis[axis_idx + 1]):
            axis_strength[axis_idx] = np.linalg.norm(axis_pix[axis_idx] - ee_pix)
    order = list(np.argsort(-axis_strength))
    chosen_axes = [idx for idx in order if axis_strength[idx] > 1e-6]
    if len(chosen_axes) < 2:
        chosen_axes = [0, 1]
    top_two = [int(chosen_axes[0]), int(chosen_axes[1])]

    axis_dot_to_proj_x: Dict[int, float] = {}
    proj_x_vec = np.asarray(axis_pix[0] - ee_pix, dtype=np.float64)
    proj_x_norm = float(np.linalg.norm(proj_x_vec))
    for axis_idx in top_two:
        vec = np.asarray(axis_pix[axis_idx] - ee_pix, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if proj_x_norm > 1e-8 and norm > 1e-8:
            axis_dot_to_proj_x[axis_idx] = float(np.dot(vec, proj_x_vec) / (norm * proj_x_norm))
        else:
            axis_dot_to_proj_x[axis_idx] = float(vec[0] / norm) if norm > 1e-8 else -1.0
    top_two_sorted = sorted(top_two, key=lambda idx: axis_dot_to_proj_x[idx], reverse=True)
    primary_axis, secondary_axis = top_two_sorted[0], top_two_sorted[1]

    R_base_ee = quat_to_rotmat(ee_quat, order=quaternion_order)
    origin_base = np.asarray(ee_pos, dtype=np.float64)
    axis_dirs_base = R_base_ee @ np.eye(3, dtype=np.float64)
    u = axis_dirs_base[:, primary_axis]
    v = axis_dirs_base[:, secondary_axis]
    corners_base = np.vstack(
        [
            origin_base + (-float(neg_primary_m)) * u + (float(pos_secondary_m)) * v,
            origin_base + (-float(neg_primary_m)) * u + (-float(neg_secondary_m)) * v,
            origin_base + (float(pos_primary_m)) * u + (-float(neg_secondary_m)) * v,
            origin_base + (float(pos_primary_m)) * u + (float(pos_secondary_m)) * v,
        ]
    )

    cam_points = _transform_base_points_to_camera(
        points_base=corners_base,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        camera_mode=camera_mode,
        quaternion_order=quaternion_order,
        T_ee_cam=T_ee_cam,
        T_cam_base_static=T_cam_base_static,
    )
    pix, roi_vis = project_points_camera_frame(cam_points, calib.K, calib.dist)
    return pix, roi_vis, (primary_axis, secondary_axis)


def _normalize_box_xyxy(box_xyxy: Sequence[float], width: int, height: int) -> list[float]:
    box = np.asarray(box_xyxy, dtype=np.float32)
    if box.shape != (4,):
        raise ValueError(f"Expected [x_min,y_min,x_max,y_max], got {box_xyxy}")
    if ((box >= 0.0) & (box <= 1.0)).all():
        normalized = box
    else:
        normalized = np.array(
            [box[0] / width, box[1] / height, box[2] / width, box[3] / height], dtype=np.float32
        )
    x1, y1, x2, y2 = [float(v) for v in normalized]
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    normalized = np.array([x1, y1, x2, y2], dtype=np.float32)
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        raise ValueError(f"Bounding box is empty after clipping: {normalized.tolist()}")
    return normalized.tolist()


def _rotate_points_clockwise(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    x_new = (height - 1.0) - y
    y_new = x
    return np.stack([x_new, y_new], axis=1)


def _validate_mask_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    bool_mask = _to_binary_mask(mask)
    if bool_mask.shape == (height, width):
        return bool_mask
    raise RuntimeError(
        "Predicted mask shape mismatch. "
        f"Expected {(height, width)}, got {bool_mask.shape}."
    )


def _build_label_map(
    frame_obj_masks: Dict[int, np.ndarray] | None, width: int, height: int
) -> np.ndarray:
    label_map = np.zeros((height, width), dtype=np.uint8)
    if not frame_obj_masks:
        return label_map
    for obj_id in sorted(frame_obj_masks.keys()):
        if obj_id not in LABEL_COLORS_RGB:
            continue
        mask = _validate_mask_shape(frame_obj_masks[obj_id], width=width, height=height)
        label_map[mask] = np.uint8(obj_id)
    return label_map


def _render_segmentation_frame(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label, rgb in LABEL_COLORS_RGB.items():
        out[label_map == label] = np.asarray(rgb, dtype=np.uint8)
    return out


def _morphological_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    kernel_size = 2 * int(radius) + 1
    x = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None, ...]
    dilated = torch.nn.functional.max_pool2d(
        x, kernel_size=kernel_size, stride=1, padding=radius
    )
    closed = 1.0 - torch.nn.functional.max_pool2d(
        1.0 - dilated, kernel_size=kernel_size, stride=1, padding=radius
    )
    return (closed[0, 0] > 0.5).cpu().numpy()


def _apply_morphological_close_to_label_maps(
    label_maps: np.ndarray, radius: int
) -> np.ndarray:
    if radius <= 0 or label_maps.size == 0:
        return label_maps
    out = np.zeros_like(label_maps, dtype=np.uint8)
    label_ids = sorted(int(label) for label in LABEL_COLORS_RGB.keys() if int(label) != 0)
    for frame_idx in range(label_maps.shape[0]):
        frame_in = label_maps[frame_idx]
        frame_out = np.zeros_like(frame_in, dtype=np.uint8)
        for label_id in label_ids:
            mask = frame_in == label_id
            if not np.any(mask):
                continue
            closed_mask = _morphological_close(mask, radius=radius)
            frame_out[closed_mask] = np.uint8(label_id)
        out[frame_idx] = frame_out
    return out


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


def load_episode_annotation(annotation_path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = [
        "episode_id",
        "observation.state.cartesian_position",
        "observation.state.cartesian_orientation_quat",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in {annotation_path}: {missing}")
    episode_id = int(data["episode_id"])
    ee_pos_rows = np.asarray(data["observation.state.cartesian_position"], dtype=np.float64)
    ee_quat_rows = np.asarray(data["observation.state.cartesian_orientation_quat"], dtype=np.float64)
    return episode_id, ee_pos_rows, ee_quat_rows


def _run_text_prompt(
    text_predictor,
    chunk_frames: List[np.ndarray],
    prompt_obj_id: int,
    prompt_text: str,
    mask_label_id: int,
    projected_points_by_frame: Dict[int, Tuple[int, int]] | None = None,
    reprompt_every: int | None = None,
    reprompt_min_border_px: int = 0,
) -> Dict[int, Dict[int, np.ndarray]]:
    preprocessed_dir = Path(tempfile.mkdtemp(prefix="sam3_preprocessed_frames_"))
    for idx, frame_rgb in enumerate(chunk_frames):
        Image.fromarray(frame_rgb).save(preprocessed_dir / f"{idx:06d}.jpg", quality=95)

    start_response = text_predictor.handle_request(
        request=dict(type="start_session", resource_path=str(preprocessed_dir))
    )
    session_id = start_response["session_id"]

    outputs_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
    frame_h, frame_w = chunk_frames[0].shape[:2]
    try:
        interval = int(reprompt_every) if reprompt_every is not None else 0
        if interval <= 0:
            interval = len(chunk_frames)
        for window_start in range(0, len(chunk_frames), interval):
            window_len = min(interval, len(chunk_frames) - window_start)
            # Decide whether to reprompt at this window boundary.
            # The first window always gets a prompt. Subsequent windows are
            # skipped when the projected EE point is closer than
            # reprompt_min_border_px pixels to any frame border (or absent).
            is_first_window = window_start == 0
            do_reprompt = is_first_window
            if not do_reprompt:
                if projected_points_by_frame is None or reprompt_min_border_px <= 0:
                    do_reprompt = True
                else:
                    point_xy = projected_points_by_frame.get(window_start)
                    if point_xy is not None:
                        px, py = int(point_xy[0]), int(point_xy[1])
                        b = reprompt_min_border_px
                        do_reprompt = (
                            b <= px < frame_w - b and b <= py < frame_h - b
                        )
            if do_reprompt:
                _ = text_predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
                _ = text_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=window_start,
                        obj_id=prompt_obj_id,
                        text=prompt_text,
                    )
                )
            for stream_response in text_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                    propagation_direction="forward",
                    start_frame_index=window_start,
                    max_frame_num_to_track=window_len,
                )
            ):
                frame_idx = int(stream_response["frame_index"])
                out = stream_response["outputs"]
                scores = np.asarray(out["out_probs"], dtype=np.float32)
                masks = np.asarray(out["out_binary_masks"])
                if len(scores) == 0 or len(masks) == 0:
                    continue
                best_idx = int(np.argmax(scores))
                if projected_points_by_frame is not None:
                    point_xy = projected_points_by_frame.get(frame_idx)
                    if point_xy is not None:
                        px, py = int(point_xy[0]), int(point_xy[1])
                        candidate_indices: List[int] = []
                        for idx in range(len(masks)):
                            mask_arr = np.asarray(masks[idx])
                            while mask_arr.ndim > 2:
                                mask_arr = mask_arr[0]
                            if mask_arr.ndim != 2:
                                continue
                            h, w = mask_arr.shape[:2]
                            if px < 0 or py < 0 or px >= w or py >= h:
                                continue
                            if bool(mask_arr[py, px]):
                                candidate_indices.append(idx)
                        if candidate_indices:
                            candidate_scores = scores[np.asarray(candidate_indices, dtype=np.int64)]
                            best_idx = int(candidate_indices[int(np.argmax(candidate_scores))])
                        else:
                            # Fallback: no mask contains the projected EE point directly,
                            # so pick the mask whose centroid is closest to it.
                            dists: List[float] = []
                            for idx in range(len(masks)):
                                mask_arr = np.asarray(masks[idx])
                                while mask_arr.ndim > 2:
                                    mask_arr = mask_arr[0]
                                ys, xs = np.where(mask_arr)
                                if len(xs) == 0:
                                    dists.append(float("inf"))
                                else:
                                    cx, cy = float(xs.mean()), float(ys.mean())
                                    dists.append((cx - px) ** 2 + (cy - py) ** 2)
                            best_idx = int(np.argmin(np.asarray(dists, dtype=np.float64)))
                frame_store = outputs_per_frame.setdefault(frame_idx, {})
                frame_store[mask_label_id] = np.asarray(masks[best_idx])
    finally:
        try:
            _ = text_predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        finally:
            for jpg_path in preprocessed_dir.glob("*.jpg"):
                jpg_path.unlink(missing_ok=True)
            preprocessed_dir.rmdir()
    return outputs_per_frame


def _write_frames_to_temp_video(chunk_frames: List[np.ndarray], fps: float) -> Path:
    if not chunk_frames:
        raise ValueError("chunk_frames must not be empty")
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam3_chunk_video_"))
    tmp_video = tmp_dir / "chunk.mp4"
    media.write_video(
        str(tmp_video),
        [_ensure_rgb_uint8(frame) for frame in chunk_frames],
        fps=float(fps) if fps and fps > 0 else 30.0,
    )
    return tmp_video

def _run_box_prompt(
    box_predictor,
    chunk_frames: List[np.ndarray],
    prompt_obj_id: int,
    box_xyxy: Sequence[float],
    mask_label_id: int,
    output_fps: float,
) -> Dict[int, Dict[int, np.ndarray]]:
    chunk_video_path = _write_frames_to_temp_video(chunk_frames, fps=output_fps)
    inference_state = box_predictor.init_state(video_path=str(chunk_video_path))
    h, w = chunk_frames[0].shape[:2]
    norm_xyxy = _normalize_box_xyxy(box_xyxy=box_xyxy, width=w, height=h)
    print(f"norm_xyxy: {norm_xyxy}")

    _ = box_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=prompt_obj_id,
        box=np.asarray([norm_xyxy], dtype=np.float32),
    )

    outputs_per_frame: Dict[int, Dict[int, np.ndarray]] = {}
    for (
        frame_idx,
        _obj_ids,
        _low_res_masks,
        video_res_masks,
        obj_scores,
    ) in box_predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(chunk_frames),
        reverse=False,
        propagate_preflight=True,
    ):
        if video_res_masks is None:
            continue
        num_masks = int(video_res_masks.shape[0])
        if num_masks <= 0:
            continue
        best_mask_idx = 0
        if obj_scores is not None:
            score_tensor = obj_scores.detach().reshape(-1)
            if score_tensor.numel() >= num_masks:
                best_mask_idx = int(torch.argmax(score_tensor[:num_masks]).item())
        selected = video_res_masks[best_mask_idx]
        frame_store = outputs_per_frame.setdefault(int(frame_idx), {})
        frame_store[mask_label_id] = np.asarray(selected.detach().cpu().numpy())

    try:
        box_predictor.clear_all_points_in_video(inference_state)
    except Exception:
        pass
    chunk_video_path.unlink(missing_ok=True)
    chunk_video_path.parent.rmdir()
    return outputs_per_frame


def process_video(
    text_predictor,
    box_predictor,
    episode_id: int,
    view_index: int,
    video_path: Path,
    output_video_path: Path,
    output_mask_path: Path,
    prompts: Iterable[Dict[str, object]],
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    calib: Calibration | None,
    camera_name: str | None,
    quaternion_order: str,
    axis_length_m: float,
    roi_neg_primary_m: float,
    roi_pos_primary_m: float,
    roi_neg_secondary_m: float,
    roi_pos_secondary_m: float,
    output_fps: float,
    max_frames: int | None,
    wrist_priming_video_path: Path,
    downscaled_long_side_px: int | None,
    chunk_max_frames: int,
    morph_close_radius: int,
    apply_morph_close: bool,
    reprompt_min_border_px: int = 0,
) -> None:
    target_video_np = media.read_video(str(video_path))
    if target_video_np is None or len(target_video_np) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")
    source_frame_count = int(len(target_video_np))
    max_frame_num_to_track = (
        source_frame_count if max_frames is None else min(max_frames, source_frame_count)
    )
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV required: pip install opencv-python")

    original_frames: List[np.ndarray] = [
        _ensure_rgb_uint8(frame_rgb) for frame_rgb in target_video_np[:max_frame_num_to_track]
    ]
    if not original_frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    info = VideoInfo(
        width=int(original_frames[0].shape[1]),
        height=int(original_frames[0].shape[0]),
        fps=30.0,
        frame_count=len(original_frames),
    )

    target_frame_count = min(len(original_frames), len(ee_pos_rows), len(ee_quat_rows))
    if target_frame_count <= 0:
        raise RuntimeError(f"No overlapping frames between video and states: {video_path}")
    original_frames = original_frames[:target_frame_count]
    is_wrist_view = view_index == WRIST_VIEW_INDEX

    input_h, input_w = original_frames[0].shape[:2]
    if is_wrist_view:
        rotated_target_frames = [np.rot90(frame, k=-1) for frame in original_frames]
        rotated_h, rotated_w = rotated_target_frames[0].shape[:2]
        pre_h, pre_w = _compute_scaled_shape(rotated_h, rotated_w, downscaled_long_side_px)
        preprocessed_target_frames = _resize_frames_to_shape(rotated_target_frames, pre_h, pre_w)
    else:
        pre_h, pre_w = _compute_scaled_shape(input_h, input_w, downscaled_long_side_px)
        preprocessed_target_frames = _resize_frames_to_shape(original_frames, pre_h, pre_w)

    priming_frames: List[np.ndarray] = []
    if wrist_priming_video_path.exists():
        priming_video_np = media.read_video(str(wrist_priming_video_path))
        for frame_rgb in priming_video_np:
            priming_frames.append(np.rot90(_ensure_rgb_uint8(frame_rgb), k=1))
        if priming_frames:
            priming_frames = list(reversed(priming_frames))
            priming_frames = _resize_frames_to_shape(priming_frames, pre_h, pre_w)

    camera_mode = None
    T_ee_cam = None
    T_cam_base_static = None
    if calib is not None and camera_name is not None:
        camera_mode = CAMERA_EXTRINSIC_MODE.get(camera_name, "ee_camera")
        if camera_mode == "base_camera":
            T_base_cam, direction_mode = choose_base_cam_direction(
                ee_pos_rows=ee_pos_rows[:target_frame_count],
                T_base_cam_raw=calib.T_ee_camera,
                K=calib.K,
                dist=calib.dist,
                width=info.width,
                height=info.height,
                sample_n=20,
            )
            T_cam_base_static = np.linalg.inv(T_base_cam)
        else:
            T_ee_cam, direction_mode = choose_ee_cam_direction(
                ee_pos_rows=ee_pos_rows[:target_frame_count],
                ee_quat_rows=ee_quat_rows[:target_frame_count],
                T_ee_cam_raw=calib.T_ee_camera,
                K=calib.K,
                dist=calib.dist,
                width=info.width,
                height=info.height,
                quaternion_order=quaternion_order,
                sample_n=20,
            )
        print(f"[episode {episode_id} cam={camera_name}] extrinsic direction: {direction_mode}")

    label_maps = np.zeros((target_frame_count, info.height, info.width), dtype=np.uint8)
    chunk_limit = int(chunk_max_frames) if int(chunk_max_frames) > 0 else target_frame_count
    chunk_bounds = _iter_chunk_bounds(target_frame_count, chunk_limit)

    global_first_target_frame = preprocessed_target_frames[0]
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        chunk_target_frames = preprocessed_target_frames[chunk_start:chunk_end]
        merged_outputs: Dict[int, Dict[int, np.ndarray]] = {}

        for prompt_spec in prompts:
            prompt_obj_id = int(prompt_spec["obj_id"])
            mask_label_id = int(prompt_spec.get("label_id", prompt_obj_id))
            select_mask_based_on_projected_ee = bool(prompt_spec.get("select_mask_based_on_projected_ee", False))
            use_text = "text" in prompt_spec
            use_projection = bool(prompt_spec.get("box_from_projection", False))
            use_explicit_box = "box" in prompt_spec
            use_priming_video = bool(prompt_spec.get("use_priming_video", False))
            prompt_priming = priming_frames if (use_priming_video and priming_frames) else []
            chunk_frames = [*prompt_priming, *chunk_target_frames]
            frame_index_offset = len(prompt_priming)

            prompt_outputs: Dict[int, Dict[int, np.ndarray]]
            if use_text:
                prompt_text = str(prompt_spec["text"])
                reprompt_every = (
                    int(prompt_spec["reprompt_every"])
                    if "reprompt_every" in prompt_spec
                    else None
                )
                # For text prompts, always anchor from the first frame of the full video,
                # then track into this chunk's frames.
                chunk_frames = [global_first_target_frame, *prompt_priming, *chunk_target_frames]
                frame_index_offset = 1 + len(prompt_priming)
                projected_points_by_frame: Dict[int, Tuple[int, int]] | None = None
                if (
                    select_mask_based_on_projected_ee 
                    and "hand" in prompt_text.lower()
                    and calib is not None
                    and camera_mode is not None
                ):
                    projected_points_by_frame = {}
                    if is_wrist_view:
                        src_w, src_h = info.height, info.width
                    else:
                        src_w, src_h = info.width, info.height
                    scale_x = float(pre_w) / float(src_w)
                    scale_y = float(pre_h) / float(src_h)
                    for target_local_idx in range(chunk_end - chunk_start):
                        global_frame_idx = chunk_start + target_local_idx
                        ee_pix, _axis_pix, vis = project_ee_and_axes(
                            ee_pos=ee_pos_rows[global_frame_idx],
                            ee_quat=ee_quat_rows[global_frame_idx],
                            calib=calib,
                            camera_mode=camera_mode,
                            quaternion_order=quaternion_order,
                            axis_length_m=axis_length_m,
                            T_ee_cam=T_ee_cam,
                            T_cam_base_static=T_cam_base_static,
                        )
                        if not bool(vis[0]):
                            continue
                        ee_point = ee_pix.copy()
                        if is_wrist_view:
                            ee_point = _rotate_points_clockwise(
                                np.asarray([ee_point], dtype=np.float64),
                                width=info.width,
                                height=info.height,
                            )[0]
                        ee_point[0] *= scale_x
                        ee_point[1] *= scale_y
                        px = int(np.round(ee_point[0]))
                        py = int(np.round(ee_point[1]))
                        if 0 <= px < pre_w and 0 <= py < pre_h:
                            projected_points_by_frame[target_local_idx + frame_index_offset] = (
                                px,
                                py,
                            )
                prompt_outputs = _run_text_prompt(
                    text_predictor=text_predictor,
                    chunk_frames=chunk_frames,
                    prompt_obj_id=prompt_obj_id,
                    prompt_text=prompt_text,
                    mask_label_id=mask_label_id,
                    projected_points_by_frame=projected_points_by_frame,
                    reprompt_every=reprompt_every,
                    reprompt_min_border_px=reprompt_min_border_px,
                )
            else:
                box_xyxy: Sequence[float]
                if use_projection:
                    if calib is None or camera_mode is None:
                        raise RuntimeError(
                            f"Prompt asks for box_from_projection but no calibration for view={view_index}"
                        )
                    global_frame_idx = chunk_start
                    ee_pix, axis_pix, vis = project_ee_and_axes(
                        ee_pos=ee_pos_rows[global_frame_idx],
                        ee_quat=ee_quat_rows[global_frame_idx],
                        calib=calib,
                        camera_mode=camera_mode,
                        quaternion_order=quaternion_order,
                        axis_length_m=axis_length_m,
                        T_ee_cam=T_ee_cam,
                        T_cam_base_static=T_cam_base_static,
                    )
                    roi_polygon, roi_vis, _axes = project_ee_axis_aligned_roi_polygon(
                        ee_pos=ee_pos_rows[global_frame_idx],
                        ee_quat=ee_quat_rows[global_frame_idx],
                        calib=calib,
                        camera_mode=camera_mode,
                        quaternion_order=quaternion_order,
                        T_ee_cam=T_ee_cam,
                        T_cam_base_static=T_cam_base_static,
                        ee_pix=ee_pix,
                        axis_pix=axis_pix,
                        vis=vis,
                        neg_primary_m=roi_neg_primary_m,
                        pos_primary_m=roi_pos_primary_m,
                        neg_secondary_m=roi_neg_secondary_m,
                        pos_secondary_m=roi_pos_secondary_m,
                    )
                    if not bool(np.all(roi_vis)):
                        raise RuntimeError(
                            f"Projected ROI not fully visible for episode={episode_id} frame={global_frame_idx}"
                        )
                    if is_wrist_view:
                        roi_polygon = _rotate_points_clockwise(
                            roi_polygon, width=info.width, height=info.height
                        )
                        src_w, src_h = info.height, info.width
                    else:
                        src_w, src_h = info.width, info.height
                    scale_x = float(pre_w) / float(src_w)
                    scale_y = float(pre_h) / float(src_h)
                    roi_scaled = roi_polygon.copy()
                    roi_scaled[:, 0] *= scale_x
                    roi_scaled[:, 1] *= scale_y
                    x1 = float(np.min(roi_scaled[:, 0]))
                    y1 = float(np.min(roi_scaled[:, 1]))
                    x2 = float(np.max(roi_scaled[:, 0]))
                    y2 = float(np.max(roi_scaled[:, 1]))
                    box_xyxy = [x1, y1, x2, y2]
                elif use_explicit_box:
                    box_xyxy = list(prompt_spec["box"])  # type: ignore[arg-type]
                else:
                    raise ValueError(
                        f"Prompt must have one of text/box/box_from_projection: {prompt_spec}"
                    )

                prompt_outputs = _run_box_prompt(
                    box_predictor=box_predictor,
                    chunk_frames=chunk_frames,
                    prompt_obj_id=prompt_obj_id,
                    box_xyxy=box_xyxy,
                    mask_label_id=mask_label_id,
                    output_fps=output_fps,
                )

            for sam3_frame_idx, frame_obj_map in prompt_outputs.items():
                target_local_idx = sam3_frame_idx - frame_index_offset
                if target_local_idx < 0 or target_local_idx >= (chunk_end - chunk_start):
                    continue
                target_frame_map = merged_outputs.setdefault(target_local_idx, {})
                for label_id, mask in frame_obj_map.items():
                    target_frame_map[int(label_id)] = np.asarray(mask)

        for local_idx in range(chunk_end - chunk_start):
            frame_obj_masks = merged_outputs.get(local_idx)
            frame_label_map = _build_label_map(frame_obj_masks=frame_obj_masks, width=pre_w, height=pre_h)
            if is_wrist_view:
                frame_label_map = np.rot90(frame_label_map, k=1)
            if frame_label_map.shape != (info.height, info.width):
                frame_label_map = np.asarray(
                    Image.fromarray(frame_label_map).resize((info.width, info.height), resample=Image.NEAREST),
                    dtype=np.uint8,
                )
            label_maps[chunk_start + local_idx] = frame_label_map

        if len(chunk_bounds) > 1:
            print(
                f"[episode {episode_id} view={view_index}] "
                f"chunk {chunk_idx + 1}/{len(chunk_bounds)} done frames {chunk_start}-{chunk_end - 1}"
            )

    if apply_morph_close and morph_close_radius > 0:
        label_maps = _apply_morphological_close_to_label_maps(
            label_maps, radius=morph_close_radius
        )

    output_frames = [
        _render_segmentation_frame(label_maps[frame_idx]) for frame_idx in range(target_frame_count)
    ]
    media.write_video(
        str(output_video_path),
        output_frames,
        fps=float(output_fps) if output_fps and output_fps > 0 else float(info.fps),
    )

    np.save(output_mask_path, label_maps)
    print(f"[done] video={video_path}")
    print(f"[done] segmentation video: {output_video_path}")
    print(f"[done] segmentation mask: {output_mask_path}")
    print(f"[done] frames={target_frame_count} labels={np.unique(label_maps).tolist()}")


def _free_vram_per_gpu_gb() -> List[Tuple[int, float]]:
    """Return [(gpu_index, free_gb), ...] for every visible CUDA device."""
    result = []
    n = torch.cuda.device_count()
    for i in range(n):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        result.append((i, free_bytes / 1024 ** 3))
    return result


def wait_for_vram(min_free_gb: float, poll_interval_s: float = 30.0) -> int:
    """Block until at least one GPU has >= *min_free_gb* GB of free VRAM.

    Returns the index of the GPU that satisfied the requirement.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices found; cannot check VRAM.")

    while True:
        gpu_stats = _free_vram_per_gpu_gb()
        status = "  ".join(f"gpu{i}: {gb:.1f} GB free" for i, gb in gpu_stats)
        best_idx, best_free = max(gpu_stats, key=lambda x: x[1])
        print(f"[vram-check] {status}")
        if best_free >= min_free_gb:
            print(
                f"[vram-check] gpu{best_idx} has {best_free:.1f} GB free "
                f"(>= {min_free_gb} GB required). Proceeding."
            )
            return best_idx
        print(
            f"[vram-check] Not enough free VRAM (need {min_free_gb} GB, "
            f"best is {best_free:.1f} GB on gpu{best_idx}). "
            f"Retrying in {poll_interval_s:.0f}s..."
        )
        time.sleep(poll_interval_s)


def rotate_video_180_inplace(video_path: Path, target_fps: float | None = None) -> None:
    assert cv2 is not None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for rotation: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps = float(target_fps) if target_fps is not None and target_fps > 0 else source_fps
    if fps <= 0:
        fps = 30.0

    temp_path = video_path.with_name(
        f"{video_path.stem}.rot180.{uuid.uuid4().hex[:8]}{video_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create temporary rotated video: {temp_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.rotate(frame, cv2.ROTATE_180))
    finally:
        cap.release()
        writer.release()

    temp_path.replace(video_path)


def main() -> None:
    args = parse_args()
    run_started_s = time.time()
    dataset_root = args.dataset_root
    annotation_dir = dataset_root / "annotation"
    videos_dir = dataset_root / "videos"
    processed_videos = 0
    run_failed = False

    if not annotation_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")

    if args.episode_ids is None:
        episode_ids = discover_episode_ids(annotation_dir=annotation_dir)
    else:
        parsed = _parse_episode_id_tokens(args.episode_ids)
        episode_ids = sorted(parsed)

    rotate_view1_episode_ids = _parse_episode_id_tokens(args.rotate_view1_episode_ids)
    selected_views = _selected_views(args.view_mode)
    calibration_map = load_calibration_map(args.calibration_dir)

    if args.min_free_vram_gb is not None:
        wait_for_vram(
            min_free_gb=args.min_free_vram_gb,
            poll_interval_s=args.vram_poll_interval,
        )

    sam3_model = build_sam3_video_model()
    box_predictor = sam3_model.tracker
    box_predictor.backbone = sam3_model.detector.backbone
    text_predictor = build_sam3_video_predictor()

    _discord(
        args.discord_webhook_url,
        args.discord_mention,
        f"final_extract_segmentation_masks.py STARTED | "
        f"dataset={dataset_root.name} | episodes={len(episode_ids)} | "
        f"views={args.view_mode}",
    )

    try:
        for requested_episode_id in episode_ids:
            episode_dir = videos_dir / str(requested_episode_id)
            if not episode_dir.exists():
                print(f"[warning] Missing episode directory: {episode_dir}")
                continue
            annotation_path = annotation_dir / f"{requested_episode_id}.json"
            if not annotation_path.exists():
                print(f"[warning] Missing annotation file: {annotation_path}")
                continue

            ann_episode_id, ee_pos_rows, ee_quat_rows = load_episode_annotation(annotation_path)
            episode_id = ann_episode_id if ann_episode_id != requested_episode_id else requested_episode_id
            if ann_episode_id != requested_episode_id:
                print(
                    f"[warning] Episode mismatch: id list={requested_episode_id} "
                    f"annotation={ann_episode_id}; using annotation id."
                )

            for view_index in sorted(selected_views):
                prompts = VIEW_PROMPTS.get(view_index)
                if not prompts:
                    print(f"[warning] No prompts configured for view={view_index}; skipping.")
                    continue
                input_video_path = episode_dir / f"{view_index}_rgb.mp4"
                if not input_video_path.exists():
                    print(f"[warning] Missing input video: {input_video_path}")
                    continue
                if view_index == WRIST_VIEW_INDEX and (
                    requested_episode_id in rotate_view1_episode_ids
                    or episode_id in rotate_view1_episode_ids
                ):
                    print(
                        f"[episode {episode_id}] rotating {input_video_path.name} by 180 degrees"
                    )
                    rotate_video_180_inplace(input_video_path, target_fps=args.output_fps)

                output_video_path = episode_dir / f"{view_index}_segmentation.mp4"
                output_mask_path = episode_dir / f"{view_index}_segmentation_mask.npy"
                camera_name = VIDEO_INDEX_TO_CAMERA.get(view_index)
                calib = calibration_map.get(camera_name) if camera_name is not None else None
                if calib is None:
                    print(
                        f"[warning] No calibration for view={view_index} camera={camera_name}; "
                        "projection prompts may fail."
                    )
                print(
                    f"[episode {episode_id}] processing view={view_index} "
                    f"video={input_video_path.name}"
                )
                _discord(
                    args.discord_webhook_url,
                    args.discord_mention,
                    f"[{processed_videos + 1}/{len(episode_ids) * len(selected_views)}] "
                    f"processing episode={episode_id} view={view_index} "
                    f"({input_video_path.name})",
                )
                process_video(
                    text_predictor=text_predictor,
                    box_predictor=box_predictor,
                    episode_id=episode_id,
                    view_index=view_index,
                    video_path=input_video_path,
                    output_video_path=output_video_path,
                    output_mask_path=output_mask_path,
                    prompts=prompts,
                    ee_pos_rows=ee_pos_rows,
                    ee_quat_rows=ee_quat_rows,
                    calib=calib,
                    camera_name=camera_name,
                    quaternion_order=args.quaternion_order,
                    axis_length_m=args.axis_length_m,
                    roi_neg_primary_m=args.roi_neg_primary_m,
                    roi_pos_primary_m=args.roi_pos_primary_m,
                    roi_neg_secondary_m=args.roi_neg_secondary_m,
                    roi_pos_secondary_m=args.roi_pos_secondary_m,
                    output_fps=args.output_fps,
                    max_frames=args.max_frames,
                    wrist_priming_video_path=args.wrist_priming_video_path,
                    downscaled_long_side_px=args.downscaled_long_side_px,
                    chunk_max_frames=args.chunk_max_frames,
                    morph_close_radius=args.morph_close_radius,
                    apply_morph_close=_morph_close_enabled_for_view(
                        view_index=view_index,
                        morph_close_views=args.morph_close_views,
                    ),
                    reprompt_min_border_px=args.reprompt_min_border_px,
                )
                processed_videos += 1
    except Exception:
        run_failed = True
        raise
    finally:
        text_predictor.shutdown()
        elapsed_s = int(time.time() - run_started_s)
        status_text = "FAILED" if run_failed else "DONE"
        _discord(
            args.discord_webhook_url,
            args.discord_mention,
            f"final_extract_segmentation_masks.py {status_text} | "
            f"processed_videos={processed_videos} | elapsed={elapsed_s}s",
        )


if __name__ == "__main__":
    main()
