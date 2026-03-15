#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - dependency guard
    cv2 = None

from sam3.model_builder import build_sam3_video_predictor


DEFAULT_DATASET_ROOT = Path(
    # "/data/sam3_based_labeling_pipeline/assets/test_le_robot_dataset"
    # "/data/Ctrl-World/datasets/red_cube_not_on_red_ramp_real"
    "/data/Ctrl-World/datasets/large_real_dataset"
)
DEFAULT_CALIBRATION_DIR = Path(
    "/data/sam3_based_labeling_pipeline/assets/calibration_params"
)
DEFAULT_WRIST_PRIMING_VIDEO_PATH = Path(
    "/data/Ctrl-World/datasets/initial_hand_motion/videos/6/1_rgb.mp4"
)
WRIST_CLICK_POINT_XY = (270, 20)
WRIST_VIEW_INDEX = 1
THIRD_VIEW_INDEX = 2

# View index -> prompt list.
# `obj_id` is sent to SAM3 as the prompt object id.
# `label_id` is written into the final segmentation mask.
VIEW_PROMPTS = {
    0: [
        {"obj_id": 0, "label_id": 1, "text": "the hand"},
        {"obj_id": 1, "label_id": 2, "text": "red dice"},
    ],
    1: [
        {"obj_id": 0, "label_id": 1, "text": "robotic hand"},
        {"obj_id": 1, "label_id": 2, "text": "red dice"},
    ],
}

# Label id -> RGB color.
LABEL_COLORS_RGB = {
    1: (0, 255, 0),  # robot hand
    2: (255, 0, 0),  # red dice
}

# Video stream index inside episode folder -> calibration camera name.
VIDEO_INDEX_TO_CAMERA = {
    0: "oakd_side_view",
    1: "oakd_wrist_view",
}

# How to interpret camera extrinsics.
CAMERA_EXTRINSIC_MODE = {
    "oakd_side_view": "base_camera",
    "oakd_wrist_view": "ee_camera",
}

# Offset used by the dataset between previous joint and EE.
EE_TRANSLATION_OFFSET = np.array([0.13, 0.0, 0.07], dtype=np.float64)


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
        "--recompute-wrist-episode-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Episode ids/ranges to recompute only wrist view (view=1). "
            "Example: --recompute-wrist-episode-ids 10-20 24"
        ),
    )
    parser.add_argument(
        "--recompute-third-episode-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Episode ids/ranges to recompute only third view (view=2). "
            "Example: --recompute-third-episode-ids 30-40"
        ),
    )
    parser.add_argument(
        "--recompute-both-episode-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Episode ids/ranges to recompute both wrist (view=1) and third (view=2). "
            "Example: --recompute-both-episode-ids 50-60"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum frames to track per video.",
    )
    parser.add_argument(
        "--chunk-max-frames",
        type=int,
        default=500,
        help=(
            "Maximum number of target frames to process per SAM3 chunk. "
            "Use <= 0 to disable chunking and process all frames in one pass."
        ),
    )
    parser.add_argument(
        "--downscaled-long-side-px",
        type=int,
        default=1008, # best results with this value
        help=(
            "Optional maximum pixel size for the larger side of preprocessed frames fed to SAM3. "
            "Frames are resized with preserved aspect ratio only when their larger side exceeds this value."
        ),
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=5.0,
        help="FPS for saved segmentation videos.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing camera_intrinsics.pkl and transformations.pkl.",
    )
    parser.add_argument(
        "--wrist-priming-video-path",
        type=Path,
        default=DEFAULT_WRIST_PRIMING_VIDEO_PATH,
        help="Priming video used for wrist-view SAM3 warm-up.",
    )
    parser.add_argument(
        "--quaternion-order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion component ordering in annotation orientation arrays.",
    )
    parser.add_argument(
        "--debug-projection-images",
        type=int,
        default=0,
        help="Number of sampled debug images to save per processed video (0 to disable).",
    )
    parser.add_argument(
        "--axis-length-m",
        type=float,
        default=0.05,
        help="Axis length in meters for projected EE coordinate frame rendering.",
    )
    parser.add_argument(
        "--roi-neg-primary-m",
        type=float,
        default=0.05,
        help="ROI extent in negative primary projected-axis direction (meters).",
    )
    parser.add_argument(
        "--roi-pos-primary-m",
        type=float,
        default=0.12,
        help="ROI extent in positive primary projected-axis direction (meters).",
    )
    parser.add_argument(
        "--roi-neg-secondary-m",
        type=float,
        default=0.12,
        help="ROI extent in negative secondary projected-axis direction (meters).",
    )
    parser.add_argument(
        "--roi-pos-secondary-m",
        type=float,
        default=0.12,
        help="ROI extent in positive secondary projected-axis direction (meters).",
    )
    parser.add_argument(
        "--box-soft-margin-px",
        type=int,
        default=8,
        help=(
            "Soft boundary around projected ROI box in pixels. Hand pixels within this "
            "margin outside the box are still kept."
        ),
    )
    parser.add_argument(
        "--hand-label-id",
        type=int,
        default=1,
        help="Label id for hand mask; ROI box filtering is applied only to this label.",
    )
    parser.add_argument(
        "--wrist-roi-neg-primary-m",
        type=float,
        default=None,
        help="Optional wrist override: negative primary projected-axis extent (meters).",
    )
    parser.add_argument(
        "--wrist-roi-pos-primary-m",
        type=float,
        default=None,
        help="Optional wrist override: positive primary projected-axis extent (meters).",
    )
    parser.add_argument(
        "--wrist-roi-neg-secondary-m",
        type=float,
        default=None,
        help="Optional wrist override: negative secondary projected-axis extent (meters).",
    )
    parser.add_argument(
        "--wrist-roi-pos-secondary-m",
        type=float,
        default=None,
        help="Optional wrist override: positive secondary projected-axis extent (meters).",
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


def _to_binary_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_arr = np.asarray(mask)
    while mask_arr.ndim > 2:
        mask_arr = mask_arr[0]
    return mask_arr > 0.0


def _episode_matches(ids: set[int], requested_episode_id: int, annotation_episode_id: int) -> bool:
    return requested_episode_id in ids or annotation_episode_id in ids


def _views_to_recompute_for_episode(
    requested_episode_id: int,
    annotation_episode_id: int,
    recompute_wrist_episode_ids: set[int],
    recompute_third_episode_ids: set[int],
    recompute_both_episode_ids: set[int],
) -> tuple[set[int], bool]:
    recompute_mode = bool(
        recompute_wrist_episode_ids
        or recompute_third_episode_ids
        or recompute_both_episode_ids
    )
    if not recompute_mode:
        return set(VIEW_PROMPTS.keys()), False

    selected_views: set[int] = set()
    if _episode_matches(recompute_both_episode_ids, requested_episode_id, annotation_episode_id):
        selected_views.update({WRIST_VIEW_INDEX, THIRD_VIEW_INDEX})
    if _episode_matches(recompute_wrist_episode_ids, requested_episode_id, annotation_episode_id):
        selected_views.add(WRIST_VIEW_INDEX)
    if _episode_matches(recompute_third_episode_ids, requested_episode_id, annotation_episode_id):
        selected_views.add(THIRD_VIEW_INDEX)
    return selected_views, True


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
    resized: List[np.ndarray] = []
    for frame in frames:
        frame_rgb = _ensure_rgb_uint8(frame)
        if frame_rgb.shape[:2] == (target_height, target_width):
            resized.append(frame_rgb)
            continue
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame = pil_frame.resize((target_width, target_height), resample=Image.BILINEAR)
        resized.append(_ensure_rgb_uint8(np.asarray(pil_frame)))
    return resized


def _compute_scaled_shape(
    height: int, width: int, downscaled_long_side_px: int | None
) -> Tuple[int, int]:
    if downscaled_long_side_px is None:
        return height, width
    if downscaled_long_side_px <= 0:
        raise ValueError(
            f"Expected --downscaled-long-side-px to be positive, got {downscaled_long_side_px}"
        )
    long_side = max(height, width)
    if long_side <= downscaled_long_side_px:
        return height, width
    scale = float(downscaled_long_side_px) / float(long_side)
    scaled_h = max(1, int(round(float(height) * scale)))
    scaled_w = max(1, int(round(float(width) * scale)))
    return scaled_h, scaled_w


def _iter_chunk_bounds(total_frames: int, chunk_max_frames: int) -> List[Tuple[int, int]]:
    if total_frames <= 0:
        return []
    if chunk_max_frames <= 0:
        return [(0, total_frames)]
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = min(total_frames, start + chunk_max_frames)
        bounds.append((start, end))
        start = end
    return bounds


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
        K, _dist_from_file = intr_pair
        K = np.asarray(K, dtype=np.float64)
        dist = np.zeros((5,), dtype=np.float64)
        calibs[cam_name] = Calibration(K=K, dist=dist, T_ee_camera=extr_map[cam_name])

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
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    quaternion_order: str,
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
    points_camera: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
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
        p_ee_base = np.asarray(ee_pos_rows[i], dtype=np.float64)
        p_ee_cam = (T_cam_base @ np.r_[p_ee_base, 1.0])[:3]
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
        p_ee_base = np.asarray(ee_pos_rows[i], dtype=np.float64)
        p_ee_cam = (T_cam_base @ np.r_[p_ee_base, 1.0])[:3]
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

    if camera_mode == "base_camera":
        assert T_cam_base_static is not None
        cam_points = (T_cam_base_static @ np.c_[world_points_base, np.ones((4, 1))].T).T[:, :3]
    else:
        assert T_ee_cam is not None
        R_base_prev, p_base_prev = compute_prev_joint_pose_from_ee(
            ee_pos, ee_quat, quaternion_order
        )
        T_base_prev = make_transform(R_base_prev, p_base_prev)
        T_base_cam = T_base_prev @ T_ee_cam
        T_cam_base = np.linalg.inv(T_base_cam)
        cam_points = (T_cam_base @ np.c_[world_points_base, np.ones((4, 1))].T).T[:, :3]

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
    # Pick the two EE axes that are most visible in the image
    # (largest projected vector magnitudes from EE origin).
    axis_strength = np.full((3,), -1.0, dtype=np.float64)
    for axis_idx in range(3):
        if bool(vis[0]) and bool(vis[axis_idx + 1]):
            axis_strength[axis_idx] = np.linalg.norm(axis_pix[axis_idx] - ee_pix)
    order = list(np.argsort(-axis_strength))
    chosen_axes = [idx for idx in order if axis_strength[idx] > 1e-6]
    if len(chosen_axes) < 2:
        # Fallback for degenerate/occluded cases.
        chosen_axes = [0, 1]
    top_two = [int(chosen_axes[0]), int(chosen_axes[1])]

    # Primary axis: highest dot product with the projected EE X-axis.
    # Compare only among the two strongest projected axes.
    axis_dot_to_proj_x: Dict[int, float] = {}
    proj_x_vec = np.asarray(axis_pix[0] - ee_pix, dtype=np.float64)
    proj_x_norm = float(np.linalg.norm(proj_x_vec))
    for axis_idx in top_two:
        vec = np.asarray(axis_pix[axis_idx] - ee_pix, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if proj_x_norm > 1e-8 and norm > 1e-8:
            axis_dot_to_proj_x[axis_idx] = float(np.dot(vec, proj_x_vec) / (norm * proj_x_norm))
        else:
            # Fallback when projected x-axis is degenerate: use image-space x component.
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
    pix, vis = project_points_camera_frame(cam_points, calib.K, calib.dist)
    return pix, vis, (primary_axis, secondary_axis)


def rasterize_hull_mask(
    width: int,
    height: int,
    points_xy: np.ndarray,
) -> np.ndarray:
    pts = np.round(np.asarray(points_xy, dtype=np.float64)).astype(np.int32)
    canvas = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.fillConvexPoly(canvas, hull, 1)
    return canvas.astype(bool)


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


def draw_ee_overlay(
    frame_rgb: np.ndarray,
    ee_pix: np.ndarray,
    axis_pix: np.ndarray,
    vis: np.ndarray,
    roi_polygon: np.ndarray | None = None,
    roi_polygon_vis: np.ndarray | None = None,
) -> np.ndarray:
    out_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    x, y = int(round(float(ee_pix[0]))), int(round(float(ee_pix[1])))
    cv2.circle(out_bgr, (x, y), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        out_bgr,
        (x, y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    axis_colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # x,y,z
    for axis_idx in range(3):
        if not bool(vis[0]) or not bool(vis[axis_idx + 1]):
            continue
        ex = int(round(float(axis_pix[axis_idx, 0])))
        ey = int(round(float(axis_pix[axis_idx, 1])))
        cv2.line(out_bgr, (x, y), (ex, ey), axis_colors_bgr[axis_idx], 2, cv2.LINE_AA)
        cv2.circle(out_bgr, (ex, ey), 3, axis_colors_bgr[axis_idx], -1, cv2.LINE_AA)
    if (
        roi_polygon is not None
        and roi_polygon_vis is not None
        and len(roi_polygon) >= 3
        and bool(np.all(roi_polygon_vis))
    ):
        poly = np.round(np.asarray(roi_polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out_bgr, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


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


def _write_debug_segmentation_video(
    label_maps: np.ndarray, output_path: Path, fps: float
) -> None:
    if label_maps.size == 0:
        return
    h, w = label_maps.shape[1], label_maps.shape[2]
    writer = _create_video_writer(path=output_path, width=w, height=h, fps=fps)
    try:
        for frame_label_map in label_maps:
            frame_rgb = _render_segmentation_frame(frame_label_map.astype(np.uint8, copy=False))
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


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
    mask_label_id: int,
    max_frame_num_to_track: int,
    ee_target_pix_by_frame: Dict[int, np.ndarray] | None = None,
    required_point_xy: tuple[int, int] | None = None,
    start_frame_index: int | None = 0,
    propagation_direction: str = "both",
) -> Dict[int, Dict[str, np.ndarray]]:
    ee_candidate_radius_px = 5.0

    def _min_sq_dist_mask_to_point(mask_arr: np.ndarray, point_xy: np.ndarray) -> float:
        mask_bool = _to_binary_mask(mask_arr)
        ys, xs = np.nonzero(mask_bool)
        if xs.size == 0:
            return float("inf")
        px, py = float(point_xy[0]), float(point_xy[1])
        dx = xs.astype(np.float64) - px
        dy = ys.astype(np.float64) - py
        return float(np.min(dx * dx + dy * dy))

    outputs_per_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for stream_response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction=propagation_direction,
            start_frame_index=start_frame_index,
            max_frame_num_to_track=max_frame_num_to_track,
        )
    ):
        frame_idx = int(stream_response["frame_index"])
        out = stream_response["outputs"]

        scores = np.asarray(out["out_probs"], dtype=np.float32)
        masks = np.asarray(out["out_binary_masks"])
        if len(scores) == 0 or len(masks) == 0:
            continue

        if required_point_xy is not None:
            req_x, req_y = int(required_point_xy[0]), int(required_point_xy[1])
            selected_idx: int | None = None
            for idx, mask in enumerate(masks):
                mask_bool = _to_binary_mask(mask)
                h, w = mask_bool.shape[:2]
                if 0 <= req_x < w and 0 <= req_y < h and mask_bool[req_y, req_x]:
                    selected_idx = idx
                    break
            if selected_idx is None:
                continue
            best_idx = int(selected_idx)
            frame_store = outputs_per_frame.setdefault(frame_idx, {})
            frame_store[mask_label_id] = np.asarray(masks[best_idx])
            continue

        target_pix = (
            ee_target_pix_by_frame.get(frame_idx)
            if ee_target_pix_by_frame is not None
            else None
        )
        if target_pix is not None:
            min_sq_dists = np.asarray(
                [_min_sq_dist_mask_to_point(mask, target_pix) for mask in masks],
                dtype=np.float64,
            )
            in_range = min_sq_dists <= (ee_candidate_radius_px * ee_candidate_radius_px)
            if bool(np.any(in_range)):
                candidate_indices = np.nonzero(in_range)[0]
                best_local = int(np.argmax(scores[candidate_indices]))
                best_idx = int(candidate_indices[best_local])
            else:
                best_idx = int(np.argmax(scores))
        else:
            best_idx = int(np.argmax(scores))
        frame_store = outputs_per_frame.setdefault(frame_idx, {})
        frame_store[mask_label_id] = np.asarray(masks[best_idx])

    return outputs_per_frame


def _merge_outputs(
    dst: Dict[int, Dict[int, np.ndarray]],
    src: Dict[int, Dict[int, np.ndarray]],
) -> None:
    for frame_idx, src_obj_map in src.items():
        frame_map = dst.setdefault(frame_idx, {})
        for obj_id, mask in src_obj_map.items():
            frame_map[int(obj_id)] = np.asarray(mask)


def _validate_mask_shape(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    bool_mask = _to_binary_mask(mask)
    if bool_mask.shape == (height, width):
        return bool_mask
    raise RuntimeError(
        "Predicted mask shape mismatch. "
        f"Expected {(height, width)}, got {bool_mask.shape}."
    )


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
        mask = _validate_mask_shape(frame_obj_masks[obj_id], width=width, height=height)
        label_map[mask] = np.uint8(obj_id)
    return label_map


def _render_segmentation_frame(label_map: np.ndarray) -> np.ndarray:
    out = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label, rgb in LABEL_COLORS_RGB.items():
        out[label_map == label] = np.asarray(rgb, dtype=np.uint8)
    return out


def process_video(
    predictor,
    episode_id: int,
    view_index: int,
    video_path: Path,
    output_video_path: Path,
    output_mask_path: Path,
    debug_dir: Path,
    prompts: Iterable[Dict[str, str | int]],
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    calib: Calibration | None,
    camera_name: str | None,
    quaternion_order: str,
    debug_projection_images: int,
    axis_length_m: float,
    roi_neg_primary_m: float,
    roi_pos_primary_m: float,
    roi_neg_secondary_m: float,
    roi_pos_secondary_m: float,
    box_soft_margin_px: int,
    hand_label_id: int,
    output_fps: float,
    max_frames: int | None,
    wrist_priming_video_path: Path,
    downscaled_long_side_px: int | None,
    chunk_max_frames: int,
) -> None:
    info = _open_video_info(video_path)
    # Always track the full length of each video for both views.
    # (Keep `max_frames` in signature for compatibility with existing call sites.)
    max_frame_num_to_track = info.frame_count
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

    target_frame_count = len(original_frames)
    n_state = min(len(ee_pos_rows), len(ee_quat_rows))
    target_frame_count = min(target_frame_count, n_state)
    if target_frame_count <= 0:
        raise RuntimeError(
            f"No overlapping frames between video and pose arrays for video: {video_path}"
        )
    original_frames = original_frames[:target_frame_count]
    is_wrist_view = view_index == WRIST_VIEW_INDEX
    preprocessed_target_frames: List[np.ndarray]
    priming_frames: List[np.ndarray] = []
    input_h, input_w = original_frames[0].shape[:2]
    target_h, target_w = _compute_scaled_shape(
        height=input_h,
        width=input_w,
        downscaled_long_side_px=downscaled_long_side_px,
    )
    if is_wrist_view:
        if not wrist_priming_video_path.exists():
            raise FileNotFoundError(
                f"Wrist priming video not found: {wrist_priming_video_path}"
            )
        target_rotated_frames = [np.rot90(frame, k=-1) for frame in original_frames]
        rotated_h, rotated_w = target_rotated_frames[0].shape[:2]
        target_h, target_w = _compute_scaled_shape(
            height=rotated_h,
            width=rotated_w,
            downscaled_long_side_px=downscaled_long_side_px,
        )
        target_rotated_frames = _resize_frames_to_shape(target_rotated_frames, target_h, target_w)
        priming_cap = cv2.VideoCapture(str(wrist_priming_video_path))
        if not priming_cap.isOpened():
            raise RuntimeError(f"Could not open wrist priming video: {wrist_priming_video_path}")
        priming_frames: List[np.ndarray] = []
        try:
            while True:
                ok, frame_bgr = priming_cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                priming_frames.append(np.rot90(_ensure_rgb_uint8(frame_rgb), k=1))
        finally:
            priming_cap.release()
        if not priming_frames:
            raise RuntimeError(f"No frames found in wrist priming video: {wrist_priming_video_path}")
        # Keep wrist flow behavior consistent with wrist_view_segmentation.py.
        priming_frames = list(reversed(priming_frames))
        priming_frames = _resize_frames_to_shape(priming_frames, target_h, target_w)
        preprocessed_target_frames = target_rotated_frames
    else:
        preprocessed_target_frames = _resize_frames_to_shape(original_frames, target_h, target_w)
    if downscaled_long_side_px is not None:
        pre_h, pre_w = preprocessed_target_frames[0].shape[:2]
        print(
            f"[episode {episode_id} view={view_index}] preprocessing frame size: "
            f"{input_w}x{input_h} -> {pre_w}x{pre_h}"
        )

    frame_index_offset = len(priming_frames) if is_wrist_view else 0
    pre_h, pre_w = preprocessed_target_frames[0].shape[:2]

    camera_mode = None
    direction_mode = ""
    T_ee_cam = None
    T_cam_base_static = None
    ee_target_pix_by_frame: Dict[int, np.ndarray] | None = None
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
        print(
            f"[episode {episode_id} cam={camera_name}] extrinsic direction: {direction_mode}"
        )

        # SAM3 runs on preprocessed frames.
        scale_x = float(pre_w) / float(info.width)
        scale_y = float(pre_h) / float(info.height)
        ee_target_pix_by_frame = {}
        for frame_idx in range(target_frame_count):
            ee_pix, _axis_pix, vis = project_ee_and_axes(
                ee_pos=ee_pos_rows[frame_idx],
                ee_quat=ee_quat_rows[frame_idx],
                calib=calib,
                camera_mode=camera_mode,
                quaternion_order=quaternion_order,
                axis_length_m=axis_length_m,
                T_ee_cam=T_ee_cam,
                T_cam_base_static=T_cam_base_static,
            )
            if bool(vis[0]):
                ee_target_pix_by_frame[frame_idx] = np.asarray(
                    [float(ee_pix[0]) * scale_x, float(ee_pix[1]) * scale_y], dtype=np.float64
                )

    label_maps = np.zeros(
        (target_frame_count, info.height, info.width),
        dtype=np.uint8,
    )
    chunk_limit = int(chunk_max_frames)
    if chunk_limit <= 0:
        chunk_limit = target_frame_count
    if is_wrist_view:
        chunk_bounds = _iter_chunk_bounds(target_frame_count, chunk_limit)
    else:
        chunk_bounds = [(0, target_frame_count)]
    required_point_xy_wrist: tuple[int, int] | None = None
    if is_wrist_view:
        rotated_h = int(info.width)
        rotated_w = int(info.height)
        click_scale_x = float(pre_w) / float(rotated_w)
        click_scale_y = float(pre_h) / float(rotated_h)
        click_x = int(round(float(WRIST_CLICK_POINT_XY[0]) * click_scale_x))
        click_y = int(round(float(WRIST_CLICK_POINT_XY[1]) * click_scale_y))
        click_x = max(0, min(click_x, pre_w - 1))
        click_y = max(0, min(click_y, pre_h - 1))
        required_point_xy_wrist = (click_x, click_y)

    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        chunk_target_frames = preprocessed_target_frames[chunk_start:chunk_end]
        if is_wrist_view:
            chunk_frames = [*priming_frames, *chunk_target_frames]
        else:
            chunk_frames = chunk_target_frames
        sam3_frame_num_to_track = len(chunk_frames)

        preprocessed_dir = Path(tempfile.mkdtemp(prefix="sam3_preprocessed_frames_"))
        for idx, frame_rgb in enumerate(chunk_frames):
            Image.fromarray(frame_rgb).save(preprocessed_dir / f"{idx:06d}.jpg", quality=95)

        start_response = predictor.handle_request(
            request=dict(type="start_session", resource_path=str(preprocessed_dir))
        )
        session_id = start_response["session_id"]

        merged_outputs: Dict[int, Dict[int, np.ndarray]] = {}
        try:
            for prompt_spec in prompts:
                prompt_obj_id = int(prompt_spec["obj_id"])
                mask_label_id = int(prompt_spec.get("label_id", prompt_obj_id))
                text = str(prompt_spec["text"])
                if mask_label_id == 0:
                    raise ValueError(
                        "label_id=0 is reserved for background in this script. "
                        "Use non-zero label_id (for example 1 for hand, 2 for cube) "
                        "while keeping prompt obj_id as 0/1 if desired."
                    )

                _ = predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id)
                )
                _ = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        obj_id=prompt_obj_id,
                        text=text,
                    )
                )

                chunk_ee_target_pix_by_frame: Dict[int, np.ndarray] | None = None
                if (
                    ee_target_pix_by_frame is not None
                    and view_index == 0
                    and prompt_obj_id == 0
                ):
                    local_targets: Dict[int, np.ndarray] = {}
                    for local_idx, global_idx in enumerate(range(chunk_start, chunk_end)):
                        target = ee_target_pix_by_frame.get(global_idx)
                        if target is not None:
                            local_targets[local_idx + frame_index_offset] = target
                    chunk_ee_target_pix_by_frame = local_targets

                propagation_direction = "forward" if is_wrist_view else "both"
                prompt_outputs = _run_propagation_for_prompt(
                    predictor=predictor,
                    session_id=session_id,
                    mask_label_id=mask_label_id,
                    max_frame_num_to_track=sam3_frame_num_to_track,
                    start_frame_index=0,
                    propagation_direction=propagation_direction,
                    ee_target_pix_by_frame=chunk_ee_target_pix_by_frame,
                    required_point_xy=(
                        required_point_xy_wrist
                        if (is_wrist_view and mask_label_id == hand_label_id)
                        else None
                    ),
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

        for local_idx in range(chunk_end - chunk_start):
            sam3_frame_idx = local_idx + frame_index_offset
            frame_obj_masks = merged_outputs.get(sam3_frame_idx)
            frame_label_map = _build_label_map(
                frame_obj_masks=frame_obj_masks,
                width=pre_w,
                height=pre_h,
            )
            if is_wrist_view:
                frame_label_map = np.rot90(frame_label_map, k=1)
            if frame_label_map.shape != (info.height, info.width):
                frame_label_map = np.asarray(
                    Image.fromarray(frame_label_map).resize(
                        (info.width, info.height), resample=Image.NEAREST
                    ),
                    dtype=np.uint8,
                )
            label_maps[chunk_start + local_idx] = frame_label_map

        if len(chunk_bounds) > 1:
            print(
                f"[episode {episode_id} view={view_index}] "
                f"chunk {chunk_idx + 1}/{len(chunk_bounds)} done "
                f"frames {chunk_start}-{chunk_end - 1}"
            )

    debug_indices: List[int] = []
    if debug_projection_images > 0 and target_frame_count > 0:
        n_debug = min(debug_projection_images, target_frame_count)
        debug_indices = np.linspace(0, target_frame_count - 1, n_debug, dtype=int).tolist()
        debug_dir.mkdir(parents=True, exist_ok=True)

    writer = _create_video_writer(
        path=output_video_path,
        width=info.width,
        height=info.height,
        fps=output_fps if output_fps > 0 else info.fps,
    )
    filtered_label_maps = np.zeros_like(label_maps)
    try:
        for frame_idx in range(target_frame_count):
            frame_label_map = label_maps[frame_idx].copy()

            if calib is not None and camera_mode is not None:
                ee_pix, axis_pix, vis = project_ee_and_axes(
                    ee_pos=ee_pos_rows[frame_idx],
                    ee_quat=ee_quat_rows[frame_idx],
                    calib=calib,
                    camera_mode=camera_mode,
                    quaternion_order=quaternion_order,
                    axis_length_m=axis_length_m,
                    T_ee_cam=T_ee_cam,
                    T_cam_base_static=T_cam_base_static,
                )
                if frame_idx in debug_indices:
                    orig_overlay = draw_ee_overlay(
                        frame_rgb=original_frames[frame_idx],
                        ee_pix=ee_pix,
                        axis_pix=axis_pix,
                        vis=vis,
                    )
                    seg_overlay = draw_ee_overlay(
                        frame_rgb=_render_segmentation_frame(frame_label_map),
                        ee_pix=ee_pix,
                        axis_pix=axis_pix,
                        vis=vis,
                    )
                    side_by_side = np.concatenate([orig_overlay, seg_overlay], axis=1)
                    debug_path = debug_dir / f"frame_{frame_idx:06d}.png"
                    Image.fromarray(side_by_side).save(debug_path)
            filtered_label_maps[frame_idx] = frame_label_map
            frame_rgb = _render_segmentation_frame(frame_label_map)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()

    np.save(output_mask_path, filtered_label_maps)

    print(f"[done] video={video_path}")
    print(f"[done] segmentation video: {output_video_path}")
    print(f"[done] segmentation mask: {output_mask_path}")
    if debug_indices:
        print(f"[done] projection debug images: {debug_dir}")
    print(
        f"[done] frames={target_frame_count} "
        f"labels={np.unique(filtered_label_maps).tolist()}"
    )


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
    ee_quat_rows = np.asarray(
        data["observation.state.cartesian_orientation_quat"], dtype=np.float64
    )
    return episode_id, ee_pos_rows, ee_quat_rows


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
    rotate_view1_episode_ids = _parse_episode_id_tokens(args.rotate_view1_episode_ids)
    recompute_wrist_episode_ids = _parse_episode_id_tokens(args.recompute_wrist_episode_ids)
    recompute_third_episode_ids = _parse_episode_id_tokens(args.recompute_third_episode_ids)
    recompute_both_episode_ids = _parse_episode_id_tokens(args.recompute_both_episode_ids)

    calibration_map = load_calibration_map(args.calibration_dir)
    predictor = build_sam3_video_predictor()
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
            episode_id = requested_episode_id
            if ann_episode_id != requested_episode_id:
                print(
                    f"[warning] Episode mismatch: id list={requested_episode_id} "
                    f"annotation={ann_episode_id}; using annotation id."
                )
                episode_id = ann_episode_id

            selected_views, recompute_mode = _views_to_recompute_for_episode(
                requested_episode_id=requested_episode_id,
                annotation_episode_id=episode_id,
                recompute_wrist_episode_ids=recompute_wrist_episode_ids,
                recompute_third_episode_ids=recompute_third_episode_ids,
                recompute_both_episode_ids=recompute_both_episode_ids,
            )
            if recompute_mode and not selected_views:
                continue

            for view_index in sorted(selected_views):
                input_video_path = episode_dir / f"{view_index}_rgb.mp4"
                if not input_video_path.exists():
                    print(f"[warning] Missing input video: {input_video_path}")
                    continue
                if view_index == 1 and (
                    requested_episode_id in rotate_view1_episode_ids
                    or episode_id in rotate_view1_episode_ids
                ):
                    print(
                        f"[episode {episode_id}] rotating {input_video_path.name} by 180 degrees"
                    )
                    rotate_video_180_inplace(input_video_path, target_fps=args.output_fps)

                output_video_path = episode_dir / f"{view_index}_segmentation.mp4"
                output_mask_path = episode_dir / f"{view_index}_segmentation_mask.npy"
                output_debug_dir = episode_dir / f"{view_index}_projection_debug"
                prompts = VIEW_PROMPTS[view_index]
                camera_name = VIDEO_INDEX_TO_CAMERA.get(view_index)
                calib = calibration_map.get(camera_name) if camera_name is not None else None
                if calib is None:
                    print(
                        f"[warning] No calibration for view={view_index} camera={camera_name}; "
                        "projection overlay disabled."
                    )

                print(
                    f"[episode {episode_id}] processing view={view_index} "
                    f"video={input_video_path.name}"
                )
                roi_neg_primary_m = args.roi_neg_primary_m
                roi_pos_primary_m = args.roi_pos_primary_m
                roi_neg_secondary_m = args.roi_neg_secondary_m
                roi_pos_secondary_m = args.roi_pos_secondary_m
                if view_index == 1:
                    if args.wrist_roi_neg_primary_m is not None:
                        roi_neg_primary_m = args.wrist_roi_neg_primary_m
                    if args.wrist_roi_pos_primary_m is not None:
                        roi_pos_primary_m = args.wrist_roi_pos_primary_m
                    if args.wrist_roi_neg_secondary_m is not None:
                        roi_neg_secondary_m = args.wrist_roi_neg_secondary_m
                    if args.wrist_roi_pos_secondary_m is not None:
                        roi_pos_secondary_m = args.wrist_roi_pos_secondary_m

                process_video(
                    predictor=predictor,
                    episode_id=episode_id,
                    view_index=view_index,
                    video_path=input_video_path,
                    output_video_path=output_video_path,
                    output_mask_path=output_mask_path,
                    debug_dir=output_debug_dir,
                    prompts=prompts,
                    ee_pos_rows=ee_pos_rows,
                    ee_quat_rows=ee_quat_rows,
                    calib=calib,
                    camera_name=camera_name,
                    quaternion_order=args.quaternion_order,
                    debug_projection_images=args.debug_projection_images,
                    axis_length_m=args.axis_length_m,
                    roi_neg_primary_m=roi_neg_primary_m,
                    roi_pos_primary_m=roi_pos_primary_m,
                    roi_neg_secondary_m=roi_neg_secondary_m,
                    roi_pos_secondary_m=roi_pos_secondary_m,
                    box_soft_margin_px=args.box_soft_margin_px,
                    hand_label_id=args.hand_label_id,
                    output_fps=args.output_fps,
                    max_frames=args.max_frames,
                    wrist_priming_video_path=args.wrist_priming_video_path,
                    downscaled_long_side_px=args.downscaled_long_side_px,
                    chunk_max_frames=args.chunk_max_frames,
                )
    finally:
        predictor.shutdown()


if __name__ == "__main__":
    main()
