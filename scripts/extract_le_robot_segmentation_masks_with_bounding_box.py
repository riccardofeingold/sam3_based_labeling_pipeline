#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
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
    "/data/sam3_based_labeling_pipeline/assets/test_le_robot_dataset"
    # "/data/Ctrl-World/datasets/red_cube_not_on_red_ramp_real"
)
DEFAULT_CALIBRATION_DIR = Path(
    "/data/sam3_based_labeling_pipeline/assets/calibration_params"
)

# View index -> prompt list. obj_id maps directly to mask label id.
VIEW_PROMPTS = {
    0: [
        {"obj_id": 0, "text": "the hand"},
        {"obj_id": 1, "text": "red dice"},
    ],
    1: [
        {"obj_id": 0, "text": "the hand"},
        {"obj_id": 1, "text": "red dice"},
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
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum frames to track per video.",
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
        "--quaternion-order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion component ordering in annotation orientation arrays.",
    )
    parser.add_argument(
        "--debug-projection-images",
        type=int,
        default=6,
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
    ee_target_pix_by_frame: Dict[int, np.ndarray] | None = None,
) -> Dict[int, Dict[str, np.ndarray]]:
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

        target_pix = (
            ee_target_pix_by_frame.get(frame_idx)
            if ee_target_pix_by_frame is not None
            else None
        )
        if target_pix is not None:
            best_idx = min(
                range(len(masks)),
                key=lambda i: _min_sq_dist_mask_to_point(masks[i], target_pix),
            )
        else:
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
    episode_id: int,
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
    n_state = min(len(ee_pos_rows), len(ee_quat_rows))
    max_frame_num_to_track = min(max_frame_num_to_track, n_state)
    if max_frame_num_to_track <= 0:
        raise RuntimeError(
            f"No overlapping frames between video and pose arrays for video: {video_path}"
        )
    original_frames = original_frames[:max_frame_num_to_track]
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

    camera_mode = None
    direction_mode = ""
    T_ee_cam = None
    T_cam_base_static = None
    ee_target_pix_by_frame: Dict[int, np.ndarray] | None = None
    if calib is not None and camera_name is not None:
        camera_mode = CAMERA_EXTRINSIC_MODE.get(camera_name, "ee_camera")
        if camera_mode == "base_camera":
            T_base_cam, direction_mode = choose_base_cam_direction(
                ee_pos_rows=ee_pos_rows[:max_frame_num_to_track],
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
                ee_pos_rows=ee_pos_rows[:max_frame_num_to_track],
                ee_quat_rows=ee_quat_rows[:max_frame_num_to_track],
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

        # SAM3 runs on preprocessed frames. Scale projected EE pixels into that space.
        pre_h, pre_w = preprocessed_frames[0].shape[:2]
        sx = pre_w / float(info.width)
        sy = pre_h / float(info.height)
        ee_target_pix_by_frame = {}
        for frame_idx in range(max_frame_num_to_track):
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
                    [ee_pix[0] * sx, ee_pix[1] * sy], dtype=np.float64
                )

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
                ee_target_pix_by_frame=(
                    ee_target_pix_by_frame if obj_id == hand_label_id else None
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

    debug_indices: List[int] = []
    if debug_projection_images > 0 and max_frame_num_to_track > 0:
        n_debug = min(debug_projection_images, max_frame_num_to_track)
        debug_indices = np.linspace(0, max_frame_num_to_track - 1, n_debug, dtype=int).tolist()
        debug_dir.mkdir(parents=True, exist_ok=True)

    writer = _create_video_writer(
        path=output_video_path,
        width=info.width,
        height=info.height,
        fps=output_fps if output_fps > 0 else info.fps,
    )
    roi_neg_primary_m = max(0.0, float(roi_neg_primary_m))
    roi_pos_primary_m = max(0.0, float(roi_pos_primary_m))
    roi_neg_secondary_m = max(0.0, float(roi_neg_secondary_m))
    roi_pos_secondary_m = max(0.0, float(roi_pos_secondary_m))
    filtered_label_maps = np.zeros_like(label_maps)
    try:
        for frame_idx in range(max_frame_num_to_track):
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
                roi_poly_pix, roi_poly_vis, _chosen_axes = project_ee_axis_aligned_roi_polygon(
                    ee_pos=ee_pos_rows[frame_idx],
                    ee_quat=ee_quat_rows[frame_idx],
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
                if bool(np.all(roi_poly_vis)):
                    roi_mask = rasterize_hull_mask(
                        width=info.width,
                        height=info.height,
                        points_xy=roi_poly_pix,
                    )
                    if box_soft_margin_px > 0:
                        kernel_radius = int(box_soft_margin_px)
                        kernel_size = kernel_radius * 2 + 1
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                        )
                        roi_mask_uint8 = roi_mask.astype(np.uint8)
                        roi_mask = cv2.dilate(roi_mask_uint8, kernel, iterations=1) > 0
                    hand_mask = frame_label_map == np.uint8(hand_label_id)
                    frame_label_map[hand_mask & (~roi_mask)] = np.uint8(0)
                filtered_label_maps[frame_idx] = frame_label_map

                if frame_idx in debug_indices:
                    orig_overlay = draw_ee_overlay(
                        frame_rgb=original_frames[frame_idx],
                        ee_pix=ee_pix,
                        axis_pix=axis_pix,
                        vis=vis,
                        roi_polygon=roi_poly_pix,
                        roi_polygon_vis=roi_poly_vis,
                    )
                    seg_overlay = draw_ee_overlay(
                        frame_rgb=_render_segmentation_frame(frame_label_map),
                        ee_pix=ee_pix,
                        axis_pix=axis_pix,
                        vis=vis,
                        roi_polygon=roi_poly_pix,
                        roi_polygon_vis=roi_poly_vis,
                    )
                    side_by_side = np.concatenate([orig_overlay, seg_overlay], axis=1)
                    debug_path = debug_dir / f"frame_{frame_idx:06d}.png"
                    Image.fromarray(side_by_side).save(debug_path)
            else:
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
        f"[done] frames={max_frame_num_to_track} "
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

            for view_index in (0, 1):
                input_video_path = episode_dir / f"{view_index}_rgb.mp4"
                if not input_video_path.exists():
                    print(f"[warning] Missing input video: {input_video_path}")
                    continue

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
                )
    finally:
        predictor.shutdown()


if __name__ == "__main__":
    main()
