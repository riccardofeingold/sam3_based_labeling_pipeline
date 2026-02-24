#!/usr/bin/env python3
"""
Project only the end-effector (palm) pose onto video frames.

This script intentionally does NOT compute fingertip pixel positions or fingertip
bounding boxes. It is focused on visualizing EE projection for debugging.
"""

from __future__ import annotations

import argparse
import json
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - dependency guard
    cv2 = None


DEFAULT_CALIBRATION_DIR = Path(
    "/data/sam3_based_labeling_pipeline/assets/calibration_params"
)
DEFAULT_ANNOTATION_DIR = Path(
    "/data/sam3_based_labeling_pipeline/assets/test_le_robot_dataset/annotation"
)
DEFAULT_VIDEO_DIR = Path(
    "/data/sam3_based_labeling_pipeline/assets/test_le_robot_dataset/videos"
)
DEFAULT_OUTPUT_DIR = Path("/data/sam3_based_labeling_pipeline/assets/videos/tests")

# Camera stream index inside episode video folder -> calibration camera name.
VIDEO_INDEX_TO_CAMERA = {
    0: "oakd_side_view",
    1: "oakd_wrist_view",
}

# How to interpret extrinsics for each camera stream.
# - base_camera: static camera in base/world frame (side view)
# - ee_camera: camera pose relative to end-effector (wrist view)
CAMERA_EXTRINSIC_MODE = {
    "oakd_side_view": "base_camera",
    "oakd_wrist_view": "ee_camera",
}

# EE position correction applied before projection/chaining.
EE_TRANSLATION_OFFSET = np.array([0.13, 0.0, 0.07], dtype=np.float64)


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    T_ee_camera: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project end-effector pose and render overlay video."
    )
    group_ann = parser.add_mutually_exclusive_group(required=False)
    group_ann.add_argument("--annotation-json", type=Path, default=None)
    group_ann.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)

    group_vid = parser.add_mutually_exclusive_group(required=False)
    group_vid.add_argument("--video-path", type=Path, default=None)
    group_vid.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument(
        "--camera-indices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Video stream indices to process (mapped via VIDEO_INDEX_TO_CAMERA).",
    )
    parser.add_argument(
        "--quaternion-order",
        type=str,
        choices=["xyzw", "wxyz"],
        default="xyzw",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--debug-print-pixels",
        action="store_true",
        help="Print per-frame projected EE pixel and camera depth.",
    )
    parser.add_argument(
        "--diagnostic-first-n",
        type=int,
        default=0,
        help="Number of first frames used for transform/swap diagnostics.",
    )
    parser.add_argument(
        "--auto-swap-xy",
        action="store_true",
        help="Choose swapped pixel interpretation if it yields better in-frame score.",
    )
    return parser.parse_args()


def load_calibration_map(calibration_dir: Path) -> Dict[str, Calibration]:
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
        # Use pure pinhole projection (no distortion adaptation/correction).
        dist = np.zeros((5,), dtype=np.float64)
        calibs[cam_name] = Calibration(K=K, dist=dist, T_ee_camera=extr_map[cam_name])

    if not calibs:
        raise RuntimeError(f"No valid camera calibrations loaded from {calibration_dir}")
    return calibs


def resolve_annotation_files(args: argparse.Namespace) -> List[Path]:
    if args.annotation_json is not None:
        return [args.annotation_json]
    files = sorted(args.annotation_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation json files in: {args.annotation_dir}")
    return files


def resolve_video_path(episode_id: int, args: argparse.Namespace, video_index: int) -> Path:
    if args.video_path is not None:
        return args.video_path

    episode_dir = args.video_dir / str(episode_id)
    candidates = [
        episode_dir / f"{video_index}_rgb.mp4",
        episode_dir / f"{episode_id}_rgb.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    for pattern in ("*_rgb.mp4", "*.mp4"):
        files = sorted(episode_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No video found for episode {episode_id} (index {video_index}) in {episode_dir}"
    )


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
    """
    Compute previous-joint pose from EE pose using:
      T_base_prev = T_base_ee @ inv(T_prev_to_ee)
    where R_prev_to_ee = I and t_prev_to_ee = EE_TRANSLATION_OFFSET.
    """
    R_base_ee = quat_to_rotmat(ee_quat, order=quaternion_order)
    p_base_ee = np.asarray(ee_pos, dtype=np.float64)
    p_base_prev = p_base_ee - (R_base_ee @ EE_TRANSLATION_OFFSET)
    return R_base_ee, p_base_prev


def make_transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def as_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    return np.hstack([points_xyz, ones])


def project_points_camera_frame(
    points_camera: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_cam = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    visible = pts_cam[:, 2] > 1e-6
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img_pts, _ = cv2.projectPoints(pts_cam, rvec, tvec, K, dist)
    return img_pts.reshape(-1, 2), visible, pts_cam


def create_video_writer(
    preferred_path: Path, width: int, height: int, fps: float
) -> Tuple[cv2.VideoWriter, Path, str]:
    candidates = [
        (preferred_path, "mp4v"),
        (preferred_path, "avc1"),
        (preferred_path.with_suffix(".avi"), "MJPG"),
    ]
    for out_path, codec in candidates:
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*codec),
            fps if fps > 0 else 30.0,
            (width, height),
        )
        if writer.isOpened():
            print(f"[writer] codec={codec} path={out_path}")
            return writer, out_path, codec
    raise RuntimeError(f"Could not open video writer for output {preferred_path}")


def score_ee_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_ee_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int,
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
        # Project the EE point itself after computing camera pose via previous joint.
        p_ee_base = np.asarray(ee_pos_rows[i], dtype=np.float64)
        p_prev_cam = (T_cam_base @ np.r_[p_ee_base, 1.0])[:3]
        img_pts, vis, _ = project_points_camera_frame(
            np.asarray([p_prev_cam], dtype=np.float64), K, dist
        )
        if bool(vis[0]):
            visible += 1
            u, v = float(img_pts[0, 0]), float(img_pts[0, 1])
            if 0 <= u < width and 0 <= v < height:
                in_frame += 1
    return in_frame, visible, n


def score_base_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_base_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int,
) -> Tuple[int, int, int]:
    n = min(len(ee_pos_rows), sample_n)
    T_cam_base = np.linalg.inv(T_base_cam)
    points_base = np.asarray(ee_pos_rows[:n], dtype=np.float64)
    points_cam = (T_cam_base @ as_homogeneous(points_base).T).T[:, :3]
    img_pts, vis, _ = project_points_camera_frame(points_cam, K, dist)
    in_frame = int(
        np.sum(
            vis
            & (img_pts[:, 0] >= 0)
            & (img_pts[:, 0] < width)
            & (img_pts[:, 1] >= 0)
            & (img_pts[:, 1] < height)
        )
    )
    visible = int(np.sum(vis))
    return in_frame, visible, n


def choose_base_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_base_cam_raw: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int,
) -> Tuple[np.ndarray, str]:
    raw_in, raw_vis, n = score_base_cam_direction(
        ee_pos_rows,
        ee_quat_rows,
        T_base_cam_raw,
        K,
        dist,
        width,
        height,
        quaternion_order,
        sample_n,
    )
    inv_T = np.linalg.inv(T_base_cam_raw)
    inv_in, inv_vis, _ = score_base_cam_direction(
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
        return (
            T_base_cam_raw,
            f"raw(T_base_camera) in_frame={raw_in}/{n} visible={raw_vis}/{n}",
        )
    return (
        inv_T,
        f"inverse(T_base_camera) in_frame={inv_in}/{n} visible={inv_vis}/{n}",
    )


def choose_ee_cam_direction(
    ee_pos_rows: np.ndarray,
    ee_quat_rows: np.ndarray,
    T_ee_cam_raw: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    quaternion_order: str,
    sample_n: int,
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
        return (
            T_ee_cam_raw,
            f"raw(T_ee_camera) in_frame={raw_in}/{n} visible={raw_vis}/{n}",
        )
    return (
        inv_T,
        f"inverse(T_ee_camera) in_frame={inv_in}/{n} visible={inv_vis}/{n}",
    )


def draw_overlay(frame: np.ndarray, ee_pixel: np.ndarray | None) -> np.ndarray:
    out = frame.copy()
    if ee_pixel is None:
        return out
    u, v = int(round(float(ee_pixel[0]))), int(round(float(ee_pixel[1])))
    cv2.circle(out, (u, v), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.drawMarker(
        out,
        (u, v),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "end_effector",
        (u + 6, v + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def read_annotation(annotation_path: Path) -> dict:
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
    return data


def process_episode_camera(
    annotation_path: Path,
    video_path: Path,
    output_dir: Path,
    calib: Calibration,
    camera_name: str,
    video_index: int,
    args: argparse.Namespace,
) -> None:
    data = read_annotation(annotation_path)
    episode_id = int(data["episode_id"])
    ee_pos_rows = np.asarray(data["observation.state.cartesian_position"], dtype=np.float64)
    ee_quat_rows = np.asarray(
        data["observation.state.cartesian_orientation_quat"], dtype=np.float64
    )
    n_state = min(len(ee_pos_rows), len(ee_quat_rows))
    if args.max_frames is not None:
        n_state = min(n_state, args.max_frames)
    if n_state <= 0:
        raise ValueError(f"No usable states in {annotation_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if args.fps is not None and args.fps > 0:
        fps = args.fps

    n = min(n_state, video_frames) if video_frames > 0 else n_state
    if n <= 0:
        cap.release()
        raise ValueError(f"No overlapping frames between states and video for episode {episode_id}")

    camera_mode = CAMERA_EXTRINSIC_MODE.get(camera_name, "ee_camera")
    sample_n = args.diagnostic_first_n if args.diagnostic_first_n > 0 else 20
    direction_mode = ""
    T_ee_cam = None
    T_cam_base_static = None
    if camera_mode == "base_camera":
        T_base_cam, direction_mode = choose_base_cam_direction(
            ee_pos_rows=ee_pos_rows,
            ee_quat_rows=ee_quat_rows,
            T_base_cam_raw=calib.T_ee_camera,
            K=calib.K,
            dist=calib.dist,
            width=width,
            height=height,
            quaternion_order=args.quaternion_order,
            sample_n=sample_n,
        )
        T_cam_base_static = np.linalg.inv(T_base_cam)
        print(f"[episode {episode_id} cam={camera_name}] base_cam_direction: {direction_mode}")
    else:
        # Extrinsics are relative to EE pose. Chain per frame: T_base_cam = T_base_ee @ T_ee_cam.
        T_ee_cam, direction_mode = choose_ee_cam_direction(
            ee_pos_rows=ee_pos_rows,
            ee_quat_rows=ee_quat_rows,
            T_ee_cam_raw=calib.T_ee_camera,
            K=calib.K,
            dist=calib.dist,
            width=width,
            height=height,
            quaternion_order=args.quaternion_order,
            sample_n=sample_n,
        )
        print(f"[episode {episode_id} cam={camera_name}] ee_cam_direction: {direction_mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_pref = output_dir / f"episode_{episode_id}_{camera_name}_overlay.mp4"
    out_meta_path = output_dir / f"episode_{episode_id}_{camera_name}_meta.json"
    out_pixels_path = output_dir / f"episode_{episode_id}_{camera_name}_ee_pixels.npy"

    writer, out_overlay_path, writer_codec = create_video_writer(
        preferred_path=out_overlay_pref, width=width, height=height, fps=fps
    )

    ee_pixels = np.full((n, 2), -1.0, dtype=np.float64)
    visible_count = 0
    in_frame_count = 0

    for i in range(n):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        R_base_prev, p_base_prev = compute_prev_joint_pose_from_ee(
            ee_pos_rows[i], ee_quat_rows[i], args.quaternion_order
        )
        if camera_mode == "base_camera":
            # Side camera: static transform between base and camera.
            p_ee_base = np.asarray(ee_pos_rows[i], dtype=np.float64)
            p_ee_cam = (T_cam_base_static @ np.r_[p_ee_base, 1.0])[:3]
        else:
            # Wrist camera: transform depends on current EE pose.
            T_base_prev = make_transform(R_base_prev, p_base_prev)
            T_base_cam = T_base_prev @ T_ee_cam
            T_cam_base = np.linalg.inv(T_base_cam)
            p_ee_base = np.asarray(ee_pos_rows[i], dtype=np.float64)
            p_ee_cam = (T_cam_base @ np.r_[p_ee_base, 1.0])[:3]
        img_pts, vis, pts_cam = project_points_camera_frame(
            np.asarray([p_ee_cam], dtype=np.float64), calib.K, calib.dist
        )
        ee_px = img_pts[0]
        is_visible = bool(vis[0])
        z = float(pts_cam[0, 2])
        is_in_frame = is_visible and (0 <= ee_px[0] < width and 0 <= ee_px[1] < height)
        if is_visible:
            visible_count += 1
        if is_in_frame:
            in_frame_count += 1
            ee_pixels[i] = ee_px

        if args.debug_print_pixels:
            print(
                f"[episode {episode_id} cam={camera_name} idx={video_index} frame {i}] "
                f"ee_px=({ee_px[0]:.1f},{ee_px[1]:.1f}) z={z:.4f} "
                f"visible={is_visible} in_frame={is_in_frame}"
            )

        overlay = draw_overlay(frame, ee_px if is_in_frame else None)
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        if overlay.shape[:2] != (height, width):
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
        overlay = np.ascontiguousarray(overlay, dtype=np.uint8)
        writer.write(overlay)

    cap.release()
    writer.release()

    np.save(out_pixels_path, ee_pixels)
    meta = {
        "episode_id": episode_id,
        "annotation_path": str(annotation_path),
        "video_path": str(video_path),
        "camera_name": camera_name,
        "video_index": video_index,
        "camera_extrinsic_mode": camera_mode,
        "transform_chain": (
            "T_base_cam static (base_camera mode)"
            if camera_mode == "base_camera"
            else "T_base_cam = T_base_prevJoint @ T_prevJoint_camera"
        ),
        "extrinsic_direction": direction_mode,
        "writer_codec": writer_codec,
        "overlay_path": str(out_overlay_path),
        "ee_pixels_path": str(out_pixels_path),
        "processed_frames": int(n),
        "visible_frames": int(visible_count),
        "in_frame_visible_frames": int(in_frame_count),
        "image_size": [width, height],
    }
    with out_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[episode {episode_id} cam={camera_name}] processed_frames={n}")
    print(f"[episode {episode_id} cam={camera_name}] in_frame_visible_frames={in_frame_count}")
    print(f"[episode {episode_id} cam={camera_name}] saved overlay: {out_overlay_path}")
    print(f"[episode {episode_id} cam={camera_name}] saved ee pixels: {out_pixels_path}")


def main() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for projection/video rendering. "
            "Install it with: pip install opencv-python"
        )
    args = parse_args()
    annotation_files = resolve_annotation_files(args)
    calibration_map = load_calibration_map(args.calibration_dir)

    for annotation_path in annotation_files:
        episode_id = int(read_annotation(annotation_path)["episode_id"])
        for video_index in args.camera_indices:
            if video_index not in VIDEO_INDEX_TO_CAMERA:
                print(f"[episode {episode_id}] skipping unknown video index {video_index}")
                continue
            camera_name = VIDEO_INDEX_TO_CAMERA[video_index]
            if camera_name not in calibration_map:
                print(f"[episode {episode_id}] calibration missing for camera {camera_name}")
                continue
            video_path = resolve_video_path(episode_id, args, video_index=video_index)
            process_episode_camera(
                annotation_path=annotation_path,
                video_path=video_path,
                output_dir=args.output_dir,
                calib=calibration_map[camera_name],
                camera_name=camera_name,
                video_index=video_index,
                args=args,
            )


if __name__ == "__main__":
    main()
