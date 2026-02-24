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
from typing import List, Sequence, Tuple

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


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    T_extrinsic: np.ndarray


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
    parser.add_argument("--camera-name", type=str, default="oakd_side_view")
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


def load_calibration(calibration_dir: Path, camera_name: str) -> Calibration:
    path_intrinsics = calibration_dir / "camera_intrinsics.pkl"
    path_extrinsics = calibration_dir / "transformations.pkl"
    if not path_intrinsics.exists() or not path_extrinsics.exists():
        raise FileNotFoundError(
            f"Missing calibration files under {calibration_dir}: "
            f"{path_intrinsics.name}, {path_extrinsics.name}"
        )

    with path_intrinsics.open("rb") as f:
        intr_data = pkl.load(f)
    if camera_name not in intr_data:
        raise KeyError(f"Camera {camera_name!r} not found in {path_intrinsics}")
    K, _dist_from_file = intr_data[camera_name]
    K = np.asarray(K, dtype=np.float64)
    # Use pure pinhole projection (no distortion adaptation/correction).
    dist = np.zeros((5,), dtype=np.float64)

    with path_extrinsics.open("rb") as f:
        extr_data = pkl.load(f)

    T = None
    if isinstance(extr_data, list):
        for item in extr_data:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == camera_name:
                T = np.asarray(item[1], dtype=np.float64)
                break
    if T is None:
        raise KeyError(f"Camera {camera_name!r} not found in {path_extrinsics}")
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic matrix, got {T.shape}")
    return Calibration(K=K, dist=dist, T_extrinsic=T)


def resolve_annotation_files(args: argparse.Namespace) -> List[Path]:
    if args.annotation_json is not None:
        return [args.annotation_json]
    files = sorted(args.annotation_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation json files in: {args.annotation_dir}")
    return files


def resolve_video_path(episode_id: int, args: argparse.Namespace) -> Path:
    if args.video_path is not None:
        return args.video_path

    episode_dir = args.video_dir / str(episode_id)
    candidates = [
        episode_dir / f"{episode_id}_rgb.mp4",
        episode_dir / "0_rgb.mp4",
        episode_dir / "1_rgb.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    for pattern in ("*_rgb.mp4", "*.mp4"):
        files = sorted(episode_dir.glob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No video found for episode {episode_id} in {episode_dir}")


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


def make_transform(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def as_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    return np.hstack([points_xyz, ones])


def score_projection(
    points_world: np.ndarray,
    T_c_w: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int, int]:
    img_pts, visible, _ = project_points(points_world, K, dist, T_c_w)
    u = img_pts[:, 0]
    v = img_pts[:, 1]
    in_frame = visible & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return int(np.sum(in_frame)), int(np.sum(visible)), len(points_world)


def choose_extrinsic_direction(
    T_raw: np.ndarray,
    sample_world_points: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, str]:
    T_inv = np.linalg.inv(T_raw)
    raw_in, raw_vis, n = score_projection(sample_world_points, T_raw, K, dist, width, height)
    inv_in, inv_vis, _ = score_projection(sample_world_points, T_inv, K, dist, width, height)
    if raw_in >= inv_in:
        return (
            T_raw,
            f"raw(world_to_camera assumed) in_frame={raw_in}/{n} visible={raw_vis}/{n}",
        )
    return (
        T_inv,
        f"inverse(camera_to_world provided) in_frame={inv_in}/{n} visible={inv_vis}/{n}",
    )


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_c_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    pts_h = as_homogeneous(points_world)
    pts_cam = (T_c_w @ pts_h.T).T[:, :3]
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


def process_episode(
    annotation_path: Path,
    video_path: Path,
    output_dir: Path,
    calib: Calibration,
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

    n_diag = min(max(1, args.diagnostic_first_n), n) if args.diagnostic_first_n > 0 else min(10, n)
    sample_points = ee_pos_rows[:n_diag]
    T_c_w, extr_mode = choose_extrinsic_direction(
        calib.T_extrinsic, sample_points, calib.K, calib.dist, width, height
    )

    # Optional XY-swap diagnostic: some logs/tools accidentally interpret pixel order as (v, u).
    # We evaluate both interpretations and keep swapped only if explicitly enabled and better.
    sample_img, sample_vis, _ = project_points(sample_points, calib.K, calib.dist, T_c_w)
    normal_in = int(
        np.sum(
            sample_vis
            & (sample_img[:, 0] >= 0)
            & (sample_img[:, 0] < width)
            & (sample_img[:, 1] >= 0)
            & (sample_img[:, 1] < height)
        )
    )
    swapped_in = int(
        np.sum(
            sample_vis
            & (sample_img[:, 1] >= 0)
            & (sample_img[:, 1] < width)
            & (sample_img[:, 0] >= 0)
            & (sample_img[:, 0] < height)
        )
    )
    use_swapped_xy = bool(args.auto_swap_xy and swapped_in > normal_in)

    if args.diagnostic_first_n > 0:
        print(f"[diagnostic] extrinsic_choice: {extr_mode}")
        print(
            f"[diagnostic] pixel_order normal_in_frame={normal_in}/{n_diag} "
            f"swapped_in_frame={swapped_in}/{n_diag} use_swapped_xy={use_swapped_xy}"
        )
        for i in range(n_diag):
            u = float(sample_img[i, 0])
            v = float(sample_img[i, 1])
            z = float((T_c_w @ np.r_[sample_points[i], 1.0])[:3][2])
            normal_ok = bool(sample_vis[i] and (0 <= u < width) and (0 <= v < height))
            swapped_ok = bool(sample_vis[i] and (0 <= v < width) and (0 <= u < height))
            print(
                f"[diagnostic frame {i}] uv=({u:.1f},{v:.1f}) z={z:.4f} "
                f"normal_in={normal_ok} swapped_in={swapped_ok}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_pref = output_dir / f"episode_{episode_id}_overlay.mp4"
    out_meta_path = output_dir / f"episode_{episode_id}_meta.json"
    out_pixels_path = output_dir / f"episode_{episode_id}_ee_pixels.npy"

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

        # We keep orientation parsing for future axes/pose overlays, even though this
        # EE-only mode currently visualizes just the projected EE position.
        _ = quat_to_rotmat(ee_quat_rows[i], order=args.quaternion_order)
        img_pts, vis, pts_cam = project_points(
            np.asarray([ee_pos_rows[i]], dtype=np.float64), calib.K, calib.dist, T_c_w
        )
        ee_px = img_pts[0]
        if use_swapped_xy:
            ee_px = np.array([ee_px[1], ee_px[0]], dtype=np.float64)
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
                f"[episode {episode_id} frame {i}] "
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
        "camera_name": args.camera_name,
        "extrinsic_mode": extr_mode,
        "use_swapped_xy": use_swapped_xy,
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

    print(f"[episode {episode_id}] processed_frames={n}")
    print(f"[episode {episode_id}] in_frame_visible_frames={in_frame_count}")
    print(f"[episode {episode_id}] saved overlay: {out_overlay_path}")
    print(f"[episode {episode_id}] saved ee pixels: {out_pixels_path}")


def main() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required for projection/video rendering. "
            "Install it with: pip install opencv-python"
        )
    args = parse_args()
    annotation_files = resolve_annotation_files(args)
    calib = load_calibration(args.calibration_dir, args.camera_name)

    for annotation_path in annotation_files:
        episode_id = int(read_annotation(annotation_path)["episode_id"])
        video_path = resolve_video_path(episode_id, args)
        process_episode(
            annotation_path=annotation_path,
            video_path=video_path,
            output_dir=args.output_dir,
            calib=calib,
            args=args,
        )


if __name__ == "__main__":
    main()
