#!/usr/bin/env python3
"""
Render OrcaHand segmentation masks and fingertip end-effector positions via MuJoCo.

Outputs:
- dataset-mode RGB videos
- dataset-mode hand mask videos
- per-frame label maps
- fingertip positions JSONL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Prefer an onscreen backend when a display server is available.
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

try:
    import mujoco.viewer as mujoco_viewer
except Exception as exc:  # pragma: no cover
    mujoco_viewer = None
    MUJOCO_VIEWER_IMPORT_ERROR = exc
else:
    MUJOCO_VIEWER_IMPORT_ERROR = None

try:
    import mediapy as media
except Exception as exc:  # pragma: no cover
    media = None
    MEDIAPY_IMPORT_ERROR = exc
else:
    MEDIAPY_IMPORT_ERROR = None

try:
    from scipy.spatial.transform import Rotation as SciRotation
except Exception as exc:  # pragma: no cover
    SciRotation = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None


DEFAULT_MODEL_PATH = Path(
    # "/data/sam3_based_labeling_pipeline/assets/orcahand_v1b/orcahand.xml"
    "assets/orcahand_v1b/orcahand.xml"
)
DEFAULT_CALIBRATION_DIR = Path("assets/calibration_params")
DEFAULT_OUTPUT_DIR = Path("output_mujoco")
DEFAULT_DATASET_ROOT = Path(
    "assets/test_dataset"
)

VIDEO_INDEX_TO_CAMERA = {
    0: "oakd_side_view",
    1: "oakd_wrist_view",
}
CAMERA_TO_VIDEO_INDEX = {v: k for k, v in VIDEO_INDEX_TO_CAMERA.items()}

CAMERA_EXTRINSIC_MODE = {
    "oakd_side_view": "base_camera",
    "oakd_wrist_view": "ee_camera",
}
CALIB_CAMERA_TO_XML_CAMERA = {
    "oakd_side_view": "third_view_camera",
    "oakd_wrist_view": "wrist_view_camera",
}
CAMERA_REFERENCE_LINK = {
    "oakd_wrist_view": "hand_mount",
}
OPENCV_CAMERA_TO_MUJOCO_CAMERA_ROT = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
EE_LINK_OPTIMIZATION_CAMERA = "oakd_side_view"
HAND_MOUNT_OPTIMIZATION_CAMERA = "oakd_wrist_view"
HAND_MOUNT_BODY_NAME = "hand_mount"

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LABEL_ID = {name: idx + 1 for idx, name in enumerate(FINGER_ORDER)}

FALLBACK_TIP_SPECS = {
    "thumb": ("thumb_dp", np.array([0.0, 0.0, 0.03], dtype=np.float64)),
    "index": ("index_ip", np.array([0.0, 0.0, 0.04], dtype=np.float64)),
    "middle": ("middle_ip", np.array([0.0, 0.0, 0.04], dtype=np.float64)),
    "ring": ("ring_ip", np.array([0.0, 0.0, 0.04], dtype=np.float64)),
    "pinky": ("pinky_ip", np.array([0.0, 0.0, 0.04], dtype=np.float64)),
}
EXCLUDED_HAND_MASK_LINK_NAMES = {"tower"}

HAND_JOINT_ORDER = [
    "joint_palm",
    "joint_abd_thumb",
    "joint_iip_thumb",
    "joint_pip_thumb",
    "joint_dip_thumb",
    "joint_abd_index",
    "joint_pip_index",
    "joint_iip_index",
    "joint_abd_middle",
    "joint_pip_middle",
    "joint_iip_middle",
    "joint_abd_ring",
    "joint_pip_ring",
    "joint_iip_ring",
    "joint_abd_pinky",
    "joint_pip_pinky",
    "joint_iip_pinky",
]

@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    extrinsic: np.ndarray


@dataclass
class FingertipSpec:
    finger: str
    position_link: str
    local_offset_xyz: np.ndarray
    mask_link_candidates: List[str]


@dataclass
class CameraState:
    kind: str  # calibrated
    calibration_camera_name: Optional[str]
    T_world_camera: Optional[np.ndarray]
    K: Optional[np.ndarray]


@dataclass
class DatasetCameraTarget:
    camera_name: str
    calib: Calibration
    mode: str
    ee_offset: np.ndarray
    real_seg_video_path: Optional[Path]
    view_index: int
    output_video_width: int
    output_video_height: int
    episode_out: Path
    rgb_video_path: Path
    hand_mask_video_path: Path
    vis_seg_actions_video_path: Optional[Path]
    label_map_dir: Path
    per_finger_base_dir: Path
    lines_path: Path


@dataclass
class OptimizationEpisodeData:
    ep_id: int
    ann_path: Path
    joint_key: str
    joint_seq: np.ndarray
    ee_pose_seq: np.ndarray
    frame_count: int
    frame_indices: List[int]
    real_masks_by_camera: Dict[str, Dict[int, np.ndarray]]
    camera_targets: List[DatasetCameraTarget]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OrcaHand MuJoCo model + calibration to render segmentation masks and fingertip positions."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "LeRobot dataset root containing annotation/ or annotations/. "
            "If set, runs dataset batch rendering mode."
        ),
    )
    parser.add_argument(
        "--vis-seg-actions-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write simulated segmentation mask videos in dataset layout "
            "(default: <dataset_root>/vis_seg_actions)."
        ),
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Optional explicit annotation directory (overrides auto-detect from dataset-root).",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        nargs="+",
        default=None,
        help="Episode ids and ranges, e.g. 1 3 8-12. Use 'all' for all ids.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Optional frame cap per episode in dataset mode.",
    )
    parser.add_argument(
        "--hand-joint-source",
        type=str,
        choices=["action", "observation"],
        default="action",
        help="Use action.hand_joint_position or observation.state.hand_joint_position.",
    )
    parser.add_argument(
        "--palm-pose-key",
        type=str,
        default="observation.state.cartesian_position",
        help="Annotation key for absolute end-effector pose as [x,y,z,roll,pitch,yaw].",
    )
    parser.add_argument(
        "--camera-setup",
        type=str,
        choices=["calibrated"],
        default="calibrated",
        help="Use calibration-based camera.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="oakd_side_view",
        help="Camera name in calibration pickle (e.g. oakd_side_view, oakd_wrist_view).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Optional dataset camera index (0/1). Overrides --camera-name when set.",
    )
    parser.add_argument(
        "--camera-mode",
        type=str,
        choices=["auto", "base_camera", "ee_camera"],
        default="auto",
        help="How to interpret extrinsic transform from calibration.",
    )
    parser.add_argument(
        "--invert-extrinsic",
        action="store_true",
        help="Invert extrinsic matrix from calibration before use.",
    )
    parser.add_argument(
        "--ee-link",
        type=str,
        default="ee_link",
        help="Link used as EE frame when camera-mode is ee_camera.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument(
        "--no-hand-root-frame-overlay",
        action="store_true",
        help="Disable MuJoCo-rendered body coordinate frames in RGB outputs.",
    )
    parser.add_argument(
        "--hand-root-frame-axis-length",
        type=float,
        default=0.1,
        help="Axis length (meters) for MuJoCo-rendered body coordinate frames.",
    )
    parser.add_argument(
        "--save-per-finger-masks",
        action="store_true",
        help="In dataset mode, save per-finger binary mask PNG per frame.",
    )
    parser.add_argument(
        "--hand-mask-color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        default=(0, 255, 0),
        help=(
            "RGB color for the rendered hand mask videos. "
            "Example: --hand-mask-color 0 255 0"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=5.0,
        help="FPS used when writing dataset-mode rgb.mp4 and hand_mask.mp4.",
    )
    parser.add_argument(
        "--disable-auto-extrinsic-direction",
        action="store_true",
        default=True,
        help=(
            "Use raw extrinsic from calibration without heuristic inversion scoring. "
            "Default ON because the calibration stores mnt_T_cam (camera-in-mount frame), "
            "so the raw matrix is always the correct T_world_camera / T_ee_camera."
        ),
    )
    parser.add_argument(
        "--auto-extrinsic-direction",
        dest="disable_auto_extrinsic_direction",
        action="store_false",
        help="Re-enable heuristic raw/inverse extrinsic direction scoring (not recommended).",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Open an onscreen MuJoCo viewer window for debugging.",
    )
    parser.add_argument(
        "--optimize-ee-link-offset",
        action="store_true",
        help=(
            "Run sequential position optimization: first ee_link with third view only, "
            "then hand_mount with wrist view only."
        ),
    )
    parser.add_argument(
        "--ee-link-offset-search-range",
        type=float,
        default=0.03,
        help="Symmetric search range in meters for each ee_link local translation axis during optimization.",
    )
    parser.add_argument(
        "--ee-link-offset-bayes-init",
        type=int,
        default=8,
        help="Number of initial random evaluations for Bayesian optimization.",
    )
    parser.add_argument(
        "--ee-link-offset-bayes-iters",
        type=int,
        default=100,
        help="Total number of ee_link offset evaluations for Bayesian optimization.",
    )
    parser.add_argument(
        "--ee-link-offset-frame-stride",
        type=int,
        default=5,
        help="Evaluate every Nth frame from the real segmentation videos during optimization.",
    )
    parser.add_argument(
        "--ee-link-offset-max-frames",
        type=int,
        default=20,
        help="Maximum sampled frames per episode for optimization. Use <= 0 for no cap.",
    )
    return parser.parse_args()


def euler_xyz_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    assert SciRotation is not None
    return SciRotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_quat().astype(
        np.float64
    )


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    assert SciRotation is not None
    qq = np.asarray(list(q), dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qq))
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    return SciRotation.from_quat(qq / n).as_matrix().astype(np.float64)


def matrix_from_pose(pos_xyz: Iterable[float], quat_xyzw: Iterable[float]) -> np.ndarray:
    pos = np.asarray(list(pos_xyz), dtype=np.float64).reshape(3)
    rot = quat_xyzw_to_rot(np.asarray(list(quat_xyzw), dtype=np.float64).reshape(4))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    assert SciRotation is not None
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return SciRotation.from_matrix(m).as_quat().astype(np.float64)


def convert_transform_calib_to_sim(T_calib: np.ndarray) -> np.ndarray:
    return np.asarray(T_calib, dtype=np.float64).reshape(4, 4).copy()


def convert_pose_xyz_rpy_calib_to_sim(pose_xyz_rpy: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose_xyz_rpy, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        raise ValueError(
            f"Expected end-effector pose with 6 values [x,y,z,roll,pitch,yaw], got shape {arr.shape}"
        )
    return arr.copy()


def load_calibration_map(calibration_dir: Path) -> Dict[str, Calibration]:
    intr_path = calibration_dir / "camera_intrinsics.pkl"
    extr_path = calibration_dir / "transformations.pkl"
    if not intr_path.exists() or not extr_path.exists():
        raise FileNotFoundError(
            f"Missing calibration files in {calibration_dir} "
            f"(expected {intr_path.name} and {extr_path.name})."
        )

    with intr_path.open("rb") as f:
        intr_data = pickle.load(f)
    with extr_path.open("rb") as f:
        extr_data = pickle.load(f)

    extr_map: Dict[str, np.ndarray] = {}
    if isinstance(extr_data, list):
        for item in extr_data:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name = str(item[0])
                mat = np.asarray(item[1], dtype=np.float64)
                if mat.shape == (4, 4):
                    extr_map[name] = convert_transform_calib_to_sim(mat)

    out: Dict[str, Calibration] = {}
    if isinstance(intr_data, dict):
        for cam_name, pair in intr_data.items():
            if cam_name not in extr_map:
                continue
            if not isinstance(pair, (tuple, list)) or len(pair) < 1:
                continue
            K = np.asarray(pair[0], dtype=np.float64)
            dist = np.asarray(pair[1], dtype=np.float64) if len(pair) >= 2 else np.zeros((5,), dtype=np.float64)
            out[cam_name] = Calibration(K=K, dist=dist, extrinsic=extr_map[cam_name])

    if not out:
        raise RuntimeError(f"No valid calibration loaded from {calibration_dir}")
    return out


def build_fingertip_specs(
    link_name_to_idx: Dict[str, int],
) -> List[FingertipSpec]:
    specs: List[FingertipSpec] = []

    for finger in FINGER_ORDER:
        fallback_link, fallback_offset = FALLBACK_TIP_SPECS[finger]
        if fallback_link not in link_name_to_idx:
            raise RuntimeError(
                f'Could not resolve fingertip spec for "{finger}" (missing link {fallback_link}).'
            )
        specs.append(
            FingertipSpec(
                finger=finger,
                position_link=fallback_link,
                local_offset_xyz=fallback_offset.copy(),
                mask_link_candidates=[fallback_link],
            )
        )

    return specs


def pick_camera_name(args: argparse.Namespace) -> str:
    if args.camera_index is None:
        return args.camera_name
    if args.camera_index not in VIDEO_INDEX_TO_CAMERA:
        raise ValueError(
            f"Unknown --camera-index {args.camera_index}. "
            f"Valid values: {sorted(VIDEO_INDEX_TO_CAMERA)}"
        )
    return VIDEO_INDEX_TO_CAMERA[args.camera_index]


def resolve_dataset_camera_names(
    args: argparse.Namespace, calibration_map: Dict[str, Calibration]
) -> List[str]:
    preferred = [VIDEO_INDEX_TO_CAMERA[idx] for idx in sorted(VIDEO_INDEX_TO_CAMERA)]
    available = [name for name in preferred if name in calibration_map]
    if len(available) >= 2:
        return available
    selected = pick_camera_name(args)
    if selected not in calibration_map:
        raise KeyError(
            f'Camera "{selected}" not found in calibration. '
            f"Available: {sorted(calibration_map.keys())}"
        )
    return [selected]


def resolve_camera_mode(camera_name: str, mode_arg: str) -> str:
    if mode_arg != "auto":
        return mode_arg
    return CAMERA_EXTRINSIC_MODE.get(camera_name, "base_camera")


def resolve_camera_reference_link(camera_name: str, default_ee_link: str) -> str:
    return CAMERA_REFERENCE_LINK.get(camera_name, default_ee_link)


def convert_world_camera_transform_opencv_to_mujoco(T_world_camera: np.ndarray) -> np.ndarray:
    T_in = np.asarray(T_world_camera, dtype=np.float64).reshape(4, 4)
    T_out = T_in.copy()
    # OpenCV camera axes are x-right, y-down, z-forward, while MuJoCo cameras
    # follow the OpenGL convention x-right, y-up, z-back. Rotate the camera
    # frame 180 degrees around its local x-axis so the rendered view matches
    # the calibration extrinsics while keeping the camera center fixed.
    T_out[:3, :3] = T_in[:3, :3] @ OPENCV_CAMERA_TO_MUJOCO_CAMERA_ROT
    return T_out


def project_point_camera(
    point_world: np.ndarray, T_world_camera: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    T_camera_world = np.linalg.inv(T_world_camera)
    cam = (T_camera_world @ np.r_[point_world, 1.0])[:3]
    if cam[2] <= 1e-9:
        return cam, np.array([-1.0, -1.0], dtype=np.float64)
    u = K[0, 0] * (cam[0] / cam[2]) + K[0, 2]
    v = K[1, 1] * (cam[1] / cam[2]) + K[1, 2]
    return cam, np.array([u, v], dtype=np.float64)


def save_mask(path: Path, mask: np.ndarray) -> None:
    arr = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def discover_annotation_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.annotation_dir is not None:
        return args.annotation_dir
    if args.dataset_root is None:
        return None
    candidates = [args.dataset_root / "annotation", args.dataset_root / "annotations"]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find annotation folder under {args.dataset_root}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def parse_episode_id_tokens(tokens: Optional[List[str]], available_ids: Iterable[int]) -> List[int]:
    available_set = set(int(x) for x in available_ids)
    if not available_set:
        return []
    if tokens is None:
        return sorted(available_set)

    expanded_tokens: List[str] = []
    for token in tokens:
        parts = [p.strip() for p in token.split(",")]
        expanded_tokens.extend([p for p in parts if p])

    if any(tok.lower() == "all" for tok in expanded_tokens):
        return sorted(available_set)

    selected: set[int] = set()
    for token in expanded_tokens:
        if "-" in token:
            a, b = token.split("-", 1)
            if not a or not b:
                raise ValueError(f"Invalid episode id range token: {token!r}")
            lo = int(a)
            hi = int(b)
            if lo > hi:
                lo, hi = hi, lo
            selected.update(range(lo, hi + 1))
        else:
            selected.add(int(token))

    selected &= available_set
    return sorted(selected)


def resolve_dataset_base_dir(args: argparse.Namespace, annotation_dir: Path) -> Path:
    if args.dataset_root is not None:
        return args.dataset_root
    return annotation_dir.parent


def resolve_segmentation_video_path(
    annotation_data: Dict[str, object],
    dataset_base_dir: Path,
    view_index: int,
) -> Optional[Path]:
    seg_entries = annotation_data.get("segmentation_videos")
    if not isinstance(seg_entries, list):
        return None
    if view_index < 0 or view_index >= len(seg_entries):
        return None
    entry = seg_entries[view_index]
    rel_path: Optional[str] = None
    if isinstance(entry, dict):
        rel_path = entry.get("video_path")
    elif isinstance(entry, str):
        rel_path = entry
    if not rel_path:
        return None
    return (dataset_base_dir / rel_path).resolve()


def _resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return np.asarray(mask, dtype=bool)
    img = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    resized = img.resize((int(width), int(height)), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.uint8) > 127


def _resize_rgb_bilinear(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(frame_rgb, dtype=np.uint8)
    if arr.shape[:2] == (height, width):
        return arr
    img = Image.fromarray(arr, mode="RGB")
    resized = img.resize((int(width), int(height)), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def probe_video_frame_size(video_path: Path) -> Optional[Tuple[int, int]]:
    if not video_path.exists():
        return None
    
    try:
        video = media.read_video(str(video_path))
        return (video.shape[2], video.shape[1])
    except Exception:
        return None


def resolve_output_video_size(
    args: argparse.Namespace,
    ep_id: int,
    camera_name: str,
    seg_video_path: Optional[Path],
) -> Tuple[int, int]:
    dataset_video_size = (
        probe_video_frame_size(seg_video_path) if seg_video_path is not None else None
    )
    if dataset_video_size is not None:
        width = int(dataset_video_size[0])
        height = int(dataset_video_size[1])
        print(
            f"[episode {ep_id}][{camera_name}] segmentation video frame size: "
            f"{width}x{height} ({seg_video_path})"
        )
        return width, height

    width = int(args.width)
    height = int(args.height)
    if seg_video_path is not None:
        print(
            f"[episode {ep_id}][{camera_name}] could not read segmentation video frame size "
            f"from {seg_video_path}; falling back to args size {width}x{height}"
        )
    else:
        print(
            f"[episode {ep_id}][{camera_name}] no segmentation video found; "
            f"using args size {width}x{height}"
        )
    return width, height


def load_video_frames_mediapy(video_path: Path) -> np.ndarray:
    if media is None:
        raise ImportError(
            "mediapy is required for video loading. "
            f"Import error: {MEDIAPY_IMPORT_ERROR}"
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    frames = media.read_video(str(video_path))
    arr = np.asarray(frames, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB video frames from {video_path}, got shape {arr.shape}")
    return arr


def extract_green_hand_mask(frame_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected RGB frame, got shape {arr.shape}")
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (g >= 200) & (r <= 70) & (b <= 70)


def downscale_sim_mask_for_segmentation_video(mask: np.ndarray) -> np.ndarray:
    return _resize_mask_nearest(mask, width=240, height=135)


def sample_frame_indices(frame_count: int, stride: int, max_frames: int) -> List[int]:
    step = max(1, int(stride))
    indices = list(range(0, int(frame_count), step))
    if max_frames > 0 and len(indices) > int(max_frames):
        if int(max_frames) == 1:
            return [indices[0]]
        sample_pos = np.linspace(0, len(indices) - 1, int(max_frames))
        indices = [indices[int(round(v))] for v in sample_pos]
    return sorted(set(int(i) for i in indices if 0 <= int(i) < int(frame_count)))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def rbf_kernel_matrix(Xa: np.ndarray, Xb: np.ndarray, length_scale: float) -> np.ndarray:
    a = np.asarray(Xa, dtype=np.float64)
    b = np.asarray(Xb, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)
    ls2 = float(max(1e-12, length_scale * length_scale))
    return np.exp(-0.5 * sqdist / ls2)


def normal_pdf(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def normal_cdf(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))


def propose_bayes_opt_candidate(
    X: np.ndarray,
    y: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    random_state: np.random.Generator,
    num_candidates: int = 2048,
) -> np.ndarray:
    X_obs = np.asarray(X, dtype=np.float64)
    y_obs = np.asarray(y, dtype=np.float64).reshape(-1)
    if X_obs.shape[0] == 0:
        return random_state.uniform(bounds_lo, bounds_hi)

    span = np.maximum(bounds_hi - bounds_lo, 1e-9)
    length_scale = float(np.mean(span) * 0.35)
    noise = 1e-6

    K_xx = rbf_kernel_matrix(X_obs, X_obs, length_scale) + noise * np.eye(X_obs.shape[0], dtype=np.float64)
    try:
        K_inv = np.linalg.inv(K_xx)
    except np.linalg.LinAlgError:
        return random_state.uniform(bounds_lo, bounds_hi)

    X_cand = random_state.uniform(bounds_lo, bounds_hi, size=(int(num_candidates), X_obs.shape[1]))
    K_xs = rbf_kernel_matrix(X_obs, X_cand, length_scale)
    mu = K_xs.T @ (K_inv @ y_obs)
    var = 1.0 - np.sum(K_xs * (K_inv @ K_xs), axis=0)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    best = float(np.max(y_obs))
    z = (mu - best) / sigma
    ei = (mu - best) * normal_cdf(z) + sigma * normal_pdf(z)
    return np.asarray(X_cand[int(np.argmax(ei))], dtype=np.float64)


class MujocoHandSim:
    def __init__(self, model_path: Path, width: int, height: int) -> None:
        assert mujoco is not None
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        requested_width = int(width)
        requested_height = int(height)

        # Ensure offscreen framebuffer is large enough for requested render size.
        # MuJoCo defaults are often 640x480 when not specified in the model XML.
        try:
            self.model.vis.global_.offwidth = max(
                int(self.model.vis.global_.offwidth), requested_width
            )
            self.model.vis.global_.offheight = max(
                int(self.model.vis.global_.offheight), requested_height
            )
        except Exception:
            pass

        try:
            self.renderer = mujoco.Renderer(
                self.model, width=requested_width, height=requested_height
            )
            self.width = requested_width
            self.height = requested_height
        except ValueError as exc:
            max_w = int(getattr(self.model.vis.global_, "offwidth", requested_width))
            max_h = int(getattr(self.model.vis.global_, "offheight", requested_height))
            raise RuntimeError(
                f"Failed to create MuJoCo renderer at {requested_width}x{requested_height}. "
                f"Model framebuffer is {max_w}x{max_h}. "
                "Set a larger offscreen buffer in XML (<visual><global offwidth/offheight .../>) "
                "or reduce --width/--height."
            ) from exc

        self.joint_name_to_id: Dict[str, int] = {}
        self.joint_name_to_qpos_adr: Dict[str, int] = {}
        self.body_name_to_id: Dict[str, int] = {}
        self.free_joint_qpos_adr: Optional[int] = None

        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if name:
                self.body_name_to_id[name] = bid

        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jtype = int(self.model.jnt_type[jid])
            qadr = int(self.model.jnt_qposadr[jid])
            if name:
                self.joint_name_to_id[name] = jid
            if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                if not name:
                    continue
                self.joint_name_to_qpos_adr[name] = qadr
            elif jtype == int(mujoco.mjtJoint.mjJNT_FREE) and self.free_joint_qpos_adr is None:
                self.free_joint_qpos_adr = qadr

        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._camera)
        self._rgb_scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._rgb_scene_option)
        self._seg_scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._seg_scene_option)
        self._seg_scene_option.frame = int(mujoco.mjtFrame.mjFRAME_NONE)

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def reset_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.free_joint_qpos_adr is not None:
            adr = self.free_joint_qpos_adr
            self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.forward()

    def set_joint_value(self, joint_name: str, target: float) -> Optional[float]:
        jid = self.joint_name_to_id.get(joint_name)
        if jid is None:
            return None
        qadr = self.joint_name_to_qpos_adr.get(joint_name)
        if qadr is None:
            return None
        lower = float(self.model.jnt_range[jid, 0])
        upper = float(self.model.jnt_range[jid, 1])
        value = float(target)
        if lower <= upper:
            value = float(np.clip(value, lower, upper))
        self.data.qpos[qadr] = value
        return value

    def apply_joint_vector(self, joint_values: np.ndarray, joint_order: List[str]) -> Dict[str, float]:
        vals = np.asarray(joint_values, dtype=np.float64).reshape(-1)
        if vals.size < len(joint_order):
            raise ValueError(
                f"Expected at least {len(joint_order)} hand joint values, got {vals.size}."
            )
        out: Dict[str, float] = {}
        for i, joint_name in enumerate(joint_order):
            applied = self.set_joint_value(joint_name, float(vals[i]))
            if applied is not None:
                out[joint_name] = applied
        return out

    def set_base_pose(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> None:
        if self.free_joint_qpos_adr is None:
            return
        adr = self.free_joint_qpos_adr
        self.data.qpos[adr : adr + 3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
        q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        # MuJoCo free joint quaternion layout is [w, x, y, z].
        self.data.qpos[adr + 3 : adr + 7] = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

    def world_link_transform(self, body_name: str) -> np.ndarray:
        resolved = self.resolve_body_name(body_name)
        if resolved is None:
            raise KeyError(f'Body/link "{body_name}" not found in model.')
        bid = self.body_name_to_id[resolved]
        pos = np.asarray(self.data.xpos[bid], dtype=np.float64)
        rot = np.asarray(self.data.xmat[bid], dtype=np.float64).reshape(3, 3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def resolve_body_name(self, body_name: str) -> Optional[str]:
        if body_name in self.body_name_to_id:
            return body_name

        # Common pattern for compact imported models with *_jointbody naming.
        candidates = [
            f"{body_name}_jointbody",
            f"joint_{body_name}_jointbody",
        ]
        for cand in candidates:
            if cand in self.body_name_to_id:
                return cand

        # If ee_link looks like a joint name, map it to its associated jointbody.
        if body_name in self.joint_name_to_id:
            joint_body = f"{body_name}_jointbody"
            if joint_body in self.body_name_to_id:
                return joint_body

        # Suffix fallback to handle name mangling.
        suffix_matches = [name for name in self.body_name_to_id if name.endswith(body_name)]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    def get_body_local_pos(self, body_name: str) -> np.ndarray:
        resolved = self.resolve_body_name(body_name)
        if resolved is None:
            raise KeyError(f'Body/link "{body_name}" not found in model.')
        bid = self.body_name_to_id[resolved]
        return np.asarray(self.model.body_pos[bid], dtype=np.float64).copy()

    def set_body_local_pos(self, body_name: str, pos_xyz: np.ndarray) -> None:
        resolved = self.resolve_body_name(body_name)
        if resolved is None:
            raise KeyError(f'Body/link "{body_name}" not found in model.')
        bid = self.body_name_to_id[resolved]
        self.model.body_pos[bid] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)

    def _set_camera_from_extrinsic(
        self,
        calibration_camera_name: str,
        T_world_camera: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> None:
        # Convert ROS/OpenCV camera convention → MuJoCo/OpenGL camera convention.
        # T_mj is the camera pose in world space with MuJoCo axes (x-right, y-up, z-back).
        T_mj = convert_world_camera_transform_opencv_to_mujoco(T_world_camera)

        xml_camera_name = CALIB_CAMERA_TO_XML_CAMERA.get(calibration_camera_name, calibration_camera_name)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, xml_camera_name)
        if cam_id < 0:
            raise KeyError(
                f'Camera "{xml_camera_name}" not found in model for calibration view '
                f'"{calibration_camera_name}".'
            )

        # model.cam_pos/quat are in the PARENT BODY's local frame, not world frame.
        # For cameras parented to a non-world body (e.g. wrist_view_camera inside
        # the root body), we must transform T_mj into that body's local frame.
        parent_body_id = int(self.model.cam_bodyid[cam_id])
        if parent_body_id != 0:  # 0 = worldbody; world frame == local frame
            T_world_parent = np.eye(4, dtype=np.float64)
            T_world_parent[:3, :3] = np.asarray(
                self.data.xmat[parent_body_id], dtype=np.float64
            ).reshape(3, 3)
            T_world_parent[:3, 3] = np.asarray(self.data.xpos[parent_body_id], dtype=np.float64)
            T_local = np.linalg.inv(T_world_parent) @ T_mj
        else:
            T_local = T_mj

        assert SciRotation is not None
        r_local = SciRotation.from_matrix(T_local[:3, :3])
        q_local = r_local.as_quat()
        mj_quat_local = np.array([q_local[3], q_local[0], q_local[1], q_local[2]], dtype=np.float64)

        # Update model (local frame) so future mj_forward calls stay consistent.
        self.model.cam_pos[cam_id] = T_local[:3, 3]
        self.model.cam_quat[cam_id] = mj_quat_local

        # Also write the world-space values directly into data so the renderer
        # uses the correct pose without requiring an additional mj_forward call.
        self.data.cam_xpos[cam_id] = T_mj[:3, 3]
        self.data.cam_xmat[cam_id] = T_mj[:3, :3].ravel()

        self._camera.type = int(mujoco.mjtCamera.mjCAMERA_FIXED)
        self._camera.fixedcamid = int(cam_id)
        self._camera.trackbodyid = -1

        if K is not None and hasattr(self.model, "cam_intrinsic"):
            self.model.cam_intrinsic[cam_id] = np.array(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                dtype=np.float64,
            )

    def set_frame_visualization(self, enabled: bool, axis_length_m: float) -> None:
        if enabled:
            self._rgb_scene_option.frame = int(mujoco.mjtFrame.mjFRAME_BODY)
            axis_len = float(max(1e-4, axis_length_m))
            try:
                self.model.vis.scale.framelength = axis_len
            except Exception:
                pass
            try:
                self.model.vis.scale.framewidth = max(1e-5, axis_len * 0.2)
            except Exception:
                pass
        else:
            self._rgb_scene_option.frame = int(mujoco.mjtFrame.mjFRAME_NONE)

    def render_rgb_and_seg(
        self,
        camera_state: CameraState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if camera_state.T_world_camera is None or camera_state.calibration_camera_name is None:
            raise ValueError("Calibrated rendering requires T_world_camera.")
        self._set_camera_from_extrinsic(
            camera_state.calibration_camera_name,
            camera_state.T_world_camera,
            camera_state.K,
        )

        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(
            self.data,
            camera=self._camera,
            scene_option=self._rgb_scene_option,
        )
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)

        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(
            self.data,
            camera=self._camera,
            scene_option=self._seg_scene_option,
        )
        seg_raw = np.asarray(self.renderer.render())

        geom_ids = self._extract_geom_ids(seg_raw)
        return rgb, geom_ids

    def sync_viewer(self, viewer: object, camera_state: CameraState) -> None:
        if camera_state.T_world_camera is None or camera_state.calibration_camera_name is None:
            return
        self._set_camera_from_extrinsic(
            camera_state.calibration_camera_name,
            camera_state.T_world_camera,
            camera_state.K,
        )
        if hasattr(viewer, "lock"):
            with viewer.lock():
                viewer.cam.type = self._camera.type
                viewer.cam.fixedcamid = self._camera.fixedcamid
                viewer.cam.trackbodyid = self._camera.trackbodyid
                viewer.cam.lookat[:] = self._camera.lookat
                viewer.cam.distance = self._camera.distance
                viewer.cam.azimuth = self._camera.azimuth
                viewer.cam.elevation = self._camera.elevation
                viewer.opt.frame = self._rgb_scene_option.frame
        viewer.sync()

    def _extract_geom_ids(self, seg_raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(seg_raw)
        if arr.ndim == 2:
            return arr.astype(np.int32)
        if arr.ndim == 3 and arr.shape[2] >= 2:
            c0 = arr[:, :, 0].astype(np.int32)
            c1 = arr[:, :, 1].astype(np.int32)
            c0_good = np.mean((c0 >= -1) & (c0 < self.model.ngeom))
            c1_good = np.mean((c1 >= -1) & (c1 < self.model.ngeom))
            return c0 if c0_good >= c1_good else c1
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.int32)
        raise RuntimeError(f"Unexpected segmentation output shape: {arr.shape}")



def compute_world_camera_transform(
    mode: str,
    extrinsic: np.ndarray,
    sim: MujocoHandSim,
    ee_link: str,
    ee_offset: np.ndarray,
) -> np.ndarray:
    if mode == "base_camera":
        return extrinsic

    T_world_ee = sim.world_link_transform(ee_link)
    T_world_prev = T_world_ee.copy()
    T_world_prev[:3, 3] = T_world_ee[:3, 3] - T_world_ee[:3, :3] @ ee_offset
    return T_world_prev @ extrinsic


def score_extrinsic_direction_for_pose(
    mode: str,
    extrinsic: np.ndarray,
    sim: MujocoHandSim,
    ee_link: str,
    ee_offset: np.ndarray,
    points_world: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int, np.ndarray]:
    T_world_camera = compute_world_camera_transform(
        mode=mode,
        extrinsic=extrinsic,
        sim=sim,
        ee_link=ee_link,
        ee_offset=ee_offset,
    )
    in_frame = 0
    visible = 0
    for i in range(points_world.shape[0]):
        cam_xyz, pix = project_point_camera(points_world[i], T_world_camera, K)
        if cam_xyz[2] > 1e-9:
            visible += 1
            if 0 <= pix[0] < width and 0 <= pix[1] < height:
                in_frame += 1
    return in_frame, visible, T_world_camera


def decode_hand_segmentation_from_geom_ids(
    geom_ids: np.ndarray,
    model,
    hand_body_ids: set[int],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    geom = np.asarray(geom_ids, dtype=np.int32)
    if np.all(geom < 0):
        h, w = geom.shape[:2]
        return np.zeros((h, w), dtype=bool), np.full((h, w), -1, dtype=np.int32), True

    link_idx_map = np.full_like(geom, -1, dtype=np.int32)
    valid = (geom >= 0) & (geom < int(model.ngeom))
    flat_geom = geom[valid]
    body_ids = np.asarray(model.geom_bodyid, dtype=np.int32)[flat_geom]
    link_idx_map[valid] = body_ids
    hand_mask = np.isin(link_idx_map, np.asarray(sorted(hand_body_ids), dtype=np.int32))
    return hand_mask, link_idx_map, False


def apply_end_effector_pose_to_ee_link(
    sim: MujocoHandSim,
    ee_link: str,
    target_xyz_rpy: np.ndarray,
    ee_offset_xyz: np.ndarray,
) -> Dict[str, object]:
    arr_raw = np.asarray(target_xyz_rpy, dtype=np.float64).reshape(-1)
    arr = convert_pose_xyz_rpy_calib_to_sim(arr_raw)

    assert SciRotation is not None
    R_world_ee_target = SciRotation.from_euler("xyz", arr[3:6], degrees=False).as_matrix().astype(
        np.float64
    )

    ee_offset = np.asarray(ee_offset_xyz, dtype=np.float64).reshape(3)
    T_world_ee_target = np.eye(4, dtype=np.float64)
    T_world_ee_target[:3, :3] = R_world_ee_target
    T_world_ee_target[:3, 3] = np.asarray(arr[:3], dtype=np.float64)

    resolved_ee_link = sim.resolve_body_name(ee_link)
    if resolved_ee_link is None:
        raise KeyError(f'EE link "{ee_link}" could not be resolved to a body in the model.')

    T_world_root_current = sim.world_link_transform("root")
    T_world_ee_current = sim.world_link_transform(resolved_ee_link)
    T_root_ee = np.linalg.inv(T_world_root_current) @ T_world_ee_current
    T_root_ee_adjusted = T_root_ee.copy()
    T_root_ee_adjusted[:3, 3] = T_root_ee_adjusted[:3, 3] + ee_offset

    T_world_root_target = T_world_ee_target @ np.linalg.inv(T_root_ee_adjusted)

    base_quat = SciRotation.from_matrix(T_world_root_target[:3, :3]).as_quat().astype(np.float64)
    base_pos_np = np.asarray(T_world_root_target[:3, 3], dtype=np.float64)
    sim.set_base_pose(base_pos_np, base_quat)

    return {
        "ee_link": resolved_ee_link,
        "target_ee_xyz_rpy_input_calib": arr_raw[:6].tolist(),
        "target_ee_xyz_rpy_used_sim": arr[:6].tolist(),
        "ee_offset_xyz": ee_offset.tolist(),
        "applied_base_position": base_pos_np.tolist(),
        "applied_base_quaternion_xyzw": base_quat.tolist(),
    }


def _project_tip_for_camera(
    tip_world: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return project_point_camera(tip_world, T_world_camera, K)


def draw_body_frame_overlay(
    rgb: np.ndarray,
    T_world_body: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
    axis_length_m: float,
    origin_fill_rgb: Tuple[int, int, int] = (255, 255, 255),
    origin_outline_rgb: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    origin = np.asarray(T_world_body[:3, 3], dtype=np.float64)
    rot = np.asarray(T_world_body[:3, :3], dtype=np.float64).reshape(3, 3)
    axes = (
        ("x", np.array([255, 64, 64], dtype=np.uint8)),
        ("y", np.array([64, 220, 64], dtype=np.uint8)),
        ("z", np.array([64, 128, 255], dtype=np.uint8)),
    )

    origin_cam, origin_px = project_point_camera(origin, T_world_camera, K)
    if float(origin_cam[2]) <= 1e-9:
        return rgb

    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    origin_xy = (float(origin_px[0]), float(origin_px[1]))
    radius = 4

    for axis_idx, (_, color) in enumerate(axes):
        endpoint = origin + rot[:, axis_idx] * float(axis_length_m)
        endpoint_cam, endpoint_px = project_point_camera(endpoint, T_world_camera, K)
        if float(endpoint_cam[2]) <= 1e-9:
            continue
        endpoint_xy = (float(endpoint_px[0]), float(endpoint_px[1]))
        draw.line([origin_xy, endpoint_xy], fill=tuple(int(v) for v in color), width=4)

    draw.ellipse(
        [
            origin_xy[0] - radius,
            origin_xy[1] - radius,
            origin_xy[0] + radius,
            origin_xy[1] + radius,
        ],
        fill=origin_fill_rgb,
        outline=origin_outline_rgb,
    )
    return np.asarray(pil_img, dtype=np.uint8)


def load_episode_sequences(
    ann: Dict[str, object],
    ep_id: int,
    ann_path: Path,
    args: argparse.Namespace,
) -> Tuple[str, np.ndarray, np.ndarray, int]:
    joint_key = (
        "action.hand_joint_position"
        if args.hand_joint_source == "action"
        else "observation.state.hand_joint_position"
    )
    if joint_key not in ann:
        fallback_key = (
            "observation.state.hand_joint_position"
            if joint_key == "action.hand_joint_position"
            else "action.hand_joint_position"
        )
        if fallback_key in ann:
            print(f"[warning][episode {ep_id}] Missing {joint_key}; using {fallback_key} instead.")
            joint_key = fallback_key
        else:
            raise KeyError(
                f"[episode {ep_id}] Missing joint sequence key {joint_key} and fallback {fallback_key}"
            )

    if args.palm_pose_key not in ann:
        raise KeyError(
            f"[episode {ep_id}] Missing end-effector pose key {args.palm_pose_key} in {ann_path}"
        )

    joint_seq = np.asarray(ann[joint_key], dtype=np.float64)
    ee_pose_seq = np.asarray(ann[args.palm_pose_key], dtype=np.float64)
    if joint_seq.ndim != 2 or ee_pose_seq.ndim != 2:
        raise ValueError(
            f"[episode {ep_id}] Expected 2D sequences, got joint={joint_seq.shape}, ee_pose={ee_pose_seq.shape}"
        )

    frame_count = min(joint_seq.shape[0], ee_pose_seq.shape[0])
    if args.max_frames_per_episode is not None and args.max_frames_per_episode > 0:
        frame_count = min(frame_count, int(args.max_frames_per_episode))
    return joint_key, joint_seq, ee_pose_seq, frame_count


def build_episode_camera_targets(
    args: argparse.Namespace,
    ep_id: int,
    ann: Dict[str, object],
    dataset_base_dir: Path,
    calibration_map: Dict[str, Calibration],
    render_camera_names: List[str],
) -> List[DatasetCameraTarget]:
    episode_out = args.output_dir / str(ep_id)
    episode_out.mkdir(parents=True, exist_ok=True)

    camera_targets: List[DatasetCameraTarget] = []
    for render_camera_name in render_camera_names:
        view_index = CAMERA_TO_VIDEO_INDEX.get(render_camera_name, -1)
        episode_seg_video_path: Optional[Path] = None
        if view_index >= 0:
            episode_seg_video_path = resolve_segmentation_video_path(
                annotation_data=ann,
                dataset_base_dir=dataset_base_dir,
                view_index=view_index,
            )
            if episode_seg_video_path is None:
                fallback_seg = dataset_base_dir / "segmentation_videos" / str(ep_id) / f"{view_index}.mp4"
                if fallback_seg.exists():
                    episode_seg_video_path = fallback_seg.resolve()

        output_video_width, output_video_height = resolve_output_video_size(
            args=args,
            ep_id=ep_id,
            camera_name=render_camera_name,
            seg_video_path=episode_seg_video_path,
        )

        camera_episode_out = episode_out / render_camera_name
        label_map_dir = camera_episode_out / "label_map"
        per_finger_base_dir = camera_episode_out / "finger_masks"
        camera_episode_out.mkdir(parents=True, exist_ok=True)
        label_map_dir.mkdir(parents=True, exist_ok=True)
        if args.save_per_finger_masks:
            per_finger_base_dir.mkdir(parents=True, exist_ok=True)
            for finger in FINGER_ORDER:
                (per_finger_base_dir / finger).mkdir(parents=True, exist_ok=True)

        camera_targets.append(
            DatasetCameraTarget(
                camera_name=render_camera_name,
                calib=calibration_map[render_camera_name],
                mode=resolve_camera_mode(render_camera_name, args.camera_mode),
                ee_offset=np.zeros(3, dtype=np.float64),
                real_seg_video_path=episode_seg_video_path,
                view_index=view_index,
                output_video_width=output_video_width,
                output_video_height=output_video_height,
                episode_out=camera_episode_out,
                rgb_video_path=camera_episode_out / "rgb.mp4",
                hand_mask_video_path=camera_episode_out / "hand_mask.mp4",
                vis_seg_actions_video_path=None,
                label_map_dir=label_map_dir,
                per_finger_base_dir=per_finger_base_dir,
                lines_path=camera_episode_out / "fingertips_and_pose.jsonl",
            )
        )
    return camera_targets


def build_reference_world_points(
    sim: MujocoHandSim,
    ee_link: str,
    specs: List[FingertipSpec],
) -> np.ndarray:
    points_world: List[np.ndarray] = []
    resolved_ee_link = sim.resolve_body_name(ee_link)
    if resolved_ee_link is not None:
        T_world_ee = sim.world_link_transform(resolved_ee_link)
        points_world.append(T_world_ee[:3, 3].copy())
    for spec in specs:
        T_world_link = sim.world_link_transform(spec.position_link)
        tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz
        points_world.append(tip_world)
    return np.asarray(points_world, dtype=np.float64)


def choose_camera_extrinsic(
    args: argparse.Namespace,
    sim: MujocoHandSim,
    target: DatasetCameraTarget,
    ee_link: str,
    points_world: np.ndarray,
) -> Tuple[np.ndarray, str]:
    extrinsic_raw = np.asarray(target.calib.extrinsic, dtype=np.float64)
    if args.invert_extrinsic:
        return np.linalg.inv(extrinsic_raw), "forced_inverse"
    if args.disable_auto_extrinsic_direction:
        return extrinsic_raw, "forced_raw"

    raw_in, raw_vis, _ = score_extrinsic_direction_for_pose(
        mode=target.mode,
        extrinsic=extrinsic_raw,
        sim=sim,
        ee_link=resolve_camera_reference_link(target.camera_name, ee_link),
        ee_offset=target.ee_offset,
        points_world=points_world,
        K=target.calib.K,
        width=args.width,
        height=args.height,
    )
    inv_extrinsic = np.linalg.inv(extrinsic_raw)
    inv_in, inv_vis, _ = score_extrinsic_direction_for_pose(
        mode=target.mode,
        extrinsic=inv_extrinsic,
        sim=sim,
        ee_link=resolve_camera_reference_link(target.camera_name, ee_link),
        ee_offset=target.ee_offset,
        points_world=points_world,
        K=target.calib.K,
        width=args.width,
        height=args.height,
    )
    if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
        return extrinsic_raw, f"auto_raw(in_frame={raw_in},visible={raw_vis})"
    return inv_extrinsic, f"auto_inverse(in_frame={inv_in},visible={inv_vis})"


def prepare_optimization_episode_data(
    args: argparse.Namespace,
    ep_id: int,
    ann_path: Path,
    ann: Dict[str, object],
    dataset_base_dir: Path,
    calibration_map: Dict[str, Calibration],
    render_camera_names: List[str],
) -> OptimizationEpisodeData:
    joint_key, joint_seq, ee_pose_seq, frame_count = load_episode_sequences(
        ann=ann,
        ep_id=ep_id,
        ann_path=ann_path,
        args=args,
    )
    if frame_count <= 0:
        raise RuntimeError(f"[episode {ep_id}] No frames available for optimization.")

    camera_targets = build_episode_camera_targets(
        args=args,
        ep_id=ep_id,
        ann=ann,
        dataset_base_dir=dataset_base_dir,
        calibration_map=calibration_map,
        render_camera_names=render_camera_names,
    )
    if len(camera_targets) < 1:
        raise RuntimeError(
            "Body-position optimization requires at least one camera view to be available."
        )

    real_masks_by_camera: Dict[str, Dict[int, np.ndarray]] = {}
    max_shared_frames = int(frame_count)
    for target in camera_targets:
        if target.real_seg_video_path is None:
            raise FileNotFoundError(
                f"[episode {ep_id}][{target.camera_name}] Missing real segmentation video."
            )
        frames = load_video_frames_mediapy(target.real_seg_video_path)
        masks = [extract_green_hand_mask(frame) for frame in frames]
        max_shared_frames = min(max_shared_frames, len(masks))
        real_masks_by_camera[target.camera_name] = {
            idx: np.asarray(mask, dtype=bool) for idx, mask in enumerate(masks)
        }

    frame_indices = sample_frame_indices(
        frame_count=max_shared_frames,
        stride=int(args.ee_link_offset_frame_stride),
        max_frames=int(args.ee_link_offset_max_frames),
    )
    if not frame_indices:
        raise RuntimeError(f"[episode {ep_id}] No optimization frames selected.")

    return OptimizationEpisodeData(
        ep_id=ep_id,
        ann_path=ann_path,
        joint_key=joint_key,
        joint_seq=joint_seq,
        ee_pose_seq=ee_pose_seq,
        frame_count=frame_count,
        frame_indices=frame_indices,
        real_masks_by_camera=real_masks_by_camera,
        camera_targets=camera_targets,
    )


def evaluate_body_local_pos_candidate(
    sim: MujocoHandSim,
    args: argparse.Namespace,
    specs: List[FingertipSpec],
    hand_body_ids: set[int],
    optimization_episodes: List[OptimizationEpisodeData],
    body_name: str,
    candidate_local_pos: np.ndarray,
) -> float:
    sim.set_body_local_pos(body_name, candidate_local_pos)
    total_iou = 0.0
    total_count = 0

    for episode in optimization_episodes:
        zero_joint_values = np.zeros_like(episode.joint_seq[0], dtype=np.float64)

        def joint_values_for_frame(frame_idx: int) -> np.ndarray:
            if frame_idx == 0:
                return zero_joint_values
            return np.asarray(episode.joint_seq[frame_idx], dtype=np.float64)

        sim.reset_pose()
        sim.apply_joint_vector(joint_values_for_frame(0), HAND_JOINT_ORDER)
        apply_end_effector_pose_to_ee_link(
            sim,
            args.ee_link,
            episode.ee_pose_seq[0],
            episode.camera_targets[0].ee_offset,
        )
        sim.forward()

        points_world = build_reference_world_points(sim, args.ee_link, specs)
        chosen_extrinsics = {
            target.camera_name: choose_camera_extrinsic(
                args=args,
                sim=sim,
                target=target,
                ee_link=args.ee_link,
                points_world=points_world,
            )[0]
            for target in episode.camera_targets
        }

        for frame_idx in episode.frame_indices:
            sim.apply_joint_vector(joint_values_for_frame(frame_idx), HAND_JOINT_ORDER)
            apply_end_effector_pose_to_ee_link(
                sim,
                args.ee_link,
                episode.ee_pose_seq[frame_idx],
                episode.camera_targets[0].ee_offset,
            )
            sim.forward()

            for target in episode.camera_targets:
                T_world_camera = compute_world_camera_transform(
                    mode=target.mode,
                    extrinsic=np.asarray(chosen_extrinsics[target.camera_name], dtype=np.float64),
                    sim=sim,
                    ee_link=resolve_camera_reference_link(target.camera_name, args.ee_link),
                    ee_offset=target.ee_offset,
                )
                camera_state = CameraState(
                    kind="calibrated",
                    calibration_camera_name=target.camera_name,
                    T_world_camera=T_world_camera,
                    K=target.calib.K,
                )
                _, geom_ids = sim.render_rgb_and_seg(camera_state=camera_state)
                hand_mask, _, _ = decode_hand_segmentation_from_geom_ids(
                    geom_ids,
                    sim.model,
                    hand_body_ids,
                )
                sim_mask = downscale_sim_mask_for_segmentation_video(hand_mask)
                real_mask = episode.real_masks_by_camera[target.camera_name][frame_idx]
                total_iou += mask_iou(sim_mask, real_mask)
                total_count += 1

    if total_count <= 0:
        raise RuntimeError("No camera/frame pairs were evaluated during body-position optimization.")
    return float(total_iou) / float(total_count)


def optimize_body_local_position(
    sim: MujocoHandSim,
    args: argparse.Namespace,
    specs: List[FingertipSpec],
    hand_body_ids: set[int],
    optimization_episodes: List[OptimizationEpisodeData],
    body_name: str,
    log_name: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    base_local_pos = sim.get_body_local_pos(body_name)
    search_range = float(max(1e-5, args.ee_link_offset_search_range))
    bounds_lo = -np.full((3,), search_range, dtype=np.float64)
    bounds_hi = np.full((3,), search_range, dtype=np.float64)
    total_evals = int(max(1, args.ee_link_offset_bayes_iters))
    init_evals = int(max(1, min(args.ee_link_offset_bayes_init, total_evals)))

    rng = np.random.default_rng(0)
    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    for eval_idx in range(total_evals):
        if eval_idx == 0:
            delta = np.zeros((3,), dtype=np.float64)
        elif eval_idx < init_evals:
            delta = rng.uniform(bounds_lo, bounds_hi)
        else:
            delta = propose_bayes_opt_candidate(
                X=np.asarray(X_list, dtype=np.float64),
                y=np.asarray(y_list, dtype=np.float64),
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                random_state=rng,
            )

        candidate_local_pos = base_local_pos + np.asarray(delta, dtype=np.float64)
        score = evaluate_body_local_pos_candidate(
            sim=sim,
            args=args,
            specs=specs,
            hand_body_ids=hand_body_ids,
            optimization_episodes=optimization_episodes,
            body_name=body_name,
            candidate_local_pos=candidate_local_pos,
        )
        X_list.append(np.asarray(delta, dtype=np.float64))
        y_list.append(float(score))
        print(
            f"[optimize][{log_name}] eval={eval_idx + 1}/{total_evals} "
            f"delta={np.asarray(delta, dtype=np.float64).tolist()} mean_iou={score:.6f}"
        )

    best_idx = int(np.argmax(np.asarray(y_list, dtype=np.float64)))
    best_delta = np.asarray(X_list[best_idx], dtype=np.float64)
    best_local_pos = base_local_pos + best_delta
    best_score = float(y_list[best_idx])
    sim.set_body_local_pos(body_name, best_local_pos)
    sim.reset_pose()
    print(
        f"[optimize][{log_name}] best_local_pos={best_local_pos.tolist()} "
        f"best_delta={best_delta.tolist()} mean_iou={best_score:.6f}"
    )
    return best_local_pos, best_delta, best_score


def run_dataset_mode(args: argparse.Namespace) -> None:
    if args.dataset_root is None and args.annotation_dir is None:
        raise ValueError("Dataset mode requires --dataset-root or --annotation-dir.")
    if len(args.hand_mask_color) != 3 or any((c < 0 or c > 255) for c in args.hand_mask_color):
        raise ValueError("--hand-mask-color expects exactly 3 integers in [0, 255].")
    if media is None:
        raise ImportError(
            "mediapy is required for dataset-mode video export. "
            f"Import error: {MEDIAPY_IMPORT_ERROR}"
        )

    annotation_dir = discover_annotation_dir(args)
    assert annotation_dir is not None
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    dataset_base_dir = resolve_dataset_base_dir(args, annotation_dir)
    vis_seg_actions_root = (
        args.vis_seg_actions_dir
        if args.vis_seg_actions_dir is not None
        else (dataset_base_dir / "vis_seg_actions")
    )
    vis_seg_actions_root.mkdir(parents=True, exist_ok=True)

    files_by_id: Dict[int, Path] = {}
    for pth in annotation_dir.glob("*.json"):
        try:
            ep_id = int(pth.stem)
        except ValueError:
            continue
        files_by_id[ep_id] = pth
    if not files_by_id:
        raise FileNotFoundError(f"No numeric annotation json files found in {annotation_dir}")

    selected_ids = parse_episode_id_tokens(args.episode_ids, files_by_id.keys())
    if not selected_ids:
        raise RuntimeError("No matching episode ids selected.")

    calibration_map = load_calibration_map(args.calibration_dir)
    render_camera_names = resolve_dataset_camera_names(args, calibration_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = MujocoHandSim(args.model_path, width=args.width, height=args.height)
    viewer = None
    try:
        if args.show_viewer:
            if mujoco_viewer is None:
                raise ImportError(
                    "mujoco.viewer is required for --show-viewer. "
                    f"Import error: {MUJOCO_VIEWER_IMPORT_ERROR}"
                )
            viewer = mujoco_viewer.launch_passive(sim.model, sim.data)
        sim.set_frame_visualization(
            enabled=bool(not args.no_hand_root_frame_overlay),
            axis_length_m=float(args.hand_root_frame_axis_length),
        )
        specs = build_fingertip_specs(sim.body_name_to_id)
        excluded_body_ids = {
            int(sim.body_name_to_id[name])
            for name in EXCLUDED_HAND_MASK_LINK_NAMES
            if name in sim.body_name_to_id
        }
        hand_body_ids = set(sim.body_name_to_id.values()) - excluded_body_ids

        if args.optimize_ee_link_offset:
            stage_camera_specs = [
                ("ee_link", args.ee_link, [EE_LINK_OPTIMIZATION_CAMERA]),
                ("hand_mount", HAND_MOUNT_BODY_NAME, [HAND_MOUNT_OPTIMIZATION_CAMERA]),
            ]
            for stage_name, body_name, optimization_camera_names in stage_camera_specs:
                missing = [name for name in optimization_camera_names if name not in calibration_map]
                if missing:
                    raise RuntimeError(
                        f"Missing calibration for optimization stage {stage_name}: required cameras {missing}"
                    )
                optimization_episodes: List[OptimizationEpisodeData] = []
                print(
                    f"[optimize][{stage_name}] preparing episodes using cameras={optimization_camera_names}"
                )
                for ep_id in selected_ids:
                    ann_path = files_by_id[ep_id]
                    with ann_path.open("r", encoding="utf-8") as f:
                        ann = json.load(f)
                    optimization_episodes.append(
                        prepare_optimization_episode_data(
                            args=args,
                            ep_id=ep_id,
                            ann_path=ann_path,
                            ann=ann,
                            dataset_base_dir=dataset_base_dir,
                            calibration_map=calibration_map,
                            render_camera_names=optimization_camera_names,
                        )
                    )

                optimize_body_local_position(
                    sim=sim,
                    args=args,
                    specs=specs,
                    hand_body_ids=hand_body_ids,
                    optimization_episodes=optimization_episodes,
                    body_name=body_name,
                    log_name=stage_name,
                )

        for ep_id in selected_ids:
            ann_path = files_by_id[ep_id]
            with ann_path.open("r", encoding="utf-8") as f:
                ann = json.load(f)

            joint_key = (
                "action.hand_joint_position"
                if args.hand_joint_source == "action"
                else "observation.state.hand_joint_position"
            )
            if joint_key not in ann:
                fallback_key = (
                    "observation.state.hand_joint_position"
                    if joint_key == "action.hand_joint_position"
                    else "action.hand_joint_position"
                )
                if fallback_key in ann:
                    print(f"[warning][episode {ep_id}] Missing {joint_key}; using {fallback_key} instead.")
                    joint_key = fallback_key
                else:
                    raise KeyError(
                        f"[episode {ep_id}] Missing joint sequence key {joint_key} and fallback {fallback_key}"
                    )

            if args.palm_pose_key not in ann:
                raise KeyError(
                    f"[episode {ep_id}] Missing end-effector pose key {args.palm_pose_key} in {ann_path}"
                )

            joint_seq = np.asarray(ann[joint_key], dtype=np.float64)
            ee_pose_seq = np.asarray(ann[args.palm_pose_key], dtype=np.float64)
            if joint_seq.ndim != 2 or ee_pose_seq.ndim != 2:
                raise ValueError(
                    f"[episode {ep_id}] Expected 2D sequences, got joint={joint_seq.shape}, ee_pose={ee_pose_seq.shape}"
                )

            zero_joint_values = np.zeros_like(joint_seq[0], dtype=np.float64)

            def joint_values_for_frame(frame_idx: int) -> np.ndarray:
                if frame_idx == 0:
                    return zero_joint_values
                return np.asarray(joint_seq[frame_idx], dtype=np.float64)

            frame_count = min(joint_seq.shape[0], ee_pose_seq.shape[0])
            if args.max_frames_per_episode is not None and args.max_frames_per_episode > 0:
                frame_count = min(frame_count, int(args.max_frames_per_episode))
            if frame_count <= 0:
                print(f"[warning][episode {ep_id}] No frames to render; skipping.")
                continue

            episode_out = args.output_dir / str(ep_id)
            episode_out.mkdir(parents=True, exist_ok=True)

            sim.reset_pose()
            camera_targets: List[DatasetCameraTarget] = []
            for render_camera_name in render_camera_names:
                view_index = CAMERA_TO_VIDEO_INDEX.get(render_camera_name, -1)
                episode_seg_video_path: Optional[Path] = None
                if view_index >= 0:
                    episode_seg_video_path = resolve_segmentation_video_path(
                        annotation_data=ann,
                        dataset_base_dir=dataset_base_dir,
                        view_index=view_index,
                    )
                    if episode_seg_video_path is None:
                        fallback_seg = dataset_base_dir / "segmentation_videos" / str(ep_id) / f"{view_index}.mp4"
                        if fallback_seg.exists():
                            episode_seg_video_path = fallback_seg.resolve()

                output_video_width, output_video_height = resolve_output_video_size(
                    args=args,
                    ep_id=ep_id,
                    camera_name=render_camera_name,
                    seg_video_path=episode_seg_video_path,
                )

                camera_episode_out = episode_out / render_camera_name
                label_map_dir = camera_episode_out / "label_map"
                per_finger_base_dir = camera_episode_out / "finger_masks"
                vis_episode_out = vis_seg_actions_root / str(ep_id)
                vis_episode_out.mkdir(parents=True, exist_ok=True)
                vis_file_stem = (
                    str(view_index)
                    if view_index >= 0
                    else render_camera_name
                )
                vis_seg_actions_video_path = vis_episode_out / f"{vis_file_stem}.mp4"
                camera_episode_out.mkdir(parents=True, exist_ok=True)
                label_map_dir.mkdir(parents=True, exist_ok=True)
                if args.save_per_finger_masks:
                    per_finger_base_dir.mkdir(parents=True, exist_ok=True)
                    for finger in FINGER_ORDER:
                        (per_finger_base_dir / finger).mkdir(parents=True, exist_ok=True)

                camera_targets.append(
                    DatasetCameraTarget(
                        camera_name=render_camera_name,
                        calib=calibration_map[render_camera_name],
                        mode=resolve_camera_mode(render_camera_name, args.camera_mode),
                        ee_offset=np.zeros(3, dtype=np.float64),
                        real_seg_video_path=episode_seg_video_path,
                        view_index=view_index,
                        output_video_width=output_video_width,
                        output_video_height=output_video_height,
                        episode_out=camera_episode_out,
                        rgb_video_path=camera_episode_out / "rgb.mp4",
                        hand_mask_video_path=camera_episode_out / "hand_mask.mp4",
                        vis_seg_actions_video_path=vis_seg_actions_video_path,
                        label_map_dir=label_map_dir,
                        per_finger_base_dir=per_finger_base_dir,
                        lines_path=camera_episode_out / "fingertips_and_pose.jsonl",
                    )
                )

            sim.apply_joint_vector(joint_values_for_frame(0), HAND_JOINT_ORDER)
            apply_end_effector_pose_to_ee_link(
                sim,
                args.ee_link,
                ee_pose_seq[0],
                camera_targets[0].ee_offset,
            )
            sim.forward()

            sample_points_world: List[np.ndarray] = []
            resolved_ee_link = sim.resolve_body_name(args.ee_link)
            if resolved_ee_link is not None:
                T_world_ee = sim.world_link_transform(resolved_ee_link)
                sample_points_world.append(T_world_ee[:3, 3].copy())
            for spec in specs:
                T_world_link = sim.world_link_transform(spec.position_link)
                tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz
                sample_points_world.append(tip_world)
            pts = np.asarray(sample_points_world, dtype=np.float64)

            camera_runtime: Dict[str, Dict[str, object]] = {}
            for target in camera_targets:
                chosen_extrinsic: np.ndarray
                extrinsic_direction_mode = "not_applicable"
                extrinsic_raw = np.asarray(target.calib.extrinsic, dtype=np.float64)
                if args.invert_extrinsic:
                    chosen_extrinsic = np.linalg.inv(extrinsic_raw)
                    extrinsic_direction_mode = "forced_inverse"
                elif args.disable_auto_extrinsic_direction:
                    chosen_extrinsic = extrinsic_raw
                    extrinsic_direction_mode = "forced_raw"
                else:
                    raw_in, raw_vis, _ = score_extrinsic_direction_for_pose(
                        mode=target.mode,
                        extrinsic=extrinsic_raw,
                        sim=sim,
                        ee_link=resolve_camera_reference_link(target.camera_name, args.ee_link),
                        ee_offset=target.ee_offset,
                        points_world=pts,
                        K=target.calib.K,
                        width=args.width,
                        height=args.height,
                    )
                    inv_extrinsic = np.linalg.inv(extrinsic_raw)
                    inv_in, inv_vis, _ = score_extrinsic_direction_for_pose(
                        mode=target.mode,
                        extrinsic=inv_extrinsic,
                        sim=sim,
                        ee_link=resolve_camera_reference_link(target.camera_name, args.ee_link),
                        ee_offset=target.ee_offset,
                        points_world=pts,
                        K=target.calib.K,
                        width=args.width,
                        height=args.height,
                    )
                    if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
                        chosen_extrinsic = extrinsic_raw
                        extrinsic_direction_mode = f"auto_raw(in_frame={raw_in},visible={raw_vis})"
                    else:
                        chosen_extrinsic = inv_extrinsic
                        extrinsic_direction_mode = f"auto_inverse(in_frame={inv_in},visible={inv_vis})"

                camera_runtime[target.camera_name] = {
                    "target": target,
                    "chosen_extrinsic": chosen_extrinsic,
                    "extrinsic_direction_mode": extrinsic_direction_mode,
                    "camera_projection_mode": "calibrated_intrinsics",
                    "seg_degenerate_frames": 0,
                    "rgb_video_frames": [],
                    "hand_mask_video_frames": [],
                }

            line_handles: Dict[str, object] = {}
            try:
                for target in camera_targets:
                    line_handles[target.camera_name] = target.lines_path.open("w", encoding="utf-8")

                for frame_idx in range(frame_count):
                    applied_joints = sim.apply_joint_vector(joint_values_for_frame(frame_idx), HAND_JOINT_ORDER)
                    ee_pose_apply_info = apply_end_effector_pose_to_ee_link(
                        sim,
                        args.ee_link,
                        ee_pose_seq[frame_idx],
                        camera_targets[0].ee_offset,
                    )
                    sim.forward()

                    for target in camera_targets:
                        runtime = camera_runtime[target.camera_name]
                        chosen_extrinsic = np.asarray(runtime["chosen_extrinsic"], dtype=np.float64)
                        T_world_camera = compute_world_camera_transform(
                            mode=target.mode,
                            extrinsic=chosen_extrinsic,
                            sim=sim,
                            ee_link=resolve_camera_reference_link(target.camera_name, args.ee_link),
                            ee_offset=target.ee_offset,
                        )
                        camera_state = CameraState(
                            kind="calibrated",
                            calibration_camera_name=target.camera_name,
                            T_world_camera=T_world_camera,
                            K=target.calib.K,
                        )
                        if viewer is not None and target.camera_name == camera_targets[0].camera_name:
                            sim.sync_viewer(viewer, camera_state)

                        rgb, geom_ids = sim.render_rgb_and_seg(camera_state=camera_state)
                        hand_mask, link_idx_map, seg_degenerate = decode_hand_segmentation_from_geom_ids(
                            geom_ids,
                            sim.model,
                            hand_body_ids,
                        )
                        if seg_degenerate:
                            runtime["seg_degenerate_frames"] = int(runtime["seg_degenerate_frames"]) + 1

                        label_map = np.zeros((args.height, args.width), dtype=np.uint8)
                        fingertips_payload: Dict[str, Dict[str, object]] = {}
                        per_finger_masks: Dict[str, np.ndarray] = {}

                        for spec in specs:
                            T_world_link = sim.world_link_transform(spec.position_link)
                            tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz

                            cam_xyz, pix_xy = _project_tip_for_camera(
                                tip_world=tip_world,
                                T_world_camera=T_world_camera,
                                K=target.calib.K,
                            )

                            visible = bool(
                                cam_xyz[2] > 1e-9
                                and 0.0 <= pix_xy[0] < args.width
                                and 0.0 <= pix_xy[1] < args.height
                            )

                            chosen_mask = np.zeros_like(hand_mask, dtype=bool)
                            chosen_link_name: Optional[str] = None
                            for candidate_name in spec.mask_link_candidates:
                                candidate_body = sim.body_name_to_id.get(candidate_name)
                                if candidate_body is None:
                                    continue
                                candidate_mask = hand_mask & (link_idx_map == int(candidate_body))
                                if np.any(candidate_mask):
                                    chosen_mask = candidate_mask
                                    chosen_link_name = candidate_name
                                    break

                            per_finger_masks[spec.finger] = chosen_mask
                            label_id = FINGER_LABEL_ID[spec.finger]
                            label_map[chosen_mask] = label_id

                            fingertips_payload[spec.finger] = {
                                "world_xyz": tip_world.tolist(),
                                "camera_xyz": cam_xyz.tolist(),
                                "pixel_xy": pix_xy.tolist(),
                                "visible_in_frame": visible,
                                "position_link": spec.position_link,
                                "local_offset_xyz": spec.local_offset_xyz.tolist(),
                                "mask_link": chosen_link_name,
                                "mask_pixel_count": int(np.count_nonzero(chosen_mask)),
                            }

                        rgb_out = _resize_rgb_bilinear(
                            rgb,
                            width=target.output_video_width,
                            height=target.output_video_height,
                        )
                        runtime["rgb_video_frames"].append(rgb_out)

                        hand_mask_for_video = hand_mask
                        if hand_mask_for_video.shape != (target.output_video_height, target.output_video_width):
                            hand_mask_for_video = _resize_mask_nearest(
                                hand_mask_for_video,
                                width=target.output_video_width,
                                height=target.output_video_height,
                            )
                        hand_mask_rgb = np.zeros(
                            (target.output_video_height, target.output_video_width, 3),
                            dtype=np.uint8,
                        )
                        mask_color_rgb = np.asarray(args.hand_mask_color, dtype=np.uint8).reshape(1, 3)
                        hand_mask_rgb[hand_mask_for_video] = mask_color_rgb
                        runtime["hand_mask_video_frames"].append(hand_mask_rgb)

                        label_map_out = label_map
                        if label_map_out.shape != (target.output_video_height, target.output_video_width):
                            label_map_out = np.asarray(
                                Image.fromarray(label_map, mode="L").resize(
                                    (int(target.output_video_width), int(target.output_video_height)),
                                    resample=Image.NEAREST,
                                ),
                                dtype=np.uint8,
                            )
                        Image.fromarray(label_map_out, mode="L").save(
                            target.label_map_dir / f"{frame_idx:06d}.png"
                        )
                        if args.save_per_finger_masks:
                            for finger in FINGER_ORDER:
                                mask_out = per_finger_masks[finger]
                                if mask_out.shape != (target.output_video_height, target.output_video_width):
                                    mask_out = _resize_mask_nearest(
                                        mask_out,
                                        width=target.output_video_width,
                                        height=target.output_video_height,
                                    )
                                save_mask(
                                    target.per_finger_base_dir / finger / f"{frame_idx:06d}.png",
                                    mask_out,
                                )

                        line_obj = {
                            "frame_index": frame_idx,
                            "camera_name": target.camera_name,
                            "camera_mode": target.mode,
                            "hand_joint_source": joint_key,
                            "applied_hand_joints": applied_joints,
                            "end_effector_pose": ee_pose_apply_info,
                            "fingertips": fingertips_payload,
                            "hand_mask_pixel_count": int(np.count_nonzero(hand_mask)),
                        }
                        line_handles[target.camera_name].write(json.dumps(line_obj) + "\n")
            finally:
                for handle in line_handles.values():
                    handle.close()

            for target in camera_targets:
                runtime = camera_runtime[target.camera_name]
                rgb_video_frames = runtime["rgb_video_frames"]
                hand_mask_video_frames = runtime["hand_mask_video_frames"]
                if rgb_video_frames:
                    media.write_video(str(target.rgb_video_path), rgb_video_frames, fps=float(args.video_fps))
                if hand_mask_video_frames:
                    media.write_video(
                        str(target.hand_mask_video_path),
                        hand_mask_video_frames,
                        fps=float(args.video_fps),
                    )
                    if target.vis_seg_actions_video_path is not None:
                        media.write_video(
                            str(target.vis_seg_actions_video_path),
                            hand_mask_video_frames,
                            fps=float(args.video_fps),
                        )

                print(
                    f"[done][episode {ep_id}][{target.camera_name}] frames={frame_count} "
                    f"renderer=mujoco_renderer degenerate_seg={int(runtime['seg_degenerate_frames'])} "
                    f"out={target.episode_out} "
                    f"vis_seg_actions={target.vis_seg_actions_video_path}"
                )
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        sim.close()


def main() -> None:
    args = parse_args()
    if SciRotation is None:
        raise ImportError(
            "scipy is required for rotation conversions in this script. "
            f"Import error: {SCIPY_IMPORT_ERROR}"
        )
    if mujoco is None:
        raise ImportError(
            "mujoco is required for this script. "
            f"Import error: {MUJOCO_IMPORT_ERROR}"
        )
    if args.show_viewer and mujoco_viewer is None:
        raise ImportError(
            "mujoco.viewer is required for --show-viewer. "
            f"Import error: {MUJOCO_VIEWER_IMPORT_ERROR}"
        )
    run_dataset_mode(args)


if __name__ == "__main__":
    main()
