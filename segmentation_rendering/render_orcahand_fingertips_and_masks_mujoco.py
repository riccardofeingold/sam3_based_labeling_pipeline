#!/usr/bin/env python3
"""
Render OrcaHand segmentation masks and fingertip end-effector positions via MuJoCo.

Outputs:
- rgb.png
- hand_mask.png
- per-finger binary masks
- label_map.npy and label_map.png
- fingertip_positions.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# Prefer headless EGL backend when DISPLAY is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")

try:
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None
    MUJOCO_IMPORT_ERROR = exc
else:
    MUJOCO_IMPORT_ERROR = None

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
    "/data/sam3_based_labeling_pipeline/assets/orcahand_v1b/orcahand.xml"
)
DEFAULT_CALIBRATION_DIR = Path("/data/sam3_based_labeling_pipeline/assets/calibration_params")
DEFAULT_OUTPUT_DIR = Path("/data/sam3_based_labeling_pipeline/segmentation_rendering/output")
DEFAULT_DATASET_ROOT = Path(
    "/data/Ctrl-World/datasets/2026-03-14T13-34-49/large_real_dataset_5fps_135_240"
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

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_LABEL_ID = {name: idx + 1 for idx, name in enumerate(FINGER_ORDER)}

DEFAULT_EE_TRANSLATION_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
CALIB_TO_SIM_AXIS_SWAP = np.array(
    [
        [0.0, 1.0, 0.0],  # x_calib = y_sim  -> x_sim = y_calib
        [1.0, 0.0, 0.0],  # y_calib = x_sim  -> y_sim = x_calib
        [0.0, 0.0, 1.0],  # z_calib = z_sim
    ],
    dtype=np.float64,
)

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
    T_world_camera: Optional[np.ndarray]
    K: Optional[np.ndarray]


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
        default="root",
        help="Link used as EE frame when camera-mode is ee_camera.",
    )
    parser.add_argument(
        "--ee-translation-offset",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Override EE translation offset. If omitted, uses DEFAULT_EE_TRANSLATION_OFFSET."
        ),
    )
    parser.add_argument(
        "--hand-root-extra-rot-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Extra local XYZ rotation (degrees) applied at hand root after EE orientation.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument(
        "--joint-positions",
        type=str,
        default="",
        help='Comma-separated overrides, e.g. "joint_palm=0.2,joint_pip_index=0.5".',
    )
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="Also save rendered RGB image.",
    )
    parser.add_argument(
        "--no-hand-root-frame-overlay",
        action="store_true",
        help="Disable MuJoCo-rendered body coordinate frames in RGB outputs.",
    )
    parser.add_argument(
        "--hand-root-frame-axis-length",
        type=float,
        default=0.05,
        help="Axis length (meters) for MuJoCo-rendered body coordinate frames.",
    )
    parser.add_argument(
        "--save-per-finger-masks",
        action="store_true",
        help="In dataset mode, save per-finger binary mask PNG per frame.",
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
        help="Disable raw/inverse extrinsic direction scoring like final_extract_segmentation_masks.py.",
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
    T_in = np.asarray(T_calib, dtype=np.float64).reshape(4, 4)
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = CALIB_TO_SIM_AXIS_SWAP
    return X @ T_in @ X.T


def convert_pose_xyz_rpy_calib_to_sim(pose_xyz_rpy: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose_xyz_rpy, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        raise ValueError(
            f"Expected end-effector pose with 6 values [x,y,z,roll,pitch,yaw], got shape {arr.shape}"
        )
    t_calib = arr[:3]
    q_calib = euler_xyz_to_quat_xyzw(float(arr[3]), float(arr[4]), float(arr[5]))
    R_calib = quat_xyzw_to_rot(q_calib)

    S = CALIB_TO_SIM_AXIS_SWAP
    t_sim = S @ t_calib
    R_sim = S @ R_calib @ S.T
    q_sim = rot_to_quat_xyzw(R_sim)
    T_sim = matrix_from_pose(t_sim, q_sim)

    out = arr.copy()
    out[:3] = T_sim[:3, 3]
    assert SciRotation is not None
    out[3:6] = SciRotation.from_matrix(T_sim[:3, :3]).as_euler("xyz", degrees=False).astype(
        np.float64
    )
    return out


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
            dist = np.zeros((5,), dtype=np.float64)
            out[cam_name] = Calibration(K=K, dist=dist, extrinsic=extr_map[cam_name])

    if not out:
        raise RuntimeError(f"No valid calibration loaded from {calibration_dir}")
    return out


def parse_joint_overrides(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = raw.strip()
    if not text:
        return out
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f'Invalid joint override "{token}". Use "joint_name=value".')
        name, value = token.split("=", 1)
        out[name.strip()] = float(value.strip())
    return out


def load_joint_overrides(args: argparse.Namespace) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    overrides.update(parse_joint_overrides(args.joint_positions))
    return overrides


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


def resolve_camera_mode(camera_name: str, mode_arg: str) -> str:
    if mode_arg != "auto":
        return mode_arg
    return CAMERA_EXTRINSIC_MODE.get(camera_name, "base_camera")


def resolve_ee_translation_offset(
    camera_name: str, override_xyz: Optional[Iterable[float]]
) -> np.ndarray:
    _ = camera_name
    if override_xyz is not None:
        return np.asarray(list(override_xyz), dtype=np.float64).reshape(3)
    return DEFAULT_EE_TRANSLATION_OFFSET.copy()


def intrinsics_to_opengl_projection(
    K: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
) -> List[float]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    proj = np.array(
        [
            [2.0 * fx / width, 0.0, (width - 2.0 * cx) / width, 0.0],
            [0.0, 2.0 * fy / height, (2.0 * cy - height) / height, 0.0],
            [0.0, 0.0, (near + far) / (near - far), 2.0 * near * far / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return proj.T.reshape(16).tolist()


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
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return (w, h)
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return None
    try:
        for frame in iio.imiter(str(video_path)):
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim >= 2:
                return (int(arr.shape[1]), int(arr.shape[0]))
            break
    except Exception:
        return None
    return None


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
            if not name:
                continue
            self.joint_name_to_id[name] = jid
            jtype = int(self.model.jnt_type[jid])
            qadr = int(self.model.jnt_qposadr[jid])
            if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
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

    def _set_camera_from_extrinsic(self, T_world_camera: np.ndarray) -> None:
        cam_pos = np.asarray(T_world_camera[:3, 3], dtype=np.float64)
        forward = np.asarray(T_world_camera[:3, 2], dtype=np.float64)
        dist = 0.35
        lookat = cam_pos + forward * dist
        offset = cam_pos - lookat
        xy = float(np.linalg.norm(offset[:2]))
        azimuth = float(np.degrees(np.arctan2(offset[1], offset[0])))
        elevation = float(np.degrees(np.arctan2(offset[2], max(1e-9, xy))))
        self._camera.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
        self._camera.fixedcamid = -1
        self._camera.trackbodyid = -1
        self._camera.lookat[:] = lookat.reshape(3)
        self._camera.distance = float(dist)
        self._camera.azimuth = float(azimuth)
        self._camera.elevation = float(elevation)

    def set_frame_visualization(self, enabled: bool, axis_length_m: float) -> None:
        if enabled:
            self._rgb_scene_option.frame = int(mujoco.mjtFrame.mjFRAME_BODY)
            axis_len = float(max(1e-4, axis_length_m))
            try:
                self.model.vis.scale.framelength = axis_len
            except Exception:
                pass
        else:
            self._rgb_scene_option.frame = int(mujoco.mjtFrame.mjFRAME_NONE)

    def render_rgb_and_seg(
        self,
        camera_state: CameraState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if camera_state.T_world_camera is None:
            raise ValueError("Calibrated rendering requires T_world_camera.")
        self._set_camera_from_extrinsic(camera_state.T_world_camera)

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


def apply_end_effector_pose_to_hand_root(
    sim: MujocoHandSim,
    target_xyz_rpy: np.ndarray,
    ee_offset_xyz: np.ndarray,
    hand_root_extra_rot_xyz_rad: np.ndarray,
) -> Dict[str, object]:
    arr_raw = np.asarray(target_xyz_rpy, dtype=np.float64).reshape(-1)
    arr = convert_pose_xyz_rpy_calib_to_sim(arr_raw)

    assert SciRotation is not None
    base_rot = SciRotation.from_euler("xyz", arr[3:6], degrees=False)
    rot_xyz = np.asarray(hand_root_extra_rot_xyz_rad, dtype=np.float64).reshape(3)
    extra_rot = SciRotation.from_euler("xyz", rot_xyz, degrees=False)
    R_world_ee = (base_rot * extra_rot).as_matrix().astype(np.float64)
    base_quat = SciRotation.from_matrix(R_world_ee).as_quat().astype(np.float64)

    ee_offset = np.asarray(ee_offset_xyz, dtype=np.float64).reshape(3)
    base_pos_np = np.asarray(arr[:3], dtype=np.float64) - (R_world_ee @ ee_offset)
    sim.set_base_pose(base_pos_np, base_quat)

    return {
        "target_ee_xyz_rpy_input_calib": arr_raw[:6].tolist(),
        "target_ee_xyz_rpy_used_sim": arr[:6].tolist(),
        "ee_offset_xyz": ee_offset.tolist(),
        "hand_root_extra_rot_xyz_rad": rot_xyz.tolist(),
        "applied_base_position": base_pos_np.tolist(),
        "applied_base_quaternion_xyzw": base_quat.tolist(),
    }


def _project_tip_for_camera(
    tip_world: np.ndarray,
    T_world_camera: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return project_point_camera(tip_world, T_world_camera, K)


def run_dataset_mode(args: argparse.Namespace) -> None:
    if args.dataset_root is None and args.annotation_dir is None:
        raise ValueError("Dataset mode requires --dataset-root or --annotation-dir.")
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

    camera_name = pick_camera_name(args)
    resolved_ee_offset_default = resolve_ee_translation_offset(
        camera_name=camera_name, override_xyz=args.ee_translation_offset
    )

    calibration_map = load_calibration_map(args.calibration_dir)
    if camera_name not in calibration_map:
        raise KeyError(
            f'Camera "{camera_name}" not found in calibration. '
            f"Available: {sorted(calibration_map.keys())}"
        )
    calib = calibration_map[camera_name]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sim = MujocoHandSim(args.model_path, width=args.width, height=args.height)
    try:
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

        mode = resolve_camera_mode(camera_name, args.camera_mode)

        for ep_id in selected_ids:
            ann_path = files_by_id[ep_id]
            with ann_path.open("r", encoding="utf-8") as f:
                ann = json.load(f)

            if args.camera_index is not None:
                view_index = int(args.camera_index)
            else:
                view_index = CAMERA_TO_VIDEO_INDEX.get(camera_name, -1)

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

            dataset_video_size = (
                probe_video_frame_size(episode_seg_video_path)
                if episode_seg_video_path is not None
                else None
            )
            output_video_width = int(dataset_video_size[0]) if dataset_video_size is not None else int(args.width)
            output_video_height = int(dataset_video_size[1]) if dataset_video_size is not None else int(args.height)

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

            episode_ee_offset = np.asarray(resolved_ee_offset_default, dtype=np.float64).reshape(3).copy()
            episode_root_rot_deg = np.asarray(args.hand_root_extra_rot_deg, dtype=np.float64).reshape(3).copy()
            episode_root_rot_rad = np.deg2rad(episode_root_rot_deg)

            episode_out = args.output_dir / str(ep_id)
            rgb_video_path = episode_out / "rgb.mp4"
            hand_mask_video_path = episode_out / "hand_mask.mp4"
            label_map_dir = episode_out / "label_map"
            per_finger_base_dir = episode_out / "finger_masks"
            episode_out.mkdir(parents=True, exist_ok=True)
            label_map_dir.mkdir(parents=True, exist_ok=True)
            if args.save_per_finger_masks:
                per_finger_base_dir.mkdir(parents=True, exist_ok=True)
                for finger in FINGER_ORDER:
                    (per_finger_base_dir / finger).mkdir(parents=True, exist_ok=True)

            rgb_video_frames: List[np.ndarray] = []
            hand_mask_video_frames: List[np.ndarray] = []

            sim.reset_pose()

            chosen_extrinsic: Optional[np.ndarray] = None
            extrinsic_direction_mode = "not_applicable"
            camera_projection_mode = "calibrated_intrinsics"

            sim.apply_joint_vector(joint_values_for_frame(0), HAND_JOINT_ORDER)
            apply_end_effector_pose_to_hand_root(
                sim,
                ee_pose_seq[0],
                episode_ee_offset,
                episode_root_rot_rad,
            )
            sim.forward()

            extrinsic_raw = np.asarray(calib.extrinsic, dtype=np.float64)
            if args.invert_extrinsic:
                chosen_extrinsic = np.linalg.inv(extrinsic_raw)
                extrinsic_direction_mode = "forced_inverse"
            elif args.disable_auto_extrinsic_direction:
                chosen_extrinsic = extrinsic_raw
                extrinsic_direction_mode = "forced_raw"
            else:
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
                raw_in, raw_vis, _ = score_extrinsic_direction_for_pose(
                    mode=mode,
                    extrinsic=extrinsic_raw,
                    sim=sim,
                    ee_link=args.ee_link,
                    ee_offset=episode_ee_offset,
                    points_world=pts,
                    K=calib.K,
                    width=args.width,
                    height=args.height,
                )
                inv_extrinsic = np.linalg.inv(extrinsic_raw)
                inv_in, inv_vis, _ = score_extrinsic_direction_for_pose(
                    mode=mode,
                    extrinsic=inv_extrinsic,
                    sim=sim,
                    ee_link=args.ee_link,
                    ee_offset=episode_ee_offset,
                    points_world=pts,
                    K=calib.K,
                    width=args.width,
                    height=args.height,
                )
                if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
                    chosen_extrinsic = extrinsic_raw
                    extrinsic_direction_mode = f"auto_raw(in_frame={raw_in},visible={raw_vis})"
                else:
                    chosen_extrinsic = inv_extrinsic
                    extrinsic_direction_mode = f"auto_inverse(in_frame={inv_in},visible={inv_vis})"

            seg_degenerate_frames = 0
            lines_path = episode_out / "fingertips_and_pose.jsonl"

            with lines_path.open("w", encoding="utf-8") as lines_f:
                for frame_idx in range(frame_count):
                    applied_joints = sim.apply_joint_vector(joint_values_for_frame(frame_idx), HAND_JOINT_ORDER)
                    ee_pose_apply_info = apply_end_effector_pose_to_hand_root(
                        sim,
                        ee_pose_seq[frame_idx],
                        episode_ee_offset,
                        episode_root_rot_rad,
                    )
                    sim.forward()

                    assert chosen_extrinsic is not None
                    T_world_camera = compute_world_camera_transform(
                        mode=mode,
                        extrinsic=chosen_extrinsic,
                        sim=sim,
                        ee_link=args.ee_link,
                        ee_offset=episode_ee_offset,
                    )
                    camera_state = CameraState(
                        kind="calibrated",
                        T_world_camera=T_world_camera,
                        K=calib.K,
                    )

                    rgb, geom_ids = sim.render_rgb_and_seg(
                        camera_state=camera_state,
                    )
                    hand_mask, link_idx_map, seg_degenerate = decode_hand_segmentation_from_geom_ids(
                        geom_ids,
                        sim.model,
                        hand_body_ids,
                    )
                    if seg_degenerate:
                        seg_degenerate_frames += 1

                    label_map = np.zeros((args.height, args.width), dtype=np.uint8)
                    fingertips_payload: Dict[str, Dict[str, object]] = {}
                    per_finger_masks: Dict[str, np.ndarray] = {}

                    for spec in specs:
                        T_world_link = sim.world_link_transform(spec.position_link)
                        tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz

                        cam_xyz, pix_xy = _project_tip_for_camera(
                            tip_world=tip_world,
                            T_world_camera=T_world_camera,
                            K=calib.K,
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

                    rgb_out = _resize_rgb_bilinear(rgb, width=output_video_width, height=output_video_height)
                    rgb_video_frames.append(rgb_out)

                    hand_mask_vis = np.where(hand_mask, 255, 0).astype(np.uint8)
                    if hand_mask_vis.shape != (output_video_height, output_video_width):
                        hand_mask_vis = np.where(
                            _resize_mask_nearest(hand_mask, width=output_video_width, height=output_video_height),
                            255,
                            0,
                        ).astype(np.uint8)
                    hand_mask_rgb = np.repeat(hand_mask_vis[:, :, None], 3, axis=2)
                    hand_mask_video_frames.append(hand_mask_rgb)

                    Image.fromarray(label_map, mode="L").save(label_map_dir / f"{frame_idx:06d}.png")
                    if args.save_per_finger_masks:
                        for finger in FINGER_ORDER:
                            save_mask(per_finger_base_dir / finger / f"{frame_idx:06d}.png", per_finger_masks[finger])

                    line_obj = {
                        "frame_index": frame_idx,
                        "hand_joint_source": joint_key,
                        "applied_hand_joints": applied_joints,
                        "end_effector_pose": ee_pose_apply_info,
                        "fingertips": fingertips_payload,
                        "hand_mask_pixel_count": int(np.count_nonzero(hand_mask)),
                    }
                    lines_f.write(json.dumps(line_obj) + "\n")

            if rgb_video_frames:
                media.write_video(str(rgb_video_path), rgb_video_frames, fps=float(args.video_fps))
            if hand_mask_video_frames:
                media.write_video(str(hand_mask_video_path), hand_mask_video_frames, fps=float(args.video_fps))

            meta = {
                "episode_id": ep_id,
                "annotation_path": str(ann_path),
                "frame_count_rendered": frame_count,
                "camera_setup": args.camera_setup,
                "camera_name": camera_name,
                "camera_mode": mode,
                "extrinsic_direction": extrinsic_direction_mode,
                "camera_projection_mode": camera_projection_mode,
                "used_renderer": "mujoco_renderer",
                "segmentation_degenerate_frames": int(seg_degenerate_frames),
                "model_path": str(args.model_path),
                "calibration_dir": str(args.calibration_dir),
                "output_dir": str(episode_out),
                "rgb_video_path": str(rgb_video_path),
                "hand_mask_video_path": str(hand_mask_video_path),
                "video_fps": float(args.video_fps),
                "output_video_size": [int(output_video_width), int(output_video_height)],
                "hand_joint_source_key": joint_key,
                "end_effector_pose_key": args.palm_pose_key,
                "end_effector_offset_applied_at_root_xyz": episode_ee_offset.tolist(),
                "hand_root_extra_rot_xyz_deg": episode_root_rot_deg.tolist(),
                "hand_root_extra_rot_xyz_rad": episode_root_rot_rad.tolist(),
                "hand_joint_order": HAND_JOINT_ORDER,
                "image_size": [args.width, args.height],
                "hand_root_frame_overlay": {
                    "enabled": bool(not args.no_hand_root_frame_overlay),
                    "axis_length_m": float(args.hand_root_frame_axis_length),
                    "source": "mujoco_scene",
                },
                "first_frame_joint_override": "zero",
                "simulator": "mujoco",
            }
            with (episode_out / "render_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(
                f"[done][episode {ep_id}] frames={frame_count} renderer=mujoco_renderer "
                f"degenerate_seg={seg_degenerate_frames} out={episode_out}"
            )
    finally:
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

    if args.dataset_root is not None or args.annotation_dir is not None:
        run_dataset_mode(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    camera_name = pick_camera_name(args)
    resolved_ee_offset_default = resolve_ee_translation_offset(
        camera_name=camera_name, override_xyz=args.ee_translation_offset
    )

    calibration_map = load_calibration_map(args.calibration_dir)
    if camera_name not in calibration_map:
        raise KeyError(
            f'Camera "{camera_name}" not found in calibration. '
            f"Available: {sorted(calibration_map.keys())}"
        )
    calib = calibration_map[camera_name]

    sim = MujocoHandSim(args.model_path, width=args.width, height=args.height)
    try:
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

        overrides = load_joint_overrides(args)
        sim.reset_pose()
        for joint_name, target in overrides.items():
            if joint_name not in sim.joint_name_to_id:
                raise KeyError(f'Unknown joint "{joint_name}" in overrides.')
            sim.set_joint_value(joint_name, target)
        sim.forward()

        mode = resolve_camera_mode(camera_name, args.camera_mode)
        T_world_camera: Optional[np.ndarray] = None
        extrinsic_direction_mode = "not_applicable"
        camera_projection_mode = "calibrated_intrinsics"

        extrinsic_raw = np.asarray(calib.extrinsic, dtype=np.float64)
        ee_offset = np.asarray(resolved_ee_offset_default, dtype=np.float64).reshape(3)

        if args.invert_extrinsic:
            extrinsic = np.linalg.inv(extrinsic_raw)
            T_world_camera = compute_world_camera_transform(
                mode=mode,
                extrinsic=extrinsic,
                sim=sim,
                ee_link=args.ee_link,
                ee_offset=ee_offset,
            )
            extrinsic_direction_mode = "forced_inverse"
        elif not args.disable_auto_extrinsic_direction:
            sample_points_world: List[np.ndarray] = []
            resolved_ee_link = sim.resolve_body_name(args.ee_link)
            if resolved_ee_link is not None:
                T_world_ee = sim.world_link_transform(resolved_ee_link)
                sample_points_world.append(T_world_ee[:3, 3].copy())
            for spec in specs:
                T_world_link = sim.world_link_transform(spec.position_link)
                tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz
                sample_points_world.append(tip_world)
            points_world_arr = np.asarray(sample_points_world, dtype=np.float64)

            raw_in, raw_vis, T_world_camera_raw = score_extrinsic_direction_for_pose(
                mode=mode,
                extrinsic=extrinsic_raw,
                sim=sim,
                ee_link=args.ee_link,
                ee_offset=ee_offset,
                points_world=points_world_arr,
                K=calib.K,
                width=args.width,
                height=args.height,
            )
            inv_extrinsic = np.linalg.inv(extrinsic_raw)
            inv_in, inv_vis, T_world_camera_inv = score_extrinsic_direction_for_pose(
                mode=mode,
                extrinsic=inv_extrinsic,
                sim=sim,
                ee_link=args.ee_link,
                ee_offset=ee_offset,
                points_world=points_world_arr,
                K=calib.K,
                width=args.width,
                height=args.height,
            )
            if raw_in > inv_in or (raw_in == inv_in and raw_vis >= inv_vis):
                T_world_camera = T_world_camera_raw
                extrinsic_direction_mode = f"auto_raw(in_frame={raw_in},visible={raw_vis})"
            else:
                T_world_camera = T_world_camera_inv
                extrinsic_direction_mode = f"auto_inverse(in_frame={inv_in},visible={inv_vis})"
        else:
            T_world_camera = compute_world_camera_transform(
                mode=mode,
                extrinsic=extrinsic_raw,
                sim=sim,
                ee_link=args.ee_link,
                ee_offset=ee_offset,
            )
            extrinsic_direction_mode = "forced_raw"

        camera_state = CameraState(
            kind="calibrated",
            T_world_camera=T_world_camera,
            K=calib.K,
        )
        rgb, geom_ids = sim.render_rgb_and_seg(
            camera_state=camera_state,
        )

        hand_mask, link_idx_map, seg_degenerate = decode_hand_segmentation_from_geom_ids(
            geom_ids,
            sim.model,
            hand_body_ids,
        )
        if seg_degenerate:
            print("[warning] Segmentation buffer appears degenerate (all background).")

        label_map = np.zeros((args.height, args.width), dtype=np.uint8)
        fingertip_payload: Dict[str, Dict[str, object]] = {}

        for spec in specs:
            T_world_link = sim.world_link_transform(spec.position_link)
            tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz

            cam_xyz, pix_xy = _project_tip_for_camera(
                tip_world=tip_world,
                T_world_camera=T_world_camera,
                K=calib.K,
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

            label_id = FINGER_LABEL_ID[spec.finger]
            label_map[chosen_mask] = label_id
            save_mask(args.output_dir / f"{spec.finger}_mask.png", chosen_mask)

            fingertip_payload[spec.finger] = {
                "world_xyz": tip_world.tolist(),
                "camera_xyz": cam_xyz.tolist(),
                "pixel_xy": pix_xy.tolist(),
                "visible_in_frame": visible,
                "position_link": spec.position_link,
                "local_offset_xyz": spec.local_offset_xyz.tolist(),
                "mask_link": chosen_link_name,
                "mask_pixel_count": int(np.count_nonzero(chosen_mask)),
            }

        save_mask(args.output_dir / "hand_mask.png", hand_mask)
        np.save(args.output_dir / "label_map.npy", label_map)
        Image.fromarray(label_map, mode="L").save(args.output_dir / "label_map.png")
        if args.save_rgb:
            Image.fromarray(rgb, mode="RGB").save(args.output_dir / "rgb.png")

        output_json = {
            "camera_setup": args.camera_setup,
            "camera_name": camera_name,
            "camera_mode": mode,
            "extrinsic_direction": extrinsic_direction_mode,
            "camera_projection_mode": camera_projection_mode,
            "used_renderer": "mujoco_renderer",
            "model_path": str(args.model_path),
            "calibration_dir": str(args.calibration_dir),
            "image_size": [args.width, args.height],
            "near_far": [args.near, args.far],
            "hand_root_frame_overlay": {
                "enabled": bool(not args.no_hand_root_frame_overlay),
                "axis_length_m": float(args.hand_root_frame_axis_length),
                "source": "mujoco_scene",
            },
            "joint_overrides": overrides,
            "hand_mask_pixel_count": int(np.count_nonzero(hand_mask)),
            "fingertips": fingertip_payload,
            "simulator": "mujoco",
        }
        with (args.output_dir / "fingertip_positions.json").open("w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2)

        print(f"[done] output_dir: {args.output_dir}")
        print(f"[done] camera: {camera_name} (mode={mode})")
        print(f"[done] extrinsic direction: {extrinsic_direction_mode}")
        print("[done] renderer: mujoco_renderer")
        print(f"[done] saved fingertip positions: {args.output_dir / 'fingertip_positions.json'}")
        print(f"[done] saved hand mask: {args.output_dir / 'hand_mask.png'}")
        print(f"[done] saved label map: {args.output_dir / 'label_map.npy'}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
