#!/usr/bin/env python3
"""
Render OrcaHand segmentation masks and fingertip end-effector positions via PyBullet.

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
import pickle
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import pybullet as p
except Exception as exc:  # pragma: no cover - dependency guard
    p = None
    PYBULLET_IMPORT_ERROR = exc
else:
    PYBULLET_IMPORT_ERROR = None

try:
    import mediapy as media
except Exception as exc:  # pragma: no cover - dependency guard
    media = None
    MEDIAPY_IMPORT_ERROR = exc
else:
    MEDIAPY_IMPORT_ERROR = None


DEFAULT_URDF_PATH = Path(
    "/data/sam3_based_labeling_pipeline/assets/orcahand_v1b/urdf/orcahand_v1b.urdf"
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

# Matches current calibration scripts in this repository.
DEFAULT_EE_TRANSLATION_OFFSET_FOR_SIDE_VIEW = np.array([0.04403478458607229, -0.023583276493303752, 0.17973690165554085], dtype=np.float64)
DEFAULT_EE_TRANSLATION_OFFSET_FOR_WRIST_VIEW = np.array([0.0, 0.07, 0.13], dtype=np.float64)

# Fallback if no fingertip links and no reference URDF fingertip joints are found.
FALLBACK_TIP_SPECS = {
    "thumb": ("thumb_dp", np.array([0.0, 0.0, 0.03], dtype=np.float64)),
    "index": ("index_ip", np.array([-0.009, 0.0, 0.04], dtype=np.float64)),
    "middle": ("middle_ip", np.array([-0.009, 0.0, 0.04], dtype=np.float64)),
    "ring": ("ring_ip", np.array([-0.009, 0.0, 0.04], dtype=np.float64)),
    "pinky": ("pinky_ip", np.array([-0.009, 0.0, 0.04], dtype=np.float64)),
}
EXCLUDED_HAND_MASK_LINK_NAMES = {"tower"}

# Order from assets/orcahand_v1b/scheme_orcahand_v1b.yaml (gc_tendons).
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

# [wrist, thumb_mcp, thumb_abd, thumb_pip, thumb_dip, index_abd, index_mcp, index_pip, middle_abd, middle_mcp, middle_pip, ring_abd, ring_mcp, ring_pip, pinky_abd, pinky_mcp, pinky_pip]
# HAND_JOINT_ORDER = [
#     "joint_palm", 
#     "joint_iip_thumb", 
#     "joint_abd_thumb", 
#     "joint_pip_thumb", 
#     "joint_dip_thumb", 
#     "joint_abd_index", 
#     "joint_mcp_index",
#     "joint_pip_index", 
#     "joint_abd_middle", 
#     "joint_iip_middle", 
#     "joint_iip_middle", 
#     "joint_abd_ring", 
#     "joint_iip_ring", 
#     "joint_pip_ring", 
#     "joint_abd_pinky", 
#     "joint_iip_pinky", 
#     "joint_pip_pinky"]

DEFAULT_HAND_ROOT_EXTRA_ROT_DEG = np.array([90.0, 90.0, 0.0], dtype=np.float64)
# Fixed wrist-view correction: -90 deg about X, then -90 deg about Y.
WRIST_VIEW_ROT_XY_NEG90 = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


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


def _parse_vec3(text: str) -> np.ndarray:
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values but got: {text}")
    return np.asarray(vals, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OrcaHand URDF + calibration to render segmentation masks and fingertip positions."
    )
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
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
        choices=["calibrated", "generic"],
        default="calibrated",
        help="Use calibration-based camera or a generic debug camera.",
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
        default="palm",
        help="Link used as EE frame when camera-mode is ee_camera.",
    )
    parser.add_argument(
        "--ee-translation-offset",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Override EE translation offset. If omitted, uses "
            "DEFAULT_EE_TRANSLATION_OFFSET_FOR_WRIST_VIEW for wrist view and "
            "DEFAULT_EE_TRANSLATION_OFFSET_FOR_SIDE_VIEW for side view."
        ),
    )
    parser.add_argument(
        "--hand-root-extra-rot-deg",
        type=float,
        nargs=3,
        default=list(DEFAULT_HAND_ROOT_EXTRA_ROT_DEG),
        help="Extra local XYZ rotation (degrees) applied at hand root after EE orientation.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--near", type=float, default=0.01)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument(
        "--generic-target",
        type=float,
        nargs=3,
        default=[0.04, 0.0, 0.16],
        help="Target point for generic camera setup.",
    )
    parser.add_argument(
        "--generic-distance",
        type=float,
        default=0.45,
        help="Orbit distance for generic camera setup.",
    )
    parser.add_argument(
        "--generic-yaw",
        type=float,
        default=45.0,
        help="Yaw (deg) for generic camera setup.",
    )
    parser.add_argument(
        "--generic-pitch",
        type=float,
        default=-25.0,
        help="Pitch (deg) for generic camera setup.",
    )
    parser.add_argument(
        "--generic-roll",
        type=float,
        default=0.0,
        help="Roll (deg) for generic camera setup.",
    )
    parser.add_argument(
        "--generic-up-axis-index",
        type=int,
        choices=[1, 2],
        default=2,
        help="Up axis for generic camera setup (1: Y-up, 2: Z-up).",
    )
    parser.add_argument(
        "--generic-fov",
        type=float,
        default=60.0,
        help="Vertical FOV in degrees for generic camera setup.",
    )
    parser.add_argument(
        "--joint-positions",
        type=str,
        default="",
        help='Comma-separated overrides, e.g. "joint_palm=0.2,joint_pip_index=0.5".',
    )
    parser.add_argument(
        "--joint-positions-json",
        type=Path,
        default=None,
        help="Path to JSON dict {joint_name: value} for joint overrides.",
    )
    parser.add_argument(
        "--save-rgb",
        action="store_true",
        help="Also save rendered RGB image.",
    )
    parser.add_argument(
        "--no-hand-root-frame-overlay",
        action="store_true",
        help="Disable drawing the hand-root XYZ frame overlay in rendered RGB outputs.",
    )
    parser.add_argument(
        "--hand-root-frame-axis-length",
        type=float,
        default=0.05,
        help="Axis length (meters) used for hand-root frame RGB overlay.",
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
        "--optimize-alignment",
        action="store_true",
        help=(
            "Optimize hand-root offset/rotation against green hand masks in segmentation_videos "
            "before rendering each episode."
        ),
    )
    parser.add_argument(
        "--optimize-frames",
        type=int,
        default=24,
        help="Number of frames sampled per episode for optimization.",
    )
    parser.add_argument(
        "--optimize-iterations",
        type=int,
        default=40,
        help="Random-search iterations per episode for alignment optimization.",
    )
    parser.add_argument(
        "--optimize-offset-range",
        type=float,
        default=0.05,
        help="Uniform search range (meters) around ee-translation-offset.",
    )
    parser.add_argument(
        "--optimize-rot-range-deg",
        type=float,
        default=20.0,
        help="Uniform search range (degrees) around hand-root-extra-rot-deg.",
    )
    parser.add_argument(
        "--green-threshold",
        type=int,
        default=100,
        help="Minimum green channel value for extracting real hand mask from segmentation video.",
    )
    parser.add_argument(
        "--green-dominance",
        type=int,
        default=40,
        help="Required (G-R) and (G-B) margin for green-mask extraction.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["auto", "tiny", "hardware"],
        default="auto",
        help="PyBullet renderer backend. auto tries hardware first, then tiny.",
    )
    parser.add_argument(
        "--disable-auto-extrinsic-direction",
        action="store_true",
        help="Disable raw/inverse extrinsic direction scoring like final_extract_segmentation_masks.py.",
    )
    parser.add_argument(
        "--no-rewrite-mesh-paths",
        action="store_true",
        help="Disable rewriting URDF mesh filenames to absolute paths.",
    )
    return parser.parse_args()


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

    def apply_wrist_view_xy_neg90(extrinsic_in: np.ndarray) -> np.ndarray:
        mat = np.asarray(extrinsic_in, dtype=np.float64)
        if mat.shape != (4, 4):
            raise ValueError(f"Expected 4x4 extrinsic matrix, got {mat.shape}")
        R = mat[:3, :3]
        t = mat[:3, 3]
        R_new = WRIST_VIEW_ROT_XY_NEG90 @ R
        t_new = WRIST_VIEW_ROT_XY_NEG90 @ t

        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R_new
        out[:3, 3] = t_new
        return out

    extr_map: Dict[str, np.ndarray] = {}
    if isinstance(extr_data, list):
        for item in extr_data:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name = str(item[0])
                mat = np.asarray(item[1], dtype=np.float64)
                if mat.shape == (4, 4):
                    if name == "oakd_wrist_view":
                        extr_map[name] = apply_wrist_view_xy_neg90(mat)
                    else:
                        extr_map[name] = mat

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
    if args.joint_positions_json is not None:
        with args.joint_positions_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("--joint-positions-json must contain a JSON object.")
        for k, v in obj.items():
            overrides[str(k)] = float(v)
    return overrides


def matrix_from_pose(pos_xyz: Iterable[float], quat_xyzw: Iterable[float]) -> np.ndarray:
    pos = np.asarray(list(pos_xyz), dtype=np.float64).reshape(3)
    quat = np.asarray(list(quat_xyzw), dtype=np.float64).reshape(4)
    rot = np.array(p.getMatrixFromQuaternion(quat.tolist()), dtype=np.float64).reshape(3, 3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def world_link_transform(body_id: int, link_idx: int) -> np.ndarray:
    if link_idx < 0:
        base_pos, base_quat = p.getBasePositionAndOrientation(body_id)
        return matrix_from_pose(base_pos, base_quat)
    state = p.getLinkState(body_id, link_idx, computeForwardKinematics=True)
    world_pos = state[4]
    world_quat = state[5]
    return matrix_from_pose(world_pos, world_quat)


def parse_fingertip_joints_from_urdf(urdf_path: Path) -> Dict[str, Tuple[str, str, np.ndarray]]:
    if not urdf_path.exists():
        return {}
    root = ET.parse(urdf_path).getroot()
    out: Dict[str, Tuple[str, str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        parent = joint.find("parent")
        origin = joint.find("origin")
        if child is None or parent is None:
            continue
        child_link = child.attrib.get("link", "")
        parent_link = parent.attrib.get("link", "")
        if not child_link.endswith("_fingertip"):
            continue
        finger = child_link.split("_")[0].strip().lower()
        if finger not in FINGER_LABEL_ID:
            continue
        offset = np.zeros((3,), dtype=np.float64)
        if origin is not None and "xyz" in origin.attrib:
            offset = _parse_vec3(origin.attrib["xyz"])
        out[finger] = (parent_link, child_link, offset)
    return out


def build_fingertip_specs(
    link_name_to_idx: Dict[str, int],
    main_urdf_path: Path,
) -> List[FingertipSpec]:
    specs: List[FingertipSpec] = []

    from_main = parse_fingertip_joints_from_urdf(main_urdf_path)

    for finger in FINGER_ORDER:
        # Priority 1: explicit fingertip link exists in loaded URDF.
        tip_link_name = f"{finger}_fingertip"
        if tip_link_name in link_name_to_idx:
            parent_link = None
            if finger in from_main:
                parent_link = from_main[finger][0]
            candidates = [tip_link_name]
            if parent_link is not None:
                candidates.append(parent_link)
            specs.append(
                FingertipSpec(
                    finger=finger,
                    position_link=tip_link_name,
                    local_offset_xyz=np.zeros((3,), dtype=np.float64),
                    mask_link_candidates=[c for c in candidates if c in link_name_to_idx],
                )
            )
            continue

        # Priority 2: use fixed-joint fingertip offset from URDF definitions.
        if finger in from_main:
            parent_link, _child, offset = from_main[finger]
            if parent_link in link_name_to_idx:
                specs.append(
                    FingertipSpec(
                        finger=finger,
                        position_link=parent_link,
                        local_offset_xyz=offset,
                        mask_link_candidates=[parent_link],
                    )
                )
                continue

        # Priority 3: static fallback constants.
        fallback_link, fallback_offset = FALLBACK_TIP_SPECS[finger]
        if fallback_link not in link_name_to_idx:
            raise RuntimeError(
                f'Could not resolve fingertip spec for "{finger}" '
                f"(missing {tip_link_name} and fallback link {fallback_link})."
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
    if override_xyz is not None:
        return np.asarray(list(override_xyz), dtype=np.float64).reshape(3)
    if camera_name == "oakd_wrist_view":
        return DEFAULT_EE_TRANSLATION_OFFSET_FOR_WRIST_VIEW.copy()
    return DEFAULT_EE_TRANSLATION_OFFSET_FOR_SIDE_VIEW.copy()


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


def _matrix4_from_pybullet_list(values: List[float]) -> np.ndarray:
    # PyBullet returns OpenGL matrices flattened in column-major order.
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")


def world_to_pybullet_view_matrix(T_world_camera: np.ndarray) -> List[float]:
    """
    Convert OpenCV-style world->camera extrinsic convention to a PyBullet view matrix.

    OpenCV convention (used in final_extract_segmentation_masks.py):
      x: right, y: down, z: forward
    PyBullet computeViewMatrix expects:
      eye, target (forward), and up in world coordinates.
    """
    R_world_camera = T_world_camera[:3, :3]
    cam_pos_world = T_world_camera[:3, 3]
    forward_world = R_world_camera[:, 2]
    up_world = -R_world_camera[:, 1]
    target_world = cam_pos_world + forward_world
    return p.computeViewMatrix(
        cameraEyePosition=cam_pos_world.tolist(),
        cameraTargetPosition=target_world.tolist(),
        cameraUpVector=up_world.tolist(),
    )


def project_point_with_view_projection(
    point_world: np.ndarray,
    view_matrix: List[float],
    projection_matrix: List[float],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    view = _matrix4_from_pybullet_list(view_matrix)
    proj = _matrix4_from_pybullet_list(projection_matrix)
    world_h = np.r_[np.asarray(point_world, dtype=np.float64).reshape(3), 1.0]
    cam_gl = view @ world_h
    clip = proj @ cam_gl
    if abs(float(clip[3])) <= 1e-9:
        return np.array([cam_gl[0], -cam_gl[1], -cam_gl[2]], dtype=np.float64), np.array(
            [-1.0, -1.0], dtype=np.float64
        )
    ndc = clip[:3] / clip[3]
    u = (float(ndc[0]) * 0.5 + 0.5) * float(width)
    v = (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(height)
    # Convert OpenGL camera coordinates to OpenCV-like camera coordinates.
    cam_cv = np.array([cam_gl[0], -cam_gl[1], -cam_gl[2]], dtype=np.float64)
    return cam_cv, np.array([u, v], dtype=np.float64)


def _project_world_point_for_render(
    point_world: np.ndarray,
    camera_setup: str,
    T_world_camera: Optional[np.ndarray],
    K: Optional[np.ndarray],
    view_matrix: List[float],
    projection_matrix: List[float],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if camera_setup == "calibrated":
        if T_world_camera is None or K is None:
            raise ValueError("Calibrated projection requires T_world_camera and K.")
        return project_point_camera(point_world, T_world_camera, K)
    return project_point_with_view_projection(
        point_world=point_world,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        width=width,
        height=height,
    )


def overlay_hand_root_frame_on_rgb(
    rgb: np.ndarray,
    T_world_root: np.ndarray,
    camera_setup: str,
    T_world_camera: Optional[np.ndarray],
    K: Optional[np.ndarray],
    view_matrix: List[float],
    projection_matrix: List[float],
    axis_length_m: float,
) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr

    h, w = arr.shape[:2]
    origin = T_world_root[:3, 3]
    R = T_world_root[:3, :3]
    axis_len = float(max(1e-6, axis_length_m))
    axis_points = {
        "x": origin + R[:, 0] * axis_len,
        "y": origin + R[:, 1] * axis_len,
        "z": origin + R[:, 2] * axis_len,
    }

    origin_cam, origin_pix = _project_world_point_for_render(
        point_world=origin,
        camera_setup=camera_setup,
        T_world_camera=T_world_camera,
        K=K,
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        width=w,
        height=h,
    )
    if origin_cam[2] <= 1e-9:
        return arr

    img = Image.fromarray(arr[:, :, :3], mode="RGB")
    draw = ImageDraw.Draw(img)
    origin_xy = (float(origin_pix[0]), float(origin_pix[1]))
    draw.ellipse(
        [
            origin_xy[0] - 3.0,
            origin_xy[1] - 3.0,
            origin_xy[0] + 3.0,
            origin_xy[1] + 3.0,
        ],
        fill=(255, 255, 255),
    )

    axis_colors = {"x": (255, 32, 32), "y": (32, 220, 32), "z": (32, 96, 255)}
    for axis_name, axis_world in axis_points.items():
        axis_cam, axis_pix = _project_world_point_for_render(
            point_world=axis_world,
            camera_setup=camera_setup,
            T_world_camera=T_world_camera,
            K=K,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            width=w,
            height=h,
        )
        if axis_cam[2] <= 1e-9:
            continue
        endpoint_xy = (float(axis_pix[0]), float(axis_pix[1]))
        draw.line([origin_xy, endpoint_xy], fill=axis_colors[axis_name], width=3)
        draw.ellipse(
            [
                endpoint_xy[0] - 2.0,
                endpoint_xy[1] - 2.0,
                endpoint_xy[0] + 2.0,
                endpoint_xy[1] + 2.0,
            ],
            fill=axis_colors[axis_name],
        )
        draw.text((endpoint_xy[0] + 2.0, endpoint_xy[1] + 2.0), axis_name, fill=axis_colors[axis_name])
    return np.asarray(img, dtype=np.uint8)


def decode_hand_segmentation(
    segmentation: np.ndarray, body_id: int, excluded_link_indices: Optional[set[int]] = None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    seg = np.asarray(segmentation, dtype=np.int32)
    # In some renderer/backend combinations the segmentation buffer is all zeros.
    # That means "unsupported/degenerate" rather than "everything is body 0".
    if np.all(seg == 0):
        h, w = seg.shape[:2]
        return (
            np.zeros((h, w), dtype=bool),
            np.full((h, w), -1, dtype=np.int32),
            True,
        )
    valid = seg >= 0
    object_uid = seg & ((1 << 24) - 1)
    link_idx = (seg >> 24) - 1
    hand_pixels = valid & (object_uid == int(body_id))
    if excluded_link_indices:
        excluded = np.isin(link_idx, np.asarray(sorted(excluded_link_indices), dtype=np.int32))
        hand_pixels = hand_pixels & (~excluded)
    return hand_pixels, link_idx, False


def save_mask(path: Path, mask: np.ndarray) -> None:
    arr = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


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


def resolve_mesh_filename(raw_filename: str, urdf_path: Path) -> Optional[Path]:
    text = raw_filename.strip()
    if not text:
        return None
    if text.startswith("package://"):
        return None
    path = Path(text)
    if path.is_absolute():
        return path if path.exists() else None
    candidate = (urdf_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    return None


def rewrite_urdf_mesh_paths_to_absolute(urdf_path: Path) -> Tuple[Path, bool]:
    root = ET.parse(urdf_path).getroot()
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            continue
        resolved = resolve_mesh_filename(filename, urdf_path)
        if resolved is None:
            continue
        mesh.attrib["filename"] = str(resolved)
        changed = True

    if not changed:
        return urdf_path, False

    tmp_dir = Path(tempfile.mkdtemp(prefix="orcahand_render_urdf_"))
    tmp_path = tmp_dir / urdf_path.name
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=True)
    return tmp_path, True


def compute_world_camera_transform(
    mode: str,
    extrinsic: np.ndarray,
    body_id: int,
    link_name_to_idx: Dict[str, int],
    ee_link: str,
    ee_offset: np.ndarray,
) -> np.ndarray:
    if mode == "base_camera":
        return extrinsic

    if ee_link not in link_name_to_idx:
        raise KeyError(f'ee_camera mode requested but ee link "{ee_link}" was not found.')
    ee_link_idx = link_name_to_idx[ee_link]
    T_world_ee = world_link_transform(body_id, ee_link_idx)

    # Match final_extract_segmentation_masks.py chaining.
    T_world_prev = T_world_ee.copy()
    T_world_prev[:3, 3] = T_world_ee[:3, 3] - T_world_ee[:3, :3] @ ee_offset
    return T_world_prev @ extrinsic


def score_extrinsic_direction_for_pose(
    mode: str,
    extrinsic: np.ndarray,
    body_id: int,
    link_name_to_idx: Dict[str, int],
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
        body_id=body_id,
        link_name_to_idx=link_name_to_idx,
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
    path = (dataset_base_dir / rel_path).resolve()
    return path


def _frame_to_green_mask(
    frame_rgb: np.ndarray,
    green_threshold: int,
    green_dominance: int,
) -> np.ndarray:
    arr = np.asarray(frame_rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected RGB frame, got shape={arr.shape}")
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (g >= int(green_threshold)) & ((g - r) >= int(green_dominance)) & (
        (g - b) >= int(green_dominance)
    )


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


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def load_green_masks_for_indices(
    video_path: Path,
    frame_indices: List[int],
    width: Optional[int],
    height: Optional[int],
    green_threshold: int,
    green_dominance: int,
) -> Optional[Dict[int, np.ndarray]]:
    if not video_path.exists():
        return None
    if not frame_indices:
        return {}

    target = sorted(set(int(i) for i in frame_indices))
    frames_out: Dict[int, np.ndarray] = {}

    # Prefer OpenCV if available.
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            idx_set = set(target)
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                if frame_idx in idx_set:
                    frame_rgb = frame_bgr[:, :, ::-1]
                    mask = _frame_to_green_mask(
                        frame_rgb, green_threshold=green_threshold, green_dominance=green_dominance
                    )
                    if width is not None and height is not None:
                        mask = _resize_mask_nearest(mask, width=width, height=height)
                    frames_out[frame_idx] = np.asarray(mask, dtype=bool)
                    if len(frames_out) == len(target):
                        break
                frame_idx += 1
            cap.release()
            return frames_out

    # Fallback to imageio.
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception:
        return None
    idx_set = set(target)
    for frame_idx, frame in enumerate(iio.imiter(str(video_path))):
        if frame_idx not in idx_set:
            continue
        frame_rgb = np.asarray(frame, dtype=np.uint8)
        mask = _frame_to_green_mask(
            frame_rgb, green_threshold=green_threshold, green_dominance=green_dominance
        )
        if width is not None and height is not None:
            mask = _resize_mask_nearest(mask, width=width, height=height)
        frames_out[frame_idx] = np.asarray(mask, dtype=bool)
        if len(frames_out) == len(target):
            break
    return frames_out


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


def render_with_fallback(
    width: int,
    height: int,
    view_matrix: List[float],
    projection_matrix: List[float],
    renderer_candidates: List[Tuple[str, int]],
    preferred_renderer: Optional[Tuple[str, int]] = None,
) -> Tuple[Tuple, str, int]:
    local_renderer_candidates: List[Tuple[str, int]] = []
    if preferred_renderer is not None:
        local_renderer_candidates.append(preferred_renderer)
    for cand in renderer_candidates:
        if preferred_renderer is not None and cand[0] == preferred_renderer[0]:
            continue
        local_renderer_candidates.append(cand)
    if not local_renderer_candidates:
        local_renderer_candidates = renderer_candidates

    cam = None
    used_name: Optional[str] = None
    used_id: Optional[int] = None
    renderer_errors: List[str] = []
    for r_name, r_id in local_renderer_candidates:
        try:
            cam_try = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                renderer=r_id,
                flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            )
        except Exception as exc:
            renderer_errors.append(f"{r_name}: {exc}")
            continue
        seg_try = np.asarray(cam_try[4], dtype=np.int32).reshape(height, width)
        if np.all(seg_try == 0) and len(local_renderer_candidates) > 1:
            renderer_errors.append(f"{r_name}: all-zero segmentation buffer (degenerate)")
            continue
        cam = cam_try
        used_name = r_name
        used_id = r_id
        break
    if cam is None or used_name is None or used_id is None:
        raise RuntimeError(f"render failed: {'; '.join(renderer_errors)}")
    return cam, used_name, used_id


def apply_joint_vector(
    body_id: int,
    joint_name_to_id: Dict[str, int],
    joint_values: np.ndarray,
    joint_order: List[str],
) -> Dict[str, float]:
    vals = np.asarray(joint_values, dtype=np.float64).reshape(-1)
    if vals.size < len(joint_order):
        raise ValueError(
            f"Expected at least {len(joint_order)} hand joint values, got {vals.size}."
        )
    applied: Dict[str, float] = {}
    for i, joint_name in enumerate(joint_order):
        jid = joint_name_to_id.get(joint_name)
        if jid is None:
            continue
        info = p.getJointInfo(body_id, jid)
        lower = float(info[8])
        upper = float(info[9])
        target = float(vals[i])
        if lower <= upper:
            target = float(np.clip(target, lower, upper))
        p.resetJointState(body_id, jid, targetValue=target)
        applied[joint_name] = target
    return applied


def apply_end_effector_pose_to_hand_root(
    body_id: int,
    target_xyz_rpy: np.ndarray,
    ee_offset_xyz: np.ndarray,
    hand_root_extra_rot_xyz_rad: np.ndarray,
) -> Dict[str, object]:
    arr = np.asarray(target_xyz_rpy, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        raise ValueError(
            f"Expected end-effector pose with 6 values [x,y,z,roll,pitch,yaw], got shape {arr.shape}"
        )
    base_quat_raw = p.getQuaternionFromEuler([float(arr[3]), float(arr[4]), float(arr[5])])
    rot_xyz = np.asarray(hand_root_extra_rot_xyz_rad, dtype=np.float64).reshape(3)
    qx = p.getQuaternionFromEuler([float(rot_xyz[0]), 0.0, 0.0])
    qy = p.getQuaternionFromEuler([0.0, float(rot_xyz[1]), 0.0])
    qz = p.getQuaternionFromEuler([0.0, 0.0, float(rot_xyz[2])])
    _tmp_pos, quat_after_x = p.multiplyTransforms(
        [0.0, 0.0, 0.0], base_quat_raw, [0.0, 0.0, 0.0], qx
    )
    _tmp_pos, quat_after_xy = p.multiplyTransforms(
        [0.0, 0.0, 0.0], quat_after_x, [0.0, 0.0, 0.0], qy
    )
    _tmp_pos, base_quat = p.multiplyTransforms(
        [0.0, 0.0, 0.0], quat_after_xy, [0.0, 0.0, 0.0], qz
    )
    R_world_ee = np.array(p.getMatrixFromQuaternion(base_quat), dtype=np.float64).reshape(3, 3)
    ee_offset = np.asarray(ee_offset_xyz, dtype=np.float64).reshape(3)
    # Apply negative EE offset at the hand root in world frame.
    base_pos_np = np.asarray(arr[:3], dtype=np.float64) - (R_world_ee @ ee_offset)
    base_pos = base_pos_np.tolist()
    p.resetBasePositionAndOrientation(body_id, base_pos, base_quat)
    return {
        "target_ee_xyz_rpy": arr[:6].tolist(),
        "ee_offset_xyz": ee_offset.tolist(),
        "hand_root_extra_rot_xyz_rad": rot_xyz.tolist(),
        "applied_base_position": [float(v) for v in base_pos],
        "applied_base_quaternion_xyzw": [float(v) for v in base_quat],
    }


def sample_frame_indices(frame_count: int, sample_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    n = max(1, int(sample_count))
    if n >= frame_count:
        return list(range(frame_count))
    vals = np.linspace(0.0, float(frame_count - 1), num=n, dtype=np.float64)
    return sorted({int(round(v)) for v in vals})


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
    calib: Optional[Calibration] = None
    if args.camera_setup == "calibrated":
        calibration_map = load_calibration_map(args.calibration_dir)
        if camera_name not in calibration_map:
            raise KeyError(
                f'Camera "{camera_name}" not found in calibration. '
                f"Available: {sorted(calibration_map.keys())}"
            )
        calib = calibration_map[camera_name]

    renderer_candidates: List[Tuple[str, int]]
    if args.renderer == "auto":
        renderer_candidates = [
            ("hardware", p.ER_BULLET_HARDWARE_OPENGL),
            ("tiny", p.ER_TINY_RENDERER),
        ]
    elif args.renderer == "hardware":
        renderer_candidates = [("hardware", p.ER_BULLET_HARDWARE_OPENGL)]
    else:
        renderer_candidates = [("tiny", p.ER_TINY_RENDERER)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")

    rewritten_urdf_path = args.urdf_path
    rewritten_mesh_paths = False
    rewrite_tmp_dir: Path | None = None
    try:
        p.setGravity(0.0, 0.0, 0.0)
        p.setAdditionalSearchPath(str(args.urdf_path.parent))

        if not args.no_rewrite_mesh_paths:
            rewritten_urdf_path, rewritten_mesh_paths = rewrite_urdf_mesh_paths_to_absolute(
                args.urdf_path
            )
            if rewritten_mesh_paths:
                rewrite_tmp_dir = rewritten_urdf_path.parent

        body_id = p.loadURDF(
            str(rewritten_urdf_path),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=False,
        )

        joint_name_to_id: Dict[str, int] = {}
        link_name_to_idx: Dict[str, int] = {"base": -1}
        num_joints = p.getNumJoints(body_id)
        for jid in range(num_joints):
            info = p.getJointInfo(body_id, jid)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            joint_name_to_id[joint_name] = jid
            link_name_to_idx[link_name] = jid

        specs = build_fingertip_specs(
            link_name_to_idx=link_name_to_idx,
            main_urdf_path=args.urdf_path,
        )
        excluded_mask_link_indices = {
            int(link_name_to_idx[name])
            for name in EXCLUDED_HAND_MASK_LINK_NAMES
            if name in link_name_to_idx
        }

        mode = "generic_camera" if args.camera_setup == "generic" else resolve_camera_mode(
            camera_name, args.camera_mode
        )

        generic_view_matrix: Optional[List[float]] = None
        generic_projection_matrix: Optional[List[float]] = None
        if args.camera_setup == "generic":
            generic_view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[
                    float(args.generic_target[0]),
                    float(args.generic_target[1]),
                    float(args.generic_target[2]),
                ],
                distance=float(args.generic_distance),
                yaw=float(args.generic_yaw),
                pitch=float(args.generic_pitch),
                roll=float(args.generic_roll),
                upAxisIndex=int(args.generic_up_axis_index),
            )
            generic_projection_matrix = p.computeProjectionMatrixFOV(
                fov=float(args.generic_fov),
                aspect=float(args.width) / float(args.height),
                nearVal=float(args.near),
                farVal=float(args.far),
            )

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
                    print(
                        f"[warning][episode {ep_id}] Missing {joint_key}; using {fallback_key} instead."
                    )
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
            episode_root_rot_deg = (
                np.asarray(args.hand_root_extra_rot_deg, dtype=np.float64).reshape(3).copy()
            )
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

            # Reset state before each episode.
            p.resetBasePositionAndOrientation(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
            for joint_name in HAND_JOINT_ORDER:
                jid = joint_name_to_id.get(joint_name)
                if jid is None:
                    continue
                p.resetJointState(body_id, jid, targetValue=0.0)
            p.stepSimulation()

            chosen_extrinsic: Optional[np.ndarray] = None
            extrinsic_direction_mode = "not_applicable"
            camera_projection_mode = "generic_fov"
            optimization_info: Dict[str, object] = {
                "enabled": bool(args.optimize_alignment),
                "performed": False,
                "status": "disabled",
                "requested_frames": int(args.optimize_frames),
                "requested_iterations": int(args.optimize_iterations),
                "offset_range_m": float(args.optimize_offset_range),
                "rot_range_deg": float(args.optimize_rot_range_deg),
                "rotation_optimized": False,
                "green_threshold": int(args.green_threshold),
                "green_dominance": int(args.green_dominance),
                "initial_offset_xyz": episode_ee_offset.tolist(),
                "initial_hand_root_extra_rot_xyz_deg": episode_root_rot_deg.tolist(),
                "view_index": int(view_index),
                "segmentation_video_path": (
                    str(episode_seg_video_path) if episode_seg_video_path is not None else None
                ),
                "dataset_video_size": (
                    [int(dataset_video_size[0]), int(dataset_video_size[1])]
                    if dataset_video_size is not None
                    else None
                ),
            }
            preferred_renderer: Optional[Tuple[str, int]] = None

            # Choose extrinsic direction once per episode from the first frame pose.
            if args.camera_setup == "calibrated":
                assert calib is not None
                apply_joint_vector(body_id, joint_name_to_id, joint_values_for_frame(0), HAND_JOINT_ORDER)
                apply_end_effector_pose_to_hand_root(
                    body_id,
                    ee_pose_seq[0],
                    episode_ee_offset,
                    episode_root_rot_rad,
                )
                p.stepSimulation()

                extrinsic_raw = np.asarray(calib.extrinsic, dtype=np.float64)
                if args.invert_extrinsic:
                    chosen_extrinsic = np.linalg.inv(extrinsic_raw)
                    extrinsic_direction_mode = "forced_inverse"
                elif args.disable_auto_extrinsic_direction:
                    chosen_extrinsic = extrinsic_raw
                    extrinsic_direction_mode = "forced_raw"
                else:
                    sample_points_world: List[np.ndarray] = []
                    if args.ee_link in link_name_to_idx:
                        T_world_ee = world_link_transform(body_id, link_name_to_idx[args.ee_link])
                        sample_points_world.append(T_world_ee[:3, 3].copy())
                    for spec in specs:
                        link_idx = link_name_to_idx[spec.position_link]
                        T_world_link = world_link_transform(body_id, link_idx)
                        tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz
                        sample_points_world.append(tip_world)
                    pts = np.asarray(sample_points_world, dtype=np.float64)
                    raw_in, raw_vis, _ = score_extrinsic_direction_for_pose(
                        mode=mode,
                        extrinsic=extrinsic_raw,
                        body_id=body_id,
                        link_name_to_idx=link_name_to_idx,
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
                        body_id=body_id,
                        link_name_to_idx=link_name_to_idx,
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

                camera_projection_mode = "calibrated_intrinsics"

            if args.optimize_alignment:
                if args.camera_setup != "calibrated":
                    optimization_info["status"] = "skipped_non_calibrated_camera_setup"
                elif chosen_extrinsic is None or calib is None:
                    optimization_info["status"] = "skipped_missing_calibration_state"
                else:
                    if episode_seg_video_path is None:
                        optimization_info["status"] = "skipped_missing_segmentation_video_path"
                    else:
                        sample_idxs = sample_frame_indices(frame_count, args.optimize_frames)
                        target_masks = load_green_masks_for_indices(
                            video_path=episode_seg_video_path,
                            frame_indices=sample_idxs,
                            width=None,
                            height=None,
                            green_threshold=args.green_threshold,
                            green_dominance=args.green_dominance,
                        )
                        if target_masks is None:
                            optimization_info["status"] = "skipped_no_video_backend_or_unreadable_video"
                        else:
                            sample_idxs = [idx for idx in sample_idxs if idx in target_masks]
                            optimization_info["sample_frame_indices"] = sample_idxs
                            if not sample_idxs:
                                optimization_info["status"] = "skipped_no_target_masks_loaded"
                            else:
                                projection_matrix = intrinsics_to_opengl_projection(
                                    K=calib.K,
                                    width=args.width,
                                    height=args.height,
                                    near=args.near,
                                    far=args.far,
                                )

                                def evaluate_alignment(
                                    test_offset_xyz: np.ndarray,
                                ) -> float:
                                    nonlocal preferred_renderer
                                    test_offset = np.asarray(test_offset_xyz, dtype=np.float64).reshape(3)
                                    ious: List[float] = []
                                    for frame_idx in sample_idxs:
                                        apply_joint_vector(
                                            body_id,
                                            joint_name_to_id,
                                            joint_values_for_frame(frame_idx),
                                            HAND_JOINT_ORDER,
                                        )
                                        apply_end_effector_pose_to_hand_root(
                                            body_id,
                                            ee_pose_seq[frame_idx],
                                            test_offset,
                                            episode_root_rot_rad,
                                        )
                                        p.stepSimulation()
                                        T_world_camera = compute_world_camera_transform(
                                            mode=mode,
                                            extrinsic=chosen_extrinsic,
                                            body_id=body_id,
                                            link_name_to_idx=link_name_to_idx,
                                            ee_link=args.ee_link,
                                            ee_offset=test_offset,
                                        )
                                        view_matrix = world_to_pybullet_view_matrix(T_world_camera)
                                        cam, used_name, used_id = render_with_fallback(
                                            width=args.width,
                                            height=args.height,
                                            view_matrix=view_matrix,
                                            projection_matrix=projection_matrix,
                                            renderer_candidates=renderer_candidates,
                                            preferred_renderer=preferred_renderer,
                                        )
                                        preferred_renderer = (used_name, used_id)
                                        seg = np.asarray(cam[4], dtype=np.int32).reshape(
                                            args.height, args.width
                                        )
                                        hand_mask, _link_idx_map, _seg_degenerate = decode_hand_segmentation(
                                            seg, body_id, excluded_link_indices=excluded_mask_link_indices
                                        )
                                        target_mask = np.asarray(target_masks[frame_idx], dtype=bool)
                                        target_h, target_w = target_mask.shape[:2]
                                        hand_mask_resized = _resize_mask_nearest(
                                            hand_mask,
                                            width=int(target_w),
                                            height=int(target_h),
                                        )
                                        ious.append(_mask_iou(hand_mask_resized, target_mask))
                                    return float(np.mean(ious)) if ious else -1.0

                                best_offset = episode_ee_offset.copy()
                                best_score = evaluate_alignment(best_offset)
                                optimization_info["initial_mean_iou"] = float(best_score)

                                rng = np.random.default_rng(seed=int(ep_id) + 17)
                                for _ in range(max(0, int(args.optimize_iterations))):
                                    cand_offset = best_offset + rng.uniform(
                                        -float(args.optimize_offset_range),
                                        float(args.optimize_offset_range),
                                        size=3,
                                    )
                                    cand_score = evaluate_alignment(cand_offset)
                                    if cand_score > best_score:
                                        best_score = cand_score
                                        best_offset = cand_offset

                                episode_ee_offset = np.asarray(best_offset, dtype=np.float64).reshape(3)
                                optimization_info["performed"] = True
                                optimization_info["status"] = "optimized"
                                optimization_info["best_mean_iou"] = float(best_score)
                                optimization_info["optimized_offset_xyz"] = (
                                    episode_ee_offset.tolist()
                                )
                                optimization_info["optimized_hand_root_extra_rot_xyz_deg"] = (
                                    episode_root_rot_deg.tolist()
                                )
                                print(
                                    f"[opt][episode {ep_id}] best_iou={best_score:.4f} "
                                    f"offset={episode_ee_offset.tolist()} "
                                    f"root_rot_deg={episode_root_rot_deg.tolist()}"
                                )

            used_renderer_name: Optional[str] = None
            used_renderer_id: Optional[int] = None
            if preferred_renderer is not None:
                used_renderer_name = preferred_renderer[0]
                used_renderer_id = preferred_renderer[1]
            seg_degenerate_frames = 0
            lines_path = episode_out / "fingertips_and_pose.jsonl"

            with lines_path.open("w", encoding="utf-8") as lines_f:
                for frame_idx in range(frame_count):
                    applied_joints = apply_joint_vector(
                        body_id,
                        joint_name_to_id,
                        joint_values_for_frame(frame_idx),
                        HAND_JOINT_ORDER,
                    )
                    ee_pose_apply_info = apply_end_effector_pose_to_hand_root(
                        body_id,
                        ee_pose_seq[frame_idx],
                        episode_ee_offset,
                        episode_root_rot_rad,
                    )
                    p.stepSimulation()

                    if args.camera_setup == "calibrated":
                        assert calib is not None
                        assert chosen_extrinsic is not None
                        T_world_camera = compute_world_camera_transform(
                            mode=mode,
                            extrinsic=chosen_extrinsic,
                            body_id=body_id,
                            link_name_to_idx=link_name_to_idx,
                            ee_link=args.ee_link,
                            ee_offset=episode_ee_offset,
                        )
                        view_matrix = world_to_pybullet_view_matrix(T_world_camera)
                        projection_matrix = intrinsics_to_opengl_projection(
                            K=calib.K,
                            width=args.width,
                            height=args.height,
                            near=args.near,
                            far=args.far,
                        )
                    else:
                        assert generic_view_matrix is not None
                        assert generic_projection_matrix is not None
                        T_world_camera = None
                        view_matrix = generic_view_matrix
                        projection_matrix = generic_projection_matrix

                    cam, used_renderer_name, used_renderer_id = render_with_fallback(
                        width=args.width,
                        height=args.height,
                        view_matrix=view_matrix,
                        projection_matrix=projection_matrix,
                        renderer_candidates=renderer_candidates,
                        preferred_renderer=(
                            (used_renderer_name, used_renderer_id)
                            if used_renderer_name is not None and used_renderer_id is not None
                            else None
                        ),
                    )

                    rgba = np.asarray(cam[2], dtype=np.uint8).reshape(args.height, args.width, 4)
                    rgb = rgba[:, :, :3]
                    if not args.no_hand_root_frame_overlay:
                        T_world_root = world_link_transform(body_id, -1)
                        rgb = overlay_hand_root_frame_on_rgb(
                            rgb=rgb,
                            T_world_root=T_world_root,
                            camera_setup=args.camera_setup,
                            T_world_camera=T_world_camera,
                            K=(calib.K if calib is not None else None),
                            view_matrix=view_matrix,
                            projection_matrix=projection_matrix,
                            axis_length_m=float(args.hand_root_frame_axis_length),
                        )
                    seg = np.asarray(cam[4], dtype=np.int32).reshape(args.height, args.width)
                    hand_mask, link_idx_map, seg_degenerate = decode_hand_segmentation(
                        seg, body_id, excluded_link_indices=excluded_mask_link_indices
                    )
                    if seg_degenerate:
                        seg_degenerate_frames += 1

                    label_map = np.zeros((args.height, args.width), dtype=np.uint8)
                    fingertips_payload: Dict[str, Dict[str, object]] = {}
                    per_finger_masks: Dict[str, np.ndarray] = {}

                    for spec in specs:
                        link_idx = link_name_to_idx[spec.position_link]
                        T_world_link = world_link_transform(body_id, link_idx)
                        tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz

                        if args.camera_setup == "calibrated":
                            assert T_world_camera is not None
                            assert calib is not None
                            cam_xyz, pix_xy = project_point_camera(tip_world, T_world_camera, calib.K)
                        else:
                            cam_xyz, pix_xy = project_point_with_view_projection(
                                point_world=tip_world,
                                view_matrix=view_matrix,
                                projection_matrix=projection_matrix,
                                width=args.width,
                                height=args.height,
                            )
                        visible = bool(
                            cam_xyz[2] > 1e-9
                            and 0.0 <= pix_xy[0] < args.width
                            and 0.0 <= pix_xy[1] < args.height
                        )

                        chosen_mask = np.zeros_like(hand_mask, dtype=bool)
                        chosen_link_name: Optional[str] = None
                        for candidate_name in spec.mask_link_candidates:
                            candidate_idx = link_name_to_idx.get(candidate_name)
                            if candidate_idx is None:
                                continue
                            candidate_mask = hand_mask & (link_idx_map == candidate_idx)
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
                        np.asarray(rgb, dtype=np.uint8),
                        width=output_video_width,
                        height=output_video_height,
                    )
                    rgb_video_frames.append(rgb_out)
                    hand_mask_vis = np.where(hand_mask, 255, 0).astype(np.uint8)
                    if hand_mask_vis.shape != (output_video_height, output_video_width):
                        hand_mask_vis = np.where(
                            _resize_mask_nearest(
                                hand_mask,
                                width=output_video_width,
                                height=output_video_height,
                            ),
                            255,
                            0,
                        ).astype(np.uint8)
                    hand_mask_rgb = np.repeat(hand_mask_vis[:, :, None], 3, axis=2)
                    hand_mask_video_frames.append(hand_mask_rgb)
                    Image.fromarray(label_map, mode="L").save(label_map_dir / f"{frame_idx:06d}.png")
                    if args.save_per_finger_masks:
                        for finger in FINGER_ORDER:
                            save_mask(
                                per_finger_base_dir / finger / f"{frame_idx:06d}.png",
                                per_finger_masks[finger],
                            )

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
                media.write_video(
                    str(hand_mask_video_path),
                    hand_mask_video_frames,
                    fps=float(args.video_fps),
                )

            meta = {
                "episode_id": ep_id,
                "annotation_path": str(ann_path),
                "frame_count_rendered": frame_count,
                "camera_setup": args.camera_setup,
                "camera_name": camera_name,
                "camera_mode": mode,
                "extrinsic_direction": extrinsic_direction_mode,
                "camera_projection_mode": camera_projection_mode,
                "used_renderer": used_renderer_name,
                "segmentation_degenerate_frames": int(seg_degenerate_frames),
                "urdf_path": str(args.urdf_path),
                "loaded_urdf_path": str(rewritten_urdf_path),
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
                },
                "rewritten_mesh_paths": bool(rewritten_mesh_paths),
                "alignment_optimization": optimization_info,
                "first_frame_joint_override": "zero",
            }
            with (episode_out / "render_meta.json").open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(
                f"[done][episode {ep_id}] frames={frame_count} renderer={used_renderer_name} "
                f"degenerate_seg={seg_degenerate_frames} out={episode_out}"
            )
    finally:
        p.disconnect()
        if rewrite_tmp_dir is not None:
            for child in rewrite_tmp_dir.glob("*"):
                child.unlink(missing_ok=True)
            rewrite_tmp_dir.rmdir()


def main() -> None:
    args = parse_args()
    if p is None:
        raise ImportError(
            "pybullet is required for this script. "
            f"Import error: {PYBULLET_IMPORT_ERROR}"
        )
    if args.dataset_root is not None or args.annotation_dir is not None:
        run_dataset_mode(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {args.urdf_path}")

    camera_name = pick_camera_name(args)
    resolved_ee_offset_default = resolve_ee_translation_offset(
        camera_name=camera_name, override_xyz=args.ee_translation_offset
    )
    calib: Optional[Calibration] = None
    if args.camera_setup == "calibrated":
        calibration_map = load_calibration_map(args.calibration_dir)
        if camera_name not in calibration_map:
            raise KeyError(
                f'Camera "{camera_name}" not found in calibration. '
                f"Available: {sorted(calibration_map.keys())}"
            )
        calib = calibration_map[camera_name]

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet in DIRECT mode.")

    rewritten_urdf_path = args.urdf_path
    rewritten_mesh_paths = False
    rewrite_tmp_dir: Path | None = None
    try:
        p.setGravity(0.0, 0.0, -9.81)
        p.setAdditionalSearchPath(str(args.urdf_path.parent))

        if not args.no_rewrite_mesh_paths:
            rewritten_urdf_path, rewritten_mesh_paths = rewrite_urdf_mesh_paths_to_absolute(
                args.urdf_path
            )
            if rewritten_mesh_paths:
                rewrite_tmp_dir = rewritten_urdf_path.parent

        body_id = p.loadURDF(
            str(rewritten_urdf_path),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=True,
        )

        joint_name_to_id: Dict[str, int] = {}
        link_name_to_idx: Dict[str, int] = {}
        link_name_to_idx["base"] = -1

        num_joints = p.getNumJoints(body_id)
        for jid in range(num_joints):
            info = p.getJointInfo(body_id, jid)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            link_name = info[12].decode("utf-8")
            joint_name_to_id[joint_name] = jid
            link_name_to_idx[link_name] = jid

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                p.resetJointState(body_id, jid, targetValue=0.0)

        overrides = load_joint_overrides(args)
        for joint_name, target in overrides.items():
            if joint_name not in joint_name_to_id:
                raise KeyError(f'Unknown joint "{joint_name}" in overrides.')
            jid = joint_name_to_id[joint_name]
            info = p.getJointInfo(body_id, jid)
            lower = float(info[8])
            upper = float(info[9])
            clamped = float(target)
            if lower <= upper:
                clamped = float(np.clip(clamped, lower, upper))
            p.resetJointState(body_id, jid, targetValue=clamped)

        p.stepSimulation()

        specs = build_fingertip_specs(
            link_name_to_idx=link_name_to_idx,
            main_urdf_path=args.urdf_path,
        )
        excluded_mask_link_indices = {
            int(link_name_to_idx[name])
            for name in EXCLUDED_HAND_MASK_LINK_NAMES
            if name in link_name_to_idx
        }

        mode = "generic_camera" if args.camera_setup == "generic" else resolve_camera_mode(
            camera_name, args.camera_mode
        )
        T_world_camera: Optional[np.ndarray] = None
        extrinsic_direction_mode = "not_applicable"
        camera_projection_mode = "generic_fov"

        if args.camera_setup == "calibrated":
            assert calib is not None
            extrinsic_raw = np.asarray(calib.extrinsic, dtype=np.float64)
            ee_offset = np.asarray(resolved_ee_offset_default, dtype=np.float64).reshape(3)

            extrinsic_direction_mode = "forced_raw"
            if args.invert_extrinsic:
                extrinsic = np.linalg.inv(extrinsic_raw)
                T_world_camera = compute_world_camera_transform(
                    mode=mode,
                    extrinsic=extrinsic,
                    body_id=body_id,
                    link_name_to_idx=link_name_to_idx,
                    ee_link=args.ee_link,
                    ee_offset=ee_offset,
                )
                extrinsic_direction_mode = "forced_inverse"
            elif not args.disable_auto_extrinsic_direction:
                sample_points_world: List[np.ndarray] = []
                if args.ee_link in link_name_to_idx:
                    T_world_ee = world_link_transform(body_id, link_name_to_idx[args.ee_link])
                    sample_points_world.append(T_world_ee[:3, 3].copy())
                for spec in specs:
                    link_idx = link_name_to_idx[spec.position_link]
                    T_world_link = world_link_transform(body_id, link_idx)
                    tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz
                    sample_points_world.append(tip_world)
                points_world_arr = np.asarray(sample_points_world, dtype=np.float64)

                raw_in, raw_vis, T_world_camera_raw = score_extrinsic_direction_for_pose(
                    mode=mode,
                    extrinsic=extrinsic_raw,
                    body_id=body_id,
                    link_name_to_idx=link_name_to_idx,
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
                    body_id=body_id,
                    link_name_to_idx=link_name_to_idx,
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
                extrinsic = extrinsic_raw
                T_world_camera = compute_world_camera_transform(
                    mode=mode,
                    extrinsic=extrinsic,
                    body_id=body_id,
                    link_name_to_idx=link_name_to_idx,
                    ee_link=args.ee_link,
                    ee_offset=ee_offset,
                )

            assert T_world_camera is not None
            view_matrix = world_to_pybullet_view_matrix(T_world_camera)
            projection_matrix = intrinsics_to_opengl_projection(
                K=calib.K,
                width=args.width,
                height=args.height,
                near=args.near,
                far=args.far,
            )
            camera_projection_mode = "calibrated_intrinsics"
        else:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[
                    float(args.generic_target[0]),
                    float(args.generic_target[1]),
                    float(args.generic_target[2]),
                ],
                distance=float(args.generic_distance),
                yaw=float(args.generic_yaw),
                pitch=float(args.generic_pitch),
                roll=float(args.generic_roll),
                upAxisIndex=int(args.generic_up_axis_index),
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=float(args.generic_fov),
                aspect=float(args.width) / float(args.height),
                nearVal=float(args.near),
                farVal=float(args.far),
            )
            extrinsic_direction_mode = "generic_orbit_view"

        renderer_candidates: List[Tuple[str, int]]
        if args.renderer == "auto":
            renderer_candidates = [
                ("hardware", p.ER_BULLET_HARDWARE_OPENGL),
                ("tiny", p.ER_TINY_RENDERER),
            ]
        elif args.renderer == "hardware":
            renderer_candidates = [("hardware", p.ER_BULLET_HARDWARE_OPENGL)]
        else:
            renderer_candidates = [("tiny", p.ER_TINY_RENDERER)]

        cam = None
        used_renderer = None
        renderer_attempt_errors: List[str] = []
        for renderer_name, renderer_id in renderer_candidates:
            try:
                cam_try = p.getCameraImage(
                    width=args.width,
                    height=args.height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=renderer_id,
                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                )
            except Exception as exc:
                renderer_attempt_errors.append(f"{renderer_name}: {exc}")
                continue
            seg_try = np.asarray(cam_try[4], dtype=np.int32).reshape(args.height, args.width)
            if np.all(seg_try == 0) and len(renderer_candidates) > 1:
                renderer_attempt_errors.append(
                    f"{renderer_name}: all-zero segmentation buffer (degenerate)"
                )
                continue
            cam = cam_try
            used_renderer = renderer_name
            break

        if cam is None:
            joined = "; ".join(renderer_attempt_errors) if renderer_attempt_errors else "unknown"
            raise RuntimeError(f"All render attempts failed: {joined}")

        rgba = np.asarray(cam[2], dtype=np.uint8).reshape(args.height, args.width, 4)
        rgb = rgba[:, :, :3]
        if not args.no_hand_root_frame_overlay:
            T_world_root = world_link_transform(body_id, -1)
            rgb = overlay_hand_root_frame_on_rgb(
                rgb=rgb,
                T_world_root=T_world_root,
                camera_setup=args.camera_setup,
                T_world_camera=T_world_camera,
                K=(calib.K if calib is not None else None),
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
                axis_length_m=float(args.hand_root_frame_axis_length),
            )
        seg = np.asarray(cam[4], dtype=np.int32).reshape(args.height, args.width)
        hand_mask, link_idx_map, seg_degenerate = decode_hand_segmentation(
            seg, body_id, excluded_link_indices=excluded_mask_link_indices
        )
        if seg_degenerate:
            print(
                "[warning] Segmentation buffer is all zeros. "
                "This usually means renderer backend does not provide object/link ids."
            )

        label_map = np.zeros((args.height, args.width), dtype=np.uint8)
        fingertip_payload: Dict[str, Dict[str, object]] = {}

        for spec in specs:
            if spec.position_link not in link_name_to_idx:
                raise RuntimeError(
                    f'Fingertip position link "{spec.position_link}" for finger "{spec.finger}" was not found.'
                )

            link_idx = link_name_to_idx[spec.position_link]
            T_world_link = world_link_transform(body_id, link_idx)
            tip_world = T_world_link[:3, 3] + T_world_link[:3, :3] @ spec.local_offset_xyz

            if args.camera_setup == "calibrated":
                assert T_world_camera is not None
                assert calib is not None
                cam_xyz, pix_xy = project_point_camera(tip_world, T_world_camera, calib.K)
            else:
                cam_xyz, pix_xy = project_point_with_view_projection(
                    point_world=tip_world,
                    view_matrix=view_matrix,
                    projection_matrix=projection_matrix,
                    width=args.width,
                    height=args.height,
                )
            visible = bool(
                cam_xyz[2] > 1e-9
                and 0.0 <= pix_xy[0] < args.width
                and 0.0 <= pix_xy[1] < args.height
            )

            chosen_mask = np.zeros_like(hand_mask, dtype=bool)
            chosen_link_name: Optional[str] = None
            for candidate_name in spec.mask_link_candidates:
                candidate_idx = link_name_to_idx.get(candidate_name)
                if candidate_idx is None:
                    continue
                candidate_mask = hand_mask & (link_idx_map == candidate_idx)
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

        # Save global masks/maps.
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
            "used_renderer": used_renderer,
            "rewritten_mesh_paths": bool(rewritten_mesh_paths),
            "urdf_path": str(args.urdf_path),
            "loaded_urdf_path": str(rewritten_urdf_path),
            "calibration_dir": str(args.calibration_dir),
            "image_size": [args.width, args.height],
            "near_far": [args.near, args.far],
            "hand_root_frame_overlay": {
                "enabled": bool(not args.no_hand_root_frame_overlay),
                "axis_length_m": float(args.hand_root_frame_axis_length),
            },
            "joint_overrides": overrides,
            "hand_mask_pixel_count": int(np.count_nonzero(hand_mask)),
            "fingertips": fingertip_payload,
        }
        if args.camera_setup == "generic":
            output_json["generic_camera"] = {
                "target_xyz": [float(v) for v in args.generic_target],
                "distance": float(args.generic_distance),
                "yaw_deg": float(args.generic_yaw),
                "pitch_deg": float(args.generic_pitch),
                "roll_deg": float(args.generic_roll),
                "up_axis_index": int(args.generic_up_axis_index),
                "fov_deg": float(args.generic_fov),
            }
        with (args.output_dir / "fingertip_positions.json").open("w", encoding="utf-8") as f:
            json.dump(output_json, f, indent=2)

        print(f"[done] output_dir: {args.output_dir}")
        print(f"[done] camera: {camera_name} (mode={mode})")
        print(f"[done] extrinsic direction: {extrinsic_direction_mode}")
        print(f"[done] renderer: {used_renderer}")
        print(f"[done] saved fingertip positions: {args.output_dir / 'fingertip_positions.json'}")
        print(f"[done] saved hand mask: {args.output_dir / 'hand_mask.png'}")
        print(f"[done] saved label map: {args.output_dir / 'label_map.npy'}")
    finally:
        p.disconnect()
        if rewrite_tmp_dir is not None:
            for child in rewrite_tmp_dir.glob("*"):
                child.unlink(missing_ok=True)
            rewrite_tmp_dir.rmdir()


if __name__ == "__main__":
    main()
