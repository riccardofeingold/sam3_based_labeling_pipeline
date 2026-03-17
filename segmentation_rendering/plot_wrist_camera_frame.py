#!/usr/bin/env python3
"""
Plot wrist camera and hand-root offset coordinate frames in 3D.

The plot shows:
- hand root frame (root at origin)
- hand-root-offset frame (translated by ee_translation_offset)
- wrist camera frame relative to the offset frame
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CALIBRATION_DIR = Path("/data/sam3_based_labeling_pipeline/assets/calibration_params")
DEFAULT_OUTPUT_PATH = Path(
    "/data/sam3_based_labeling_pipeline/segmentation_rendering/output/wrist_camera_frame_plot.png"
)
DEFAULT_EE_TRANSLATION_OFFSET = np.array(
    [0.04403478458607229, -0.023583276493303752, 0.13], dtype=np.float64
    # [0.0, 0.07, 0.13]
)
# Fixed wrist-view correction: -90 deg about X, then -90 deg about Y.
WRIST_VIEW_ROT_XY_NEG90 = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot wrist camera frame relative to hand-root-offset frame."
    )
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--camera-name", type=str, default="oakd_wrist_view")
    parser.add_argument(
        "--ee-translation-offset",
        type=float,
        nargs=3,
        default=list(DEFAULT_EE_TRANSLATION_OFFSET),
        help="Offset location of the hand-root-offset frame in hand-root coordinates.",
    )
    parser.add_argument(
        "--extrinsic-convention",
        type=str,
        choices=["offset_to_camera", "camera_to_offset"],
        default="offset_to_camera",
        help=(
            "Interpretation of calibration extrinsic matrix. "
            "offset_to_camera matches final_extract_segmentation_masks.py chaining."
        ),
    )
    parser.add_argument(
        "--invert-extrinsic",
        action="store_true",
        help="Invert calibration extrinsic before applying convention logic.",
    )
    parser.add_argument(
        "--identity-extrinsic",
        action="store_true",
        help=(
            "Ignore calibration extrinsics and use identity for wrist camera relative to "
            "the hand-root-offset frame."
        ),
    )
    parser.add_argument("--axis-length", type=float, default=0.06)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_extrinsic_map(calibration_dir: Path) -> Dict[str, np.ndarray]:
    path = calibration_dir / "transformations.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration extrinsics file: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)

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

    out: Dict[str, np.ndarray] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name = str(item[0])
                mat = np.asarray(item[1], dtype=np.float64)
                if mat.shape == (4, 4):
                    if name == "oakd_wrist_view":
                        out[name] = apply_wrist_view_xy_neg90(mat)
                    else:
                        out[name] = mat
    if not out:
        raise RuntimeError(f"No valid 4x4 transforms loaded from {path}")
    return out


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    radius = 0.5 * max(x_range, y_range, z_range)

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def plot_frame(ax, T_world_frame: np.ndarray, axis_length: float, label: str) -> None:
    origin = T_world_frame[:3, 3]
    R = T_world_frame[:3, :3]
    axis_colors = ["r", "g", "b"]
    axis_names = ["x", "y", "z"]

    for i in range(3):
        direction = R[:, i] * float(axis_length)
        endpoint = origin + direction
        if direction[2] > 1e-9:
            vertical_hint = "up"
        elif direction[2] < -1e-9:
            vertical_hint = "down"
        else:
            vertical_hint = "flat"
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=axis_colors[i],
            linewidth=2.0,
            arrow_length_ratio=0.15,
        )
        ax.scatter([endpoint[0]], [endpoint[1]], [endpoint[2]], color=axis_colors[i], s=10)
        ax.text(
            endpoint[0],
            endpoint[1],
            endpoint[2],
            (
                f"{label}.{axis_names[i]} "
                f"[{endpoint[0]:.3f}, {endpoint[1]:.3f}, {endpoint[2]:.3f}] "
                f"({vertical_hint})"
            ),
            fontsize=6,
            color=axis_colors[i],
        )
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color="k", s=20)
    ax.text(origin[0], origin[1], origin[2], f" {label}", fontsize=9)


def plot_world_up_down_reference(ax, origin: np.ndarray, axis_length: float) -> None:
    up = np.array([0.0, 0.0, float(axis_length) * 1.4], dtype=np.float64)
    down = -up
    up_end = origin + up
    down_end = origin + down
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        up[0],
        up[1],
        up[2],
        color="m",
        linewidth=2.4,
        arrow_length_ratio=0.15,
    )
    ax.quiver(
        origin[0],
        origin[1],
        origin[2],
        down[0],
        down[1],
        down[2],
        color="c",
        linewidth=2.4,
        arrow_length_ratio=0.15,
    )
    ax.text(up_end[0], up_end[1], up_end[2], "WORLD +Z (UP)", color="m", fontsize=8)
    ax.text(down_end[0], down_end[1], down_end[2], "WORLD -Z (DOWN)", color="c", fontsize=8)


def main() -> None:
    args = parse_args()

    T_offset_camera = np.eye(4, dtype=np.float64)
    if not args.identity_extrinsic:
        extr_map = load_extrinsic_map(args.calibration_dir)
        if args.camera_name not in extr_map:
            raise KeyError(
                f'Camera "{args.camera_name}" not found in calibration extrinsics. '
                f"Available: {sorted(extr_map.keys())}"
            )
        T_offset_camera = np.asarray(extr_map[args.camera_name], dtype=np.float64)

    if args.invert_extrinsic:
        T_offset_camera = np.linalg.inv(T_offset_camera)

    T_world_root = np.eye(4, dtype=np.float64)
    T_world_offset = np.eye(4, dtype=np.float64)
    T_world_offset[:3, 3] = np.asarray(args.ee_translation_offset, dtype=np.float64).reshape(3)

    if args.extrinsic_convention == "offset_to_camera":
        # Calibration matrix maps offset-frame points into camera frame.
        # Camera pose in offset/world is therefore the inverse.
        T_world_camera = T_world_offset @ np.linalg.inv(T_offset_camera)
    else:
        # Calibration matrix is interpreted directly as camera pose in offset frame.
        T_world_camera = T_world_offset @ T_offset_camera

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    plot_frame(ax, T_world_root, axis_length=args.axis_length, label="hand_root")
    plot_frame(ax, T_world_offset, axis_length=args.axis_length, label="hand_root_offset")
    plot_frame(ax, T_world_camera, axis_length=args.axis_length, label="wrist_camera_view1")
    plot_world_up_down_reference(
        ax,
        origin=T_world_root[:3, 3],
        axis_length=args.axis_length,
    )

    offset_origin = T_world_offset[:3, 3]
    camera_origin = T_world_camera[:3, 3]
    ax.plot(
        [offset_origin[0], camera_origin[0]],
        [offset_origin[1], camera_origin[1]],
        [offset_origin[2], camera_origin[2]],
        linestyle="--",
        linewidth=1.5,
        color="gray",
        label="offset->camera origin",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Wrist Camera Frame Relative to Hand-Root-Offset Frame")
    ax.legend(loc="upper left")
    ax.grid(True)
    set_axes_equal(ax)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=180)
    print(f"[done] saved plot: {args.output_path}")
    print(f"[info] T_world_root:\n{T_world_root}")
    print(f"[info] T_world_offset:\n{T_world_offset}")
    print(f"[info] T_offset_camera_used:\n{T_offset_camera}")
    print(f"[info] T_world_camera:\n{T_world_camera}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
