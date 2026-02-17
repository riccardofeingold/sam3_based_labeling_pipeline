from typing import Iterable, Tuple

import mediapy
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor

# Load video frames (used for rendering and box normalization).
video_path = "/data/sam3_based_labeling_pipeline/assets/videos/two_orca_hands.MOV"
video_frames = mediapy.read_video(video_path)
output_video_path = (
    "/data/sam3_based_labeling_pipeline/assets/videos/two_orca_hands_mask.mp4"
)
mask_color_rgb = (0, 255, 0)  # Change this to any RGB tuple for a different mask color.
output_fps = 24
max_frame_num_to_track = 240

# Initial frame (frame 0) hand box in pixel XYXY format: (x1, y1, x2, y2).
# Tighten this box for better masks if needed.
initial_hand_box_xyxy = (171, 179, 694, 382)
text_prompt = "robot hand"


def _xyxy_pixels_to_xywh_norm(
    box_xyxy: Iterable[float], image_width: int, image_height: int
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid XYXY box with non-positive size: {box_xyxy}")
    return [x1 / image_width, y1 / image_height, (x2 - x1) / image_width, (y2 - y1) / image_height]


def _xywh_norm_to_xyxy_pixels(boxes_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes_xyxy = np.zeros_like(boxes_xywh, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] * width
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] * height
    boxes_xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2]) * width
    boxes_xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3]) * height
    return boxes_xyxy


predictor = build_sam3_video_predictor()
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

height, width = video_frames[0].shape[:2]
initial_box_xywh_norm = _xyxy_pixels_to_xywh_norm(
    initial_hand_box_xyxy, image_width=width, image_height=height
)

_ = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=text_prompt,
        bounding_boxes=[initial_box_xywh_norm],
        bounding_box_labels=[1],  # 1 = positive box prompt
    )
)

outputs_per_frame = {}
for response in predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
        start_frame_index=0,
        max_frame_num_to_track=max_frame_num_to_track,
    )
):
    out = response["outputs"]
    frame_idx = response["frame_index"]
    boxes_xywh = np.asarray(out["out_boxes_xywh"], dtype=np.float32)
    boxes_xyxy = _xywh_norm_to_xyxy_pixels(boxes_xywh, width=width, height=height)
    outputs_per_frame[frame_idx] = {
        "object_ids": np.asarray(out["out_obj_ids"]),
        "masks": np.asarray(out["out_binary_masks"]),
        "scores": np.asarray(out["out_probs"]),
        "boxes": boxes_xyxy,
    }

print(f"Processed {len(outputs_per_frame)} frames")

def _to_binary_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0.0


def _render_mask_frame(frame_shape, frame_outputs, mask_color):
    mask_frame = np.zeros(frame_shape, dtype=np.uint8)
    if frame_outputs is None or len(frame_outputs["object_ids"]) == 0:
        return mask_frame

    union_mask = np.zeros(frame_shape[:2], dtype=bool)
    for idx in range(len(frame_outputs["object_ids"])):
        union_mask |= _to_binary_mask(frame_outputs["masks"][idx])
    mask_frame[union_mask] = np.array(mask_color, dtype=np.uint8)
    return mask_frame

sorted_frame_indices = sorted(outputs_per_frame.keys())
mask_video_frames = []
for frame_idx in sorted_frame_indices:
    frame = np.asarray(video_frames[frame_idx], dtype=np.uint8)
    frame_outputs = outputs_per_frame[frame_idx]
    mask_video_frames.append(
        _render_mask_frame(frame.shape, frame_outputs, mask_color_rgb)
    )

mediapy.write_video(output_video_path, mask_video_frames, fps=output_fps)
print(f"Saved mask video to: {output_video_path}")

predictor.shutdown()

# Access results for a specific frame
if sorted_frame_indices:
    frame_0_outputs = outputs_per_frame[sorted_frame_indices[0]]
    print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
    print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
    print(f"Scores: {frame_0_outputs['scores'].tolist()}")
    print(
        f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}"
    )
    print(f"Masks shape: {frame_0_outputs['masks'].shape}")