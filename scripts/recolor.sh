SRC="datasets/converted_processed_OOD/2026-04-04T12-25-46/processed_OOD"
RED_EXCEPT_20="datasets/converted_processed_OOD/2026-04-04T12-25-46/processed_OOD_red_except_20"
FINAL_MIXED="datasets/converted_processed_OOD/2026-04-04T12-25-46/processed_OOD_red_except_20_black_20"
python3 scripts/recolor_segmentation_videos.py \
  --dataset-root "$SRC" \
  --output-dataset-root "$RED_EXCEPT_20" \
  --episode-ids 0-19 \
  --label-color 3:255,0,0 \
  --label-color 4:255,0,0 \
  --label-color 5:255,0,0
python3 scripts/recolor_segmentation_videos.py \
  --dataset-root "$RED_EXCEPT_20" \
  --output-dataset-root "$FINAL_MIXED" \
  --episode-ids 20 \
  --label-color 3:0,0,0 \
  --label-color 4:0,0,0 \
  --label-color 5:0,0,0
