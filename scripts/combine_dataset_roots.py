#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple dataset roots into one dataset root by remapping episodes "
            "to strictly increasing ids."
        )
    )
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Input dataset roots. Each root must contain annotation/ and videos/. "
            "Roots are processed in the order provided."
        ),
    )
    parser.add_argument(
        "--output-dataset-root",
        type=Path,
        required=True,
        help="Output combined dataset root. Must not already exist.",
    )
    parser.add_argument(
        "--start-episode-id",
        type=int,
        default=0,
        help="First episode id to use in the combined dataset (default: 0).",
    )
    parser.add_argument(
        "--mapping-filename",
        type=str,
        default="episode_id_mapping.json",
        help="Filename for the id mapping artifact saved under output root.",
    )
    return parser.parse_args()


def discover_numeric_episode_ids(annotation_dir: Path) -> List[int]:
    annotation_files = sorted(annotation_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"No annotation files found in {annotation_dir}")

    episode_ids: List[int] = []
    for annotation_path in annotation_files:
        stem = annotation_path.stem
        try:
            episode_ids.append(int(stem))
        except ValueError:
            print(
                f"[warning] Non-numeric annotation file ignored: {annotation_path.name}"
            )

    if not episode_ids:
        raise RuntimeError(f"No numeric episode ids discovered in {annotation_dir}")
    return sorted(set(episode_ids))


def _load_annotation_json(annotation_path: Path) -> Dict[str, object]:
    with annotation_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_annotation_json(annotation_path: Path, payload: Dict[str, object]) -> None:
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def validate_input_root(dataset_root: Path) -> None:
    annotation_dir = dataset_root / "annotation"
    videos_dir = dataset_root / "videos"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")


def main() -> None:
    args = parse_args()

    dataset_roots: List[Path] = [root.resolve() for root in args.dataset_roots]
    output_root = args.output_dataset_root.resolve()
    start_episode_id = int(args.start_episode_id)
    mapping_filename = args.mapping_filename.strip()

    if start_episode_id < 0:
        raise ValueError(f"--start-episode-id must be >= 0, got {start_episode_id}")
    if not mapping_filename:
        raise ValueError("--mapping-filename must be a non-empty string")
    if len(dataset_roots) < 2:
        raise ValueError("Please provide at least two --dataset-roots to combine.")
    if len(set(dataset_roots)) != len(dataset_roots):
        raise ValueError("Duplicate entries found in --dataset-roots.")
    if output_root.exists():
        raise FileExistsError(
            f"Output dataset root already exists: {output_root}. "
            "Please provide a new path."
        )

    for root in dataset_roots:
        validate_input_root(root)

    output_annotation_dir = output_root / "annotation"
    output_videos_dir = output_root / "videos"
    output_annotation_dir.mkdir(parents=True, exist_ok=False)
    output_videos_dir.mkdir(parents=True, exist_ok=False)

    next_episode_id = start_episode_id
    mapping_rows: List[Dict[str, object]] = []
    copied_episodes = 0

    print(
        f"[start] roots={len(dataset_roots)} output={output_root} "
        f"start_episode_id={start_episode_id}"
    )

    for root_index, input_root in enumerate(dataset_roots):
        input_annotation_dir = input_root / "annotation"
        input_videos_dir = input_root / "videos"
        source_episode_ids = discover_numeric_episode_ids(input_annotation_dir)
        print(
            f"[root {root_index + 1}/{len(dataset_roots)}] "
            f"path={input_root} episodes={len(source_episode_ids)}"
        )

        for source_episode_id in source_episode_ids:
            src_annotation_path = input_annotation_dir / f"{source_episode_id}.json"
            src_video_dir = input_videos_dir / str(source_episode_id)
            if not src_annotation_path.exists():
                raise FileNotFoundError(f"Missing annotation file: {src_annotation_path}")
            if not src_video_dir.exists():
                raise FileNotFoundError(
                    f"Missing episode video directory: {src_video_dir}"
                )

            target_episode_id = next_episode_id
            target_annotation_path = output_annotation_dir / f"{target_episode_id}.json"
            target_video_dir = output_videos_dir / str(target_episode_id)

            annotation_payload = _load_annotation_json(src_annotation_path)
            annotation_payload["episode_id"] = int(target_episode_id)
            _write_annotation_json(target_annotation_path, annotation_payload)

            shutil.copytree(src_video_dir, target_video_dir, dirs_exist_ok=False)

            mapping_rows.append(
                {
                    "source_root": str(input_root),
                    "source_episode_id": int(source_episode_id),
                    "target_episode_id": int(target_episode_id),
                }
            )

            copied_episodes += 1
            next_episode_id += 1
            print(
                f"[copied] src_root={input_root.name} src_episode={source_episode_id} "
                f"-> dst_episode={target_episode_id}"
            )

    mapping_path = output_root / mapping_filename
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping_rows, f, indent=2, ensure_ascii=True)
        f.write("\n")

    print(
        f"[done] copied_episodes={copied_episodes} "
        f"episode_id_range=[{start_episode_id}, {next_episode_id - 1}]"
    )
    print(f"[done] mapping={mapping_path}")


if __name__ == "__main__":
    main()
