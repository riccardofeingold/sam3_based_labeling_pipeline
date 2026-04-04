#!/usr/bin/env python3
"""Run ordered GPU jobs from YAML with Discord notifications.

This scheduler:
- reads jobs from a YAML file
- runs lower `order` first
- for same `order`, runs smallest `expected_minutes` first
- waits for enough free VRAM + acceptable CPU usage
- auto-selects one or multiple GPUs and exports CUDA_VISIBLE_DEVICES
- sends Discord messages when each job starts and ends

Example config:

```yaml
settings:
  poll_interval_seconds: 10
  allowed_gpus: [0, 1]

jobs:
  - name: "preprocess"
    command: "python scripts/preprocess.py"
    order: 0
    expected_minutes: 15
    required_vram_gb: 3
    max_cpu_percent: 90

  - name: "main"
    command: "python scripts/train.py --gpus {gpus}"
    order: 0
    expected_minutes: 80
    required_vram_gb: 14
    parallel_on_multiple_gpus: true
    num_gpus: 2

  - name: "post"
    command: "python scripts/post.py"
    order: 1
    expected_minutes: 10
    required_vram_gb: 2
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error as urllib_error
import urllib.request as urllib_request
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from dotenv import load_dotenv

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required. Install it with `pip install pyyaml`."
    ) from exc


load_dotenv(override=True)

NVIDIA_SMI_CMD = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
]


@dataclass
class Job:
    name: str
    command: str
    order: int
    expected_minutes: float
    required_vram_gb: float
    max_cpu_percent: Optional[float] = None
    allowed_gpus: Optional[List[int]] = None
    parallel_on_multiple_gpus: bool = False
    num_gpus: int = 1
    cwd: Optional[str] = None


@dataclass
class GpuInfo:
    index: int
    total_gb: float
    used_gb: float
    free_gb: float


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _send_discord_message(webhook_url: Optional[str], mention: str, content: str) -> None:
    if not webhook_url:
        return

    prefix = f"{mention.strip()} " if mention and mention.strip() else ""
    text = f"{prefix}{content}"
    if len(text) > 2000:
        text = text[:1997] + "..."

    payload = json.dumps({"content": text}).encode("utf-8")
    req = urllib_request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "python-gpu-job-scheduler",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"Discord webhook returned HTTP {status}")
    except (urllib_error.URLError, RuntimeError) as exc:
        print(f"[warning] Failed to send Discord message: {exc}", file=sys.stderr)


def _to_int_list(values: Any, field: str) -> Optional[List[int]]:
    if values is None:
        return None
    if not isinstance(values, list):
        raise ValueError(f"`{field}` must be a list of integers.")
    out: List[int] = []
    for v in values:
        if not isinstance(v, int):
            raise ValueError(f"`{field}` entries must be integers.")
        out.append(v)
    return out


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping.")
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Config must contain `jobs` list.")
    return data


def _parse_jobs(raw_jobs: List[Any], global_allowed_gpus: Optional[List[int]]) -> List[Job]:
    jobs: List[Job] = []
    for i, item in enumerate(raw_jobs):
        if not isinstance(item, dict):
            raise ValueError(f"jobs[{i}] must be a mapping.")

        try:
            name = str(item["name"])
            command = str(item["command"])
        except KeyError as exc:
            raise ValueError(f"jobs[{i}] missing required field: {exc}") from exc

        order = int(item.get("order", 0))
        expected_minutes = float(item.get("expected_minutes", 0.0))
        required_vram_gb = float(item.get("required_vram_gb", 0.0))
        max_cpu_percent = (
            None if item.get("max_cpu_percent") is None else float(item["max_cpu_percent"])
        )
        allowed_gpus = _to_int_list(item.get("allowed_gpus"), f"jobs[{i}].allowed_gpus")
        if allowed_gpus is None:
            allowed_gpus = global_allowed_gpus
        parallel_on_multiple_gpus = bool(item.get("parallel_on_multiple_gpus", False))
        num_gpus = int(item.get("num_gpus", 1))
        if num_gpus < 1:
            raise ValueError(f"jobs[{i}].num_gpus must be >= 1.")
        if not parallel_on_multiple_gpus:
            num_gpus = 1
        cwd = None if item.get("cwd") is None else str(item["cwd"])

        jobs.append(
            Job(
                name=name,
                command=command,
                order=order,
                expected_minutes=expected_minutes,
                required_vram_gb=required_vram_gb,
                max_cpu_percent=max_cpu_percent,
                allowed_gpus=allowed_gpus,
                parallel_on_multiple_gpus=parallel_on_multiple_gpus,
                num_gpus=num_gpus,
                cwd=cwd,
            )
        )

    jobs.sort(key=lambda j: (j.order, j.expected_minutes, j.name))
    return jobs


def _query_gpus() -> List[GpuInfo]:
    try:
        result = subprocess.run(
            NVIDIA_SMI_CMD,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("`nvidia-smi` not found.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"`nvidia-smi` failed: {exc.stderr.strip()}") from exc

    gpus: List[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx_s, total_s, used_s, free_s = parts
        try:
            idx = int(idx_s)
            total_mib = int(total_s)
            used_mib = int(used_s)
            free_mib = int(free_s)
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=idx,
                total_gb=total_mib / 1024.0,
                used_gb=used_mib / 1024.0,
                free_gb=free_mib / 1024.0,
            )
        )
    return gpus


def _pick_gpus(job: Job, gpus: List[GpuInfo]) -> List[GpuInfo]:
    eligible: List[GpuInfo] = []
    for gpu in gpus:
        if job.allowed_gpus is not None and gpu.index not in job.allowed_gpus:
            continue
        if gpu.free_gb < job.required_vram_gb:
            continue
        eligible.append(gpu)
    needed = job.num_gpus if job.parallel_on_multiple_gpus else 1
    if len(eligible) < needed:
        return []
    eligible.sort(key=lambda g: g.free_gb, reverse=True)
    return eligible[:needed]


def _cpu_ok(max_cpu_percent: Optional[float]) -> bool:
    if max_cpu_percent is None:
        return True
    cpu = psutil.cpu_percent(interval=0.5)
    return cpu <= max_cpu_percent


def _wait_for_resources(job: Job, poll_interval: float) -> List[GpuInfo]:
    while True:
        gpus = _query_gpus()
        selected_gpus = _pick_gpus(job, gpus)
        cpu_ok = _cpu_ok(job.max_cpu_percent)
        if selected_gpus and cpu_ok:
            return selected_gpus

        cpu_need = "n/a" if job.max_cpu_percent is None else f"<= {job.max_cpu_percent:.1f}%"
        needed = job.num_gpus if job.parallel_on_multiple_gpus else 1
        print(
            f"[{_now_str()}] Waiting for resources for '{job.name}': "
            f"need {needed} GPU(s) with free VRAM >= {job.required_vram_gb:.1f} GB, CPU {cpu_need}"
        )
        time.sleep(max(1.0, poll_interval))


def _format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"


def _run_job(job: Job, gpus: List[GpuInfo]) -> int:
    gpu_ids = [gpu.index for gpu in gpus]
    cmd = job.command.format(gpu=gpu_ids[0], gpus=",".join(str(i) for i in gpu_ids))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)

    print(f"[{_now_str()}] Running '{job.name}' on GPU(s) {','.join(str(i) for i in gpu_ids)}")
    print(f"[{_now_str()}] Command: {cmd}")
    proc = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        env=env,
        cwd=job.cwd,
    )
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YAML-ordered jobs with automatic GPU assignment and Discord updates."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--discord-webhook-url",
        default=os.environ.get("DISCORD_WEBHOOK_URL"),
        help="Discord webhook URL (or set DISCORD_WEBHOOK_URL).",
    )
    parser.add_argument(
        "--discord-mention",
        default="",
        help="Optional Discord mention, e.g. '@here' or '<@USER_ID>'.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=None,
        help="Override polling interval in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan and notifications, but skip command execution.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    settings = cfg.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("`settings` must be a mapping.")

    global_allowed_gpus = _to_int_list(settings.get("allowed_gpus"), "settings.allowed_gpus")
    poll_interval = float(settings.get("poll_interval_seconds", 10.0))
    if args.poll_interval is not None:
        poll_interval = float(args.poll_interval)

    jobs = _parse_jobs(cfg["jobs"], global_allowed_gpus)
    if not jobs:
        print("No jobs found in config.")
        return

    print("Execution plan:")
    for i, job in enumerate(jobs, start=1):
        needed = job.num_gpus if job.parallel_on_multiple_gpus else 1
        print(
            f"  {i}. order={job.order} expected_minutes={job.expected_minutes:g} "
            f"required_vram_gb={job.required_vram_gb:g} gpus={needed} name={job.name}"
        )

    failures: List[str] = []

    for idx, job in enumerate(jobs, start=1):
        selected_gpus = _wait_for_resources(job, poll_interval)
        gpu_ids = [gpu.index for gpu in selected_gpus]
        gpu_free_text = ", ".join(f"GPU{gpu.index}:{gpu.free_gb:.1f}GB" for gpu in selected_gpus)

        start_msg = (
            f"[{_now_str()}] STARTED ({idx}/{len(jobs)}): {job.name}\n"
            f"order={job.order}, expected_minutes={job.expected_minutes:g}\n"
            f"GPUs={','.join(str(i) for i in gpu_ids)} ({gpu_free_text})"
        )
        print(start_msg)
        _send_discord_message(args.discord_webhook_url, args.discord_mention, start_msg)

        if args.dry_run:
            end_msg = f"[{_now_str()}] ENDED ({idx}/{len(jobs)}): {job.name} | DRY RUN"
            print(end_msg)
            _send_discord_message(args.discord_webhook_url, args.discord_mention, end_msg)
            continue

        started = time.time()
        code = _run_job(job, selected_gpus)
        duration = _format_duration(time.time() - started)

        if code == 0:
            end_msg = (
                f"[{_now_str()}] ENDED ({idx}/{len(jobs)}): {job.name} "
                f"| SUCCESS | duration={duration}"
            )
            print(end_msg)
            _send_discord_message(args.discord_webhook_url, args.discord_mention, end_msg)
        else:
            end_msg = (
                f"[{_now_str()}] ENDED ({idx}/{len(jobs)}): {job.name} "
                f"| FAILED code={code} | duration={duration}"
            )
            print(end_msg, file=sys.stderr)
            _send_discord_message(args.discord_webhook_url, args.discord_mention, end_msg)
            failures.append(job.name)

    if failures:
        summary = f"[{_now_str()}] Scheduler finished with failures: {', '.join(failures)}"
        print(summary, file=sys.stderr)
        _send_discord_message(args.discord_webhook_url, args.discord_mention, summary)
        raise SystemExit(1)

    summary = f"[{_now_str()}] Scheduler finished successfully. Ran {len(jobs)} job(s)."
    print(summary)
    _send_discord_message(args.discord_webhook_url, args.discord_mention, summary)


if __name__ == "__main__":
    main()
