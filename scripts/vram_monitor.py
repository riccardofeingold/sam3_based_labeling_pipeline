#!/usr/bin/env python3
"""
Monitor GPU VRAM on remote servers via SSH and send periodic Discord notifications.

Configuration:
  - Set DISCORD_WEBHOOK_URL in .env
  - Set per-server SSH passwords in .env as SSH_PASSWORD_<key> (e.g. SSH_PASSWORD_gpu1)
    If all servers share the same password, set SSH_PASSWORD as fallback.
  - Edit the SERVERS dict below to add your hosts.

Usage:
    python scripts/vram_monitor.py                  # loop every 60s
    python scripts/vram_monitor.py --interval 120   # loop every 2 min
    python scripts/vram_monitor.py --once           # single report and exit
    python scripts/vram_monitor.py --once &         # background single shot

Run continuously in the background:
    nohup python scripts/vram_monitor.py > vram_monitor.log 2>&1 &
"""

import json
import os
import time
import argparse
import urllib.error as urllib_error
import urllib.request as urllib_request
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Edit this dict to add / remove servers.
# Keys are display names used in the Discord message.
# ---------------------------------------------------------------------------
SERVERS: dict[str, dict] = {
    # "gpu1": {"host": "192.168.1.10", "user": "riccardo", "port": 22},
    # "gpu2": {"host": "192.168.1.11", "user": "riccardo", "port": 22},
}

NVIDIA_SMI_CMD = (
    "nvidia-smi "
    "--query-gpu=index,name,memory.total,memory.used,memory.free "
    "--format=csv,noheader,nounits"
)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _ssh_password(key: str) -> str | None:
    """Return the SSH password for *key*, falling back to SSH_PASSWORD."""
    return os.environ.get(f"SSH_PASSWORD_{key}") or os.environ.get("SSH_PASSWORD")


def _fetch_vram(name: str, cfg: dict) -> tuple[str, list[dict] | str]:
    """SSH into a server and return parsed VRAM info, or an error string."""
    password = _ssh_password(name)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=cfg["host"],
            port=cfg.get("port", 22),
            username=cfg["user"],
            password=password,
            timeout=15,
            banner_timeout=15,
            auth_timeout=15,
            look_for_keys=False,
            allow_agent=False,
        )
        _, stdout, stderr = client.exec_command(NVIDIA_SMI_CMD, timeout=15)
        out = stdout.read().decode()
        err = stderr.read().decode().strip()
        if err and not out.strip():
            return name, f"nvidia-smi error: {err}"
        return name, _parse_nvidia_smi(out)
    except Exception as exc:
        return name, f"SSH error: {exc}"
    finally:
        client.close()


def _parse_nvidia_smi(output: str) -> list[dict]:
    gpus = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, gpu_name, total_mib, used_mib, free_mib = parts
        try:
            total, used, free = int(total_mib), int(used_mib), int(free_mib)
        except ValueError:
            continue
        gpus.append(
            {
                "index": int(idx),
                "name": gpu_name,
                "total_gb": total / 1024,
                "used_gb": used / 1024,
                "free_gb": free / 1024,
                "used_pct": round(100 * used / total, 1) if total else 0.0,
            }
        )
    return gpus


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_server_block(server_name: str, result: list[dict] | str) -> str:
    if isinstance(result, str):
        return f"**{server_name}** — {result}"

    if not result:
        return f"**{server_name}** — no GPUs found"

    lines = [f"**{server_name}**"]
    for g in result:
        bar_filled = int(g["used_pct"] / 5)  # 20-char bar
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        lines.append(
            f"  GPU {g['index']} {g['name']}\n"
            f"  [{bar}] {g['used_pct']}%  "
            f"Free: **{g['free_gb']:.1f} GB** / {g['total_gb']:.1f} GB"
        )
    return "\n".join(lines)


def _build_message(results: dict[str, list[dict] | str], mention: str) -> str:
    prefix = f"{mention.strip()} " if mention.strip() else ""
    blocks = [_format_server_block(name, res) for name, res in results.items()]
    body = "\n\n".join(blocks)
    return f"{prefix}**VRAM Report**\n\n{body}"


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def _send_discord_message(webhook_url: str, content: str) -> None:
    # Discord messages have a 2000-char limit; truncate if needed.
    if len(content) > 2000:
        content = content[:1997] + "..."
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib_request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=10) as resp:
            status = getattr(resp, "status", 200)
            if status >= 400:
                raise RuntimeError(f"Discord webhook returned HTTP {status}")
    except (urllib_error.URLError, RuntimeError) as exc:
        print(f"[warning] Failed to send Discord notification: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def report(webhook_url: str, mention: str) -> None:
    if not SERVERS:
        print("[vram_monitor] SERVERS dict is empty — add entries at the top of the script.")
        return

    results: dict[str, list[dict] | str] = {}
    with ThreadPoolExecutor(max_workers=len(SERVERS)) as pool:
        futures = {pool.submit(_fetch_vram, name, cfg): name for name, cfg in SERVERS.items()}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result

    # Preserve SERVERS insertion order
    ordered = {name: results[name] for name in SERVERS if name in results}
    msg = _build_message(ordered, mention)
    print(msg)
    _send_discord_message(webhook_url, msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report remote GPU VRAM usage to Discord via SSH."
    )
    parser.add_argument(
        "--discord-webhook-url",
        type=str,
        default=os.environ.get("DISCORD_WEBHOOK_URL"),
        help="Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL env var.",
    )
    parser.add_argument(
        "--discord-mention",
        type=str,
        default="",
        help="Optional mention prefix, e.g. '@here' or '<@USER_ID>'.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between reports (default: 60).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send a single report and exit.",
    )
    args = parser.parse_args()

    if not args.discord_webhook_url:
        parser.error("Provide --discord-webhook-url or set DISCORD_WEBHOOK_URL in .env")

    if args.once:
        report(args.discord_webhook_url, args.discord_mention)
        return

    print(f"[vram_monitor] Polling {len(SERVERS)} server(s) every {args.interval}s. Ctrl+C to stop.")
    while True:
        report(args.discord_webhook_url, args.discord_mention)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
