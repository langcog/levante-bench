#!/usr/bin/env python3
"""Local SSH controller for submitting and monitoring Marlowe Slurm jobs.

This intentionally drives Stanford's normal SSH login path to
login.marlowe.stanford.edu. It does not use tunnels or scheduler bypasses.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

LOGIN_HOST = "login.marlowe.stanford.edu"
BASE_DIR = Path(__file__).resolve().parent
STATE_DIR = BASE_DIR / "state"
STATE_FILE = STATE_DIR / "jobs.json"
JOBS_DIR = BASE_DIR / "jobs"
JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def load_state() -> dict[str, Any]:
    STATE_DIR.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ssh_run(sunet: str, remote_cmd: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["ssh", f"{sunet}@{LOGIN_HOST}", remote_cmd]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(message or f"SSH command failed with exit code {result.returncode}")
    return result


def read_spec(name: str) -> dict[str, Any]:
    path = JOBS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("name", "remote_workdir", "submit_command"):
        if not spec.get(key):
            raise ValueError(f"{path} is missing required key: {key}")
    return spec


def _submit_remote(sunet: str, spec: dict[str, Any]) -> str:
    remote_workdir = shlex.quote(str(spec["remote_workdir"]))
    remote = f"cd {remote_workdir} && {spec['submit_command']}"
    result = ssh_run(sunet, remote)
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    match = JOB_ID_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not parse Slurm job id from SSH output:\n{text}")
    return match.group(1)


def submit_job(sunet: str, name: str) -> str:
    spec = read_spec(name)
    job_id = _submit_remote(sunet, spec)
    state = load_state()
    previous = state.get(name, {})
    state[name] = {
        "job_id": job_id,
        "retries": int(previous.get("retries", 0)),
        "spec": spec,
        "last_submit_time": int(time.time()),
        "status": "SUBMITTED",
        "history": [*previous.get("history", []), {"event": "submit", "job_id": job_id, "time": int(time.time())}],
    }
    save_state(state)
    return job_id


def query_status(sunet: str, job_id: str) -> str:
    quoted_job_id = shlex.quote(str(job_id))
    cmd = (
        f"squeue -h -j {quoted_job_id} -o '%T' || true; "
        f"sacct -n -X -j {quoted_job_id} --format=State | head -n 1 || true"
    )
    result = ssh_run(sunet, cmd)
    lines = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else "UNKNOWN"


def status_all(sunet: str) -> dict[str, Any]:
    state = load_state()
    for rec in state.values():
        rec["status"] = query_status(sunet, rec["job_id"])
        rec["last_status_time"] = int(time.time())
    save_state(state)
    return state


def cancel_job(sunet: str, name: str) -> None:
    state = load_state()
    rec = state.get(name)
    if not rec:
        raise KeyError(name)
    ssh_run(sunet, f"scancel {shlex.quote(str(rec['job_id']))}")
    rec["status"] = "CANCELLED"
    rec.setdefault("history", []).append({"event": "cancel", "job_id": rec["job_id"], "time": int(time.time())})
    save_state(state)


def maybe_resubmit(sunet: str, name: str) -> str | None:
    state = load_state()
    rec = state.get(name)
    if not rec:
        raise KeyError(name)

    spec = rec["spec"]
    status = query_status(sunet, rec["job_id"])
    rec["status"] = status
    rec["last_status_time"] = int(time.time())

    retryable = set(spec.get("resubmit_on", []))
    retries = int(rec.get("retries", 0))
    max_retries = int(spec.get("max_retries", 0))
    if status in retryable and retries < max_retries:
        job_id = _submit_remote(sunet, spec)
        rec["job_id"] = job_id
        rec["retries"] = retries + 1
        rec["last_submit_time"] = int(time.time())
        rec["status"] = "RESUBMITTED"
        rec.setdefault("history", []).append(
            {"event": "resubmit", "job_id": job_id, "previous_status": status, "time": int(time.time())}
        )
        save_state(state)
        return job_id

    save_state(state)
    return None


def watch_loop(sunet: str, *, once: bool = False) -> None:
    if not load_state():
        print("No jobs tracked.")
        return

    while True:
        state = status_all(sunet)
        for name in list(state):
            maybe_resubmit(sunet, name)
        if once:
            break
        sleep_for = min((int(v.get("spec", {}).get("poll_seconds", 60)) for v in state.values()), default=60)
        time.sleep(sleep_for)


def list_jobs() -> dict[str, Any]:
    return load_state()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit, monitor, cancel, and resubmit Marlowe Slurm jobs over SSH.")
    parser.add_argument("--sunet", required=True, help="SUNet ID used for SSH login to Marlowe.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    submit_parser = sub.add_parser("submit", help="Submit jobs/<name>.yaml and track the Slurm job id.")
    submit_parser.add_argument("name")

    sub.add_parser("status", help="Refresh and print all tracked job statuses.")

    cancel_parser = sub.add_parser("cancel", help="Cancel a tracked job by local spec name.")
    cancel_parser.add_argument("name")

    resubmit_parser = sub.add_parser("resubmit", help="Resubmit a tracked job if its status is retryable.")
    resubmit_parser.add_argument("name")

    watch_parser = sub.add_parser("watch", help="Poll statuses and resubmit retryable jobs.")
    watch_parser.add_argument("--once", action="store_true", help="Run a single monitoring pass and exit.")

    sub.add_parser("list", help="Print local tracked job state without contacting Marlowe.")

    args = parser.parse_args(argv)

    if args.cmd == "submit":
        print(submit_job(args.sunet, args.name))
    elif args.cmd == "status":
        print(json.dumps(status_all(args.sunet), indent=2, sort_keys=True))
    elif args.cmd == "cancel":
        cancel_job(args.sunet, args.name)
    elif args.cmd == "resubmit":
        print(maybe_resubmit(args.sunet, args.name))
    elif args.cmd == "watch":
        watch_loop(args.sunet, once=args.once)
    elif args.cmd == "list":
        print(json.dumps(list_jobs(), indent=2, sort_keys=True))
    else:  # pragma: no cover
        parser.error(f"unknown command: {args.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
