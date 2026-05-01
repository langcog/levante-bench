# Marlowe Local Job Agent

Submit, monitor, cancel, and optionally resubmit Marlowe Slurm jobs from a
local machine by driving Stanford's normal SSH login flow to
`login.marlowe.stanford.edu`.

This tool intentionally uses ordinary SSH to the Marlowe login node and remote
Slurm commands (`sbatch`, `squeue`, `sacct`, `scancel`). It does not use
unsupported tunnels or scheduler bypasses.

## Setup

```bash
cd marlowe-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x marlowe_agent.py
```

## Job Specs

Create YAML files under `jobs/`. Example:

```yaml
name: vlm-eval-qwen
remote_workdir: ~/projects/vlm-bench
submit_command: sbatch run_eval.sbatch
resubmit_on:
  - FAILED
  - TIMEOUT
max_retries: 2
poll_seconds: 60
```

## Usage

Submit and track `jobs/example.yaml`:

```bash
python marlowe_agent.py --sunet YOUR_SUNET submit example
```

Refresh tracked statuses:

```bash
python marlowe_agent.py --sunet YOUR_SUNET status
```

Run one monitoring and resubmission pass:

```bash
python marlowe_agent.py --sunet YOUR_SUNET watch --once
```

Run continuously:

```bash
python marlowe_agent.py --sunet YOUR_SUNET watch
```

Cancel a tracked job:

```bash
python marlowe_agent.py --sunet YOUR_SUNET cancel example
```

Print local state without contacting Marlowe:

```bash
python marlowe_agent.py --sunet YOUR_SUNET list
```

Because Marlowe SSH access requires password and Duo, each SSH connection may
prompt for authentication depending on your SSH setup. Standard SSH
multiplexing can improve repeated commands while staying within the normal
login flow.
