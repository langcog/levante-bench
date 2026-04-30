#!/bin/bash
# Show recent Marlowe jobs with descriptive names and scheduling/failure reasons.

set -euo pipefail

START_TIME="${START_TIME:-today}"
JOB_NAME_WIDTH="${JOB_NAME_WIDTH:-80}"
REASON_WIDTH="${REASON_WIDTH:-32}"

cmd=(
  sacct
  -X
  --starttime "$START_TIME"
  --format="JobID,JobName%${JOB_NAME_WIDTH},State,Reason%${REASON_WIDTH},Elapsed,Timelimit,ExitCode"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

"${cmd[@]}"
