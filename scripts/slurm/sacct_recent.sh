#!/bin/bash
# Show recent Marlowe jobs with descriptive names and scheduling/failure reasons.

set -euo pipefail

START_TIME="${START_TIME:-today}"

cmd=(
  sacct
  -X
  --starttime "$START_TIME"
  --parsable2
  --noheader
  --format="JobName,JobID,State,Reason,Elapsed,Timelimit,ExitCode"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

"${cmd[@]}" | awk -F'|' '
function dashes(n, out) {
  out = ""
  while (length(out) < n) {
    out = out "-"
  }
  return out
}
BEGIN {
  headers[1] = "JobName"
  headers[2] = "JobID"
  headers[3] = "State"
  headers[4] = "Reason"
  headers[5] = "Elapsed"
  headers[6] = "Timelimit"
  headers[7] = "ExitCode"
  for (i = 1; i <= 7; i++) {
    width[i] = length(headers[i])
  }
}
{
  row_count++
  for (i = 1; i <= 7; i++) {
    rows[row_count, i] = $i
    if (length($i) > width[i]) {
      width[i] = length($i)
    }
  }
}
END {
  for (i = 1; i <= 7; i++) {
    printf "%-*s%s", width[i], headers[i], (i == 7 ? "\n" : "  ")
  }
  for (i = 1; i <= 7; i++) {
    printf "%-*s%s", width[i], dashes(width[i]), (i == 7 ? "\n" : "  ")
  }
  for (r = 1; r <= row_count; r++) {
    for (i = 1; i <= 7; i++) {
      printf "%-*s%s", width[i], rows[r, i], (i == 7 ? "\n" : "  ")
    }
  }
}
'
