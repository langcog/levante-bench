#!/bin/bash

# Ensure sbatch is available in PATH.
# Intended to be sourced by submit scripts.

ensure_sbatch_available() {
  if command -v sbatch >/dev/null 2>&1; then
    return 0
  fi

  # In non-login shells, module may not be initialized.
  if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh || true
    fi
  fi

  if type module >/dev/null 2>&1; then
    # Try the explicit version first, then generic module name.
    module load slurm/slurm/25.05.2 >/dev/null 2>&1 || module load slurm >/dev/null 2>&1 || true
    hash -r
  fi

  if command -v sbatch >/dev/null 2>&1; then
    echo "INFO: auto-loaded Slurm module; sbatch is now available."
    return 0
  fi

  echo "ERROR: sbatch not found in PATH." >&2
  echo "Tried auto-loading Slurm module but it is still unavailable." >&2
  echo "Run on a Marlowe login node and initialize modules, e.g.:" >&2
  echo "  source /etc/profile.d/modules.sh && module load slurm/slurm/25.05.2" >&2
  return 127
}
