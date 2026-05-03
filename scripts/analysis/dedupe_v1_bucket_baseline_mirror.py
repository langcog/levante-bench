#!/usr/bin/env python3
"""Remove redundant benchmark artifacts next to ``baseline/`` on GCS.

For each model folder under ``results/v1/<model>/``, if an object exists at::

    gs://<bucket>/results/v1/<model>/<rel>

and the mirror exists at::

    gs://<bucket>/results/v1/<model>/baseline/<rel>

then delete the **non-baseline** copy (the overlay in the model root).

Rules:

- Never delete anything under ``baseline/``.
- Never delete anything whose first path segment under ``<model>/`` is a **four-digit**
  run directory (``0001``, ``0010``, …).
- Does **not** delete model folders or unrelated prefixes; only matching duplicate paths.
- Skips paths ending with ``/`` (``gcloud storage ls`` often emits prefix/container rows that are not a single removable object).

Default is **dry-run**; pass ``--execute`` to perform ``gcloud storage rm``.

Examples::

    python scripts/analysis/dedupe_v1_bucket_baseline_mirror.py
    python scripts/analysis/dedupe_v1_bucket_baseline_mirror.py --execute
    python scripts/analysis/dedupe_v1_bucket_baseline_mirror.py --bucket gs://levante-bench --prefix results/v1 --models tinyllava internvl35-4B
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath


RUN_SEGMENT = re.compile(r"^\d{4}$")


def _normalize_ls_line(line: str) -> str:
    """Strip ``gcloud storage ls`` cruft (lines are sometimes printed as ``gs://...path:``)."""
    s = line.strip()
    while s.endswith(":"):
        s = s[:-1].rstrip()
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", default="gs://levante-bench", help="Bucket URI (default: gs://levante-bench).")
    p.add_argument(
        "--prefix",
        default="results/v1",
        help="Prefix inside bucket without leading slash (default: results/v1).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        help="Only process these model folder names (default: all immediate children under prefix).",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete objects (default: dry-run only).",
    )
    return p.parse_args()


def _no_objects(stderr: str) -> bool:
    s = stderr.lower()
    return "matched no objects" in s or "matched zero objects" in s


def gs_ls_recursive(uri: str) -> list[str]:
    """Return gs:// object URIs under uri (recursive)."""
    cmd = ["gcloud", "storage", "ls", "--recursive", uri]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if _no_objects(proc.stderr):
            return []
        raise RuntimeError(f"gcloud storage ls failed ({proc.returncode}): {proc.stderr.strip()}")
    lines = [_normalize_ls_line(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    return lines


def gs_ls_immediate(uri: str) -> list[str]:
    cmd = ["gcloud", "storage", "ls", uri]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        if _no_objects(proc.stderr):
            return []
        raise RuntimeError(f"gcloud storage ls failed ({proc.returncode}): {proc.stderr.strip()}")
    lines = [_normalize_ls_line(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    return lines


def gs_rm(uri: str, *, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN rm {uri}")
        return
    proc = subprocess.run(
        ["gcloud", "storage", "rm", uri],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud storage rm failed ({proc.returncode}): {proc.stderr.strip()}")
    print(f"DELETED {uri}")


def normalize_bucket(bucket: str) -> str:
    b = bucket.strip().rstrip("/")
    if not b.startswith("gs://"):
        b = f"gs://{b.removeprefix('gs://')}"
    return b


def object_uri_under_model(model_root: str, rel: str) -> str:
    """Build full gs:// URI for ``rel`` under ``model_root`` (must be non-empty relative path)."""
    root = model_root.rstrip("/")
    tail = _normalize_ls_line(rel.strip().lstrip("/"))
    if not tail or tail == ":":
        raise ValueError("empty object path relative to model root")
    return f"{root}/{tail}"


def main() -> int:
    args = parse_args()
    bucket = normalize_bucket(args.bucket)
    prefix = args.prefix.strip().strip("/")
    base = f"{bucket}/{prefix}/"
    # Strip paths relative to gs://bucket/<prefix>/ — not bucket root only (would wrongly use "results").
    v1_root = f"{bucket}/{prefix}".rstrip("/")

    try:
        children = gs_ls_immediate(base)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    model_uris: list[str] = []
    for raw in children:
        line = _normalize_ls_line(raw).rstrip("/")
        # Immediate child: gs://.../results/v1/<model> or gs://.../results/v1/<model>/...
        if not (line.startswith(v1_root + "/") or line == v1_root):
            continue
        suffix = line[len(v1_root) :].lstrip("/")
        if not suffix:
            continue
        seg = suffix.split("/", 1)[0]
        if not seg:
            continue
        model_uris.append(f"{v1_root}/{seg}/")

    if args.models:
        wanted = set(args.models)
        filtered = []
        for mu in model_uris:
            name = mu.rstrip("/").rsplit("/", 1)[-1]
            if name in wanted:
                filtered.append(mu)
        model_uris = sorted(filtered)
        missing = wanted - {mu.rstrip("/").rsplit("/", 1)[-1] for mu in model_uris}
        if missing:
            print(f"WARNING: no listing under prefix for models: {sorted(missing)}", file=sys.stderr)

    dry_run = not args.execute
    total_delete = 0

    for model_root in sorted(set(model_uris)):
        model_name = model_root.rstrip("/").rsplit("/", 1)[-1]
        try:
            objs = gs_ls_recursive(model_root)
        except RuntimeError as exc:
            print(f"{model_name}: skip ({exc})", file=sys.stderr)
            continue

        scan_prefix = model_root.rstrip("/") + "/"
        rels: set[str] = set()
        for uri in objs:
            u = _normalize_ls_line(uri)
            if u.startswith(scan_prefix):
                tail = u[len(scan_prefix) :].strip()
            elif u.rstrip("/") == scan_prefix.rstrip("/"):
                continue
            else:
                continue
            if not tail:
                continue
            rels.add(tail)

        baseline_marker = "baseline/"
        if not any(r.startswith(baseline_marker) for r in rels):
            continue

        to_remove: list[str] = []
        for rel in sorted(rels):
            rel = _normalize_ls_line(rel).strip().lstrip("/")
            if not rel:
                continue
            parts = PurePosixPath(rel).parts
            if not parts:
                continue
            if parts[0] == "baseline":
                continue
            if RUN_SEGMENT.match(parts[0]):
                continue
            # Prefix/container lines (not a leaf object); plain `gcloud storage rm` cannot remove these.
            if rel.endswith("/"):
                continue
            mirror = f"baseline/{rel}"
            if mirror in rels:
                to_remove.append(rel)

        if not to_remove:
            continue

        print(f"\n## {model_name}  ({len(to_remove)} overlay object(s) mirrored in baseline/)")
        for rel in to_remove:
            try:
                uri = object_uri_under_model(model_root, rel)
            except ValueError:
                print(f"{model_name}: skip invalid relative path {rel!r}", file=sys.stderr)
                continue
            gs_rm(uri, dry_run=dry_run)
            total_delete += 1

    print(f"\nDone. overlay_objects_removed={total_delete} execute={args.execute}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
