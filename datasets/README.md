# Canonical Dataset Manifests

This directory contains stable, machine-readable metadata manifests for
benchmark datasets used by this repository.

These manifests are intended to be the canonical source for:

- dataset-level metadata (name, version, description, license, citation);
- storage/distribution locations (bucket prefixes, URI templates);
- split definitions and filters;
- logical record schemas used by evaluation code; and
- provenance references (source papers and preprocessing scripts).

The immediate goal is to provide a single source of truth that can later be
used to generate Croissant metadata with minimal custom logic.
