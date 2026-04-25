# Synthetic Photo Vocab Assets

This directory contains the generated-photo rebuild of the synthetic vocabulary task.

## Goal

Build a 170-item, 4-choice vocabulary task for children aged 3-11. Each item has one correct vocabulary term and three distractors. Images should be photo-like, globally recognizable where possible, and free of text, letters, numbers, logos, signs, captions, labels, watermarks, icons, cartoons, and clip art.

## Current Version

The active generated-photo pilot asset version is:

`assets/new-vocab-photo-pilot-100-v1`

Important files:

- `manifest.csv`: task manifest consumed by `SyntheticVocabDataset`.
- `translations/item-bank-translations.csv`: localized prompt rows.
- `metadata/image_prompts.jsonl`: one audited image-generation prompt per unique term.
- `metadata/distractor_plan.csv`: target, similar distractor, and all three distractors per item.
- `metadata/vocab_generation_report.json`: summary counts and generation policy.
- `visual/vocab/*.png`: generated images, once image generation has been run.

## Design Rationale

The previous synthetic asset set used simple PIL-drawn icons with visible text labels. That made many images visually near-identical and leaked the answer through OCR. This rebuild requests realistic photographs and explicitly prohibits text or label-like content in each image prompt.

The intended full task uses 170 target vocabulary terms and 680 unique option images. A 170-item, 4-choice task has 680 option placements, so distractors are drawn from the broader filtered THINGS/AoA pool rather than only from the target list. This costs more to generate and review, but gives each item a carefully selected distractor set without reusing target images as distractors. The current local pilot has 100 target vocabulary terms and 400 unique option images.

Each item receives a unique 3-distractor set. Distractor terms are not reused across items and do not overlap the 170 target answers. Following the Long et al. style, distractors are selected from candidates near the target AoA when possible: one high-similarity distractor, one medium-similarity distractor, and one low-similarity distractor. Similarity uses OpenAI CLIP text embeddings when available, with a category/tag/age heuristic fallback.

## THINGS/AoA Inputs

The generator builds the 170-term pool from external lexical resources when these files are present under `scripts/new_vocab_assets/inputs/`:

- `things_meta.csv`: THINGS concepts and metadata. Required columns: `concept_id`, `label`, `child_safe`, `nameability`, `animacy`.
- `aoa_kuperman.csv`: age-of-acquisition norms. Required columns: `word`, `aoa`.
- `original_108.csv`: the 108 original target labels from Long et al. Required column: `label`.

These files have been downloaded/normalized in this workspace. Raw source files are kept in `scripts/new_vocab_assets/inputs/raw/`, and source details are documented in `scripts/new_vocab_assets/inputs/PROVENANCE.md`.

When all three files are present, the generator automatically uses them instead of the legacy embedded lexicon. It filters to `child_safe` THINGS concepts, joins AoA by label, excludes the original 108 labels, and keeps terms with AoA at or below 11. It then splits the filtered pool into low/mid/high AoA terciles and samples 57/56/57 items from those bands with a deterministic seed, drawing from the more nameable side of each band. Age bands are derived from the original numeric AoA and clamped to 3-11. The local `things_meta.csv` uses THINGSplus nameability and a conservative child-safety heuristic; review this before treating the pool as final.

With these inputs present, similar distractors use CLIP text embeddings by default. Install CLIP from OpenAI's repository:

```bash
pip install git+https://github.com/openai/CLIP.git
```

To run without CLIP similarity, pass `--no-use-clip-similarity`.

## Generation Instructions

The generator is:

`scripts/experimental/generate_photo_vocab_task.py`

Dry run, which writes the manifest, prompt log, distractor plan, translations, and report without creating images:

```bash
python scripts/experimental/generate_photo_vocab_task.py \
  --version new-vocab-photo-pilot-100-v1 \
  --n-items 100 \
  --seed-items-manifest scripts/new_vocab_assets/assets/new-vocab-photo-pilot-50-v1/manifest.csv \
  --provider gemini
```

Generate images with Gemini/Imagen:

```bash
python scripts/experimental/generate_photo_vocab_task.py \
  --version new-vocab-photo-pilot-100-v1 \
  --n-items 100 \
  --seed-items-manifest scripts/new_vocab_assets/assets/new-vocab-photo-pilot-50-v1/manifest.csv \
  --provider gemini \
  --generate-images \
  --validate
```

The script loads `.env` by default, so a local `GEMINI_API_KEY` or `GOOGLE_API_KEY` there is sufficient. The default Gemini image model is `imagen-4.0-generate-001`, requested through the Gemini API REST endpoint with `personGeneration` set to `dont_allow`.

## Validation And Review

The built-in validation checks:

- all manifest rows use task ID `synthetic-vocab`
- each item has exactly three distractors
- no item includes its answer as a distractor
- all distractor sets are unique
- every option term has a prompt record
- every generated image file exists
- generated images are readable and at least 512 px on each side
- exact duplicate and near-duplicate image candidates are flagged

Manual review is still required before scoring. Reviewers should inspect `metadata/distractor_plan.csv` for the high/medium/low distractor choices and inspect generated images for text leakage, cultural specificity, ambiguity, and photo realism.

## Current Results

As of the latest dry run:

- 100 manifest items were generated for `new-vocab-photo-pilot-100-v1`.
- The first 50 rows from `new-vocab-photo-pilot-50-v1` are preserved exactly.
- 400 unique image prompts were generated.
- 100 unique distractor sets were generated.
- 300 unique non-target distractor terms were generated.
- The manifest was generated from the THINGS/AoA/Long input CSVs, not the legacy embedded lexicon.
- Target terms are balanced across low/mid/high AoA terciles rather than selecting only the earliest-acquired terms.
- Age bands are clamped to 3-11.
- `configs/tasks/synthetic_vocab.yaml` points to `new-vocab-photo-pilot-100-v1/manifest.csv`.
- Full image generation has passed validation for the 100-item pilot in this workspace.
