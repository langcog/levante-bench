# Synthetic Vocabulary Corpus and Image Generation

This document describes how the LEVANTE synthetic vocabulary task was built: the lexical resources, item-construction rules, image-generation models and prompts, and how to reproduce or extend the corpus.

**This work lives on the `v2` branch** of [`langcog/levante-bench`](https://github.com/langcog/levante-bench). Clone that branch before using the generator, task loader, or downloaded assets:

```bash
git clone https://github.com/langcog/levante-bench.git
cd levante-bench
git checkout v2
```

The `main` branch does not contain this pipeline or the photo-vocab asset layout.

---

## What we built

A **4-choice picture vocabulary task** for children aged 3–11. Each item shows four photographs (A–D). The model (or child) must pick the image that matches a target word.

The current released pilot is **`new-vocab-photo-pilot-100-v1`**:

- 100 target items
- 400 unique option images (one image per option placement; distractors are never reused as other targets)
- English, German, and Spanish prompt rows
- Stored in GCS at `gs://levante-bench/corpus_data/synth_vocab` and locally at `data/assets/synth_vocab/`

The design target for a full bank is 170 items / 680 unique images. The 100-item set is the validated local pilot.

The task is registered as `synthetic-vocab` and loaded by `SyntheticVocabDataset` (`src/levante_bench/tasks/synthetic_vocab.py`). The active task config is `configs/additional_tasks/synthetic_vocab.yaml`, which points at `new-vocab-photo-pilot-100-v1/manifest.csv`.

---

## Why we rebuilt it

An earlier generator (`scripts/experimental/generate_vocab_task.py`) drew simple **PIL icons** from an embedded lexicon, often with visible text labels. Those images were visually near-identical across items and leaked the answer through OCR.

The photo rebuild (`scripts/experimental/generate_photo_vocab_task.py`) requests realistic photographs, keeps one unique image per option term, and avoids people, weapons, and label-like content. Prompts are written to disk and reviewed before any paid image API calls.

---

## How the lexicon was built

The generator does **not** use the legacy embedded word list when the three input CSVs below are present under `data/assets/synth_vocab/inputs/`. Source files and column mappings are recorded in [`data/assets/synth_vocab/inputs/PROVENANCE.md`](../data/assets/synth_vocab/inputs/PROVENANCE.md). Raw downloads sit in `inputs/raw/`.

### Input files

| File | Role | Source |
|---|---|---|
| `things_meta.csv` | Object concepts plus `child_safe`, `nameability`, `animacy` | THINGS / THINGSplus OSF project [`jum2f`](https://osf.io/jum2f/) |
| `aoa_kuperman.csv` | Age-of-acquisition norms | Hugging Face [`StephanAkkerman/English-Age-of-Acquisition`](https://huggingface.co/datasets/StephanAkkerman/English-Age-of-Acquisition) (`en.aoa.csv`) |
| `original_108.csv` | Labels to **exclude** so this bank does not overlap the public visual-vocabulary task | DevBench Visual Vocabulary manifest ([`alvinwmtan/dev-bench`](https://github.com/alvinwmtan/dev-bench)); first 108 labels, matching the original Long et al. target count |

Column notes (from the provenance file):

- `child_safe` is a **local conservative heuristic**, not an official THINGS field.
- `nameability` is THINGSplus `image-label_nameability_mean`.
- `animacy` normalizes THINGSplus `property_lives_mean` (1–7) to 0–1.
- `aoa` prefers Kuperman lemma ratings (`AoA_Kup_lem`) with `AoA_Kup` fallback.

### Sampling policy

1. Keep THINGS concepts marked `child_safe`.
2. Join Kuperman AoA by label.
3. Drop the original 108 Long / DevBench target labels.
4. Keep terms with AoA ≤ 11.
5. Split the remaining pool into low / mid / high AoA terciles.
6. Sample with a nameability-biased deterministic seed (`--seed`, default `42`). The 170-item design draws 57 / 56 / 57 from those bands; the 100-item pilot is a stratified subset of the same policy.
7. Convert numeric AoA to integer age bands and clamp to **3–11**.

The 100-item pilot **preserves the first 50 rows** of `new-vocab-photo-pilot-50-v1` exactly (`--seed-items-manifest`), then adds 50 new items whose option terms do not collide with the reserved images.

People, body parts, weapons, and a few other hard-to-depict or unsafe terms are excluded from image generation (`GENERATED_IMAGE_EXCLUDED_TERMS` in the generator).

---

## How items and distractors were built

Each item has one target and **three unique distractors**. Distractor terms:

- come from the broader filtered THINGS/AoA pool, not from other targets
- are not reused across items
- do not overlap the target-answer set

Following the Long et al. visual-vocabulary style, candidates are restricted to a nearby AoA window when possible. The three distractors are then tiered by text similarity:

1. **High** similarity to the target
2. **Medium** similarity (prefer same animacy / category, then the middle of the similarity range)
3. **Low** similarity

**Similarity model:** OpenAI CLIP **ViT-B/32** text embeddings (default). Each label is encoded with the prompt `a clear photo of {label}`. Cosine similarity ranks candidates. Install CLIP with:

```bash
pip install git+https://github.com/openai/CLIP.git
```

If CLIP is unavailable, pass `--no-use-clip-similarity` to fall back to a category / tag / age heuristic.

The per-item plan is written to `metadata/distractor_plan.csv`.

---

## How images were generated

### Generator

`scripts/experimental/generate_photo_vocab_task.py`

A dry run writes the manifest, translations, image-prompt log, distractor plan, and generation report **without** calling an image API. Image generation is opt-in (`--generate-images`).

### Models

The documented production run used **Gemini / Imagen**:

| Setting | Value |
|---|---|
| Provider | `gemini` |
| Model | `imagen-4.0-generate-001` |
| Endpoint | Gemini API REST `v1beta/models/{model}:predict` |
| Aspect ratio | 1:1 |
| `personGeneration` | `dont_allow` |
| Output | PNG under `visual/vocab/{term}.png`, at least 512 px on each side |

API key: `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` (the script loads `.env` by default). See also [secrets_setup.md](secrets_setup.md).

An **OpenAI Images** path remains in the code (`--provider openai`, default model `gpt-image-1`, size `512x512`, requires `OPENAI_API_KEY`). That is not the run used for the current 100-item photo pilot.

### Image prompts

One audited prompt per unique option term is stored in `metadata/image_prompts.jsonl`. The template is:

> Realistic color photograph of a single **{term}**. Make the subject **{difficulty}** and globally recognizable. Center the subject, use clear lighting, and keep it easy to identify. Use a plain or natural uncluttered background. Make it look like an ordinary camera photo with only the subject and background. *{optional culture-generic note}*

Difficulty wording depends on age band:

- ≤ 5: “very familiar to young children”
- ≤ 8: “familiar to children”
- otherwise: “clear and recognizable for older children”

A small `GLOBAL_FRIENDLY_NOTES` map adds extra constraints for culturally specific or landmark-prone terms (for example barn, castle, igloo, pagoda, gondola, scepter, totem): use a generic object, avoid famous landmarks, signage, people, and sacred or identifiable cultural designs.

The design intent is photo-like images **without** text, letters, numbers, logos, signs, captions, labels, watermarks, icons, cartoons, or clip art. Reviewers still inspect generated files for leakage and ambiguity; validation also flags exact and near-duplicate images.

---

## Task prompts (what the VLM sees)

Manifest `full_prompt` (English):

> You will see four pictures labeled A, B, C, and D. Choose the picture that best matches the word: {word}.

Localized rows in `translations/item-bank-translations.csv`:

| Language | Prompt |
|---|---|
| `en` | Which picture shows {word}? |
| `de` | Welches Bild zeigt {word}? |
| `es` | Que imagen muestra {word}? |

At evaluation time, `SyntheticVocabDataset` binds the four option photographs to A–D with `<imageN>` placeholders so model adapters can attach the images:

> Choose the image that matches the text: "{word}". Answer with A, B, C, or D. A: `<image1>`; B: `<image2>`; C: `<image3>`; D: `<image4>`

Option order is deterministic from `item_uid` unless `true_random_option_order` is enabled.

---

## Outputs of a generation run

For a version directory such as `data/assets/synth_vocab/assets/new-vocab-photo-pilot-100-v1/`:

| Path | Contents |
|---|---|
| `manifest.csv` | Items consumed by `SyntheticVocabDataset` |
| `translations/item-bank-translations.csv` | `en` / `de` / `es` prompt rows |
| `metadata/image_prompts.jsonl` | One image-generation prompt per unique term |
| `metadata/distractor_plan.csv` | High / medium / low distractors per item |
| `metadata/vocab_generation_report.json` | Counts, age-band mix, and written policy |
| `visual/vocab/*.png` | Generated photographs |

Built-in `--validate` checks: task ID, three distractors, no answer-as-distractor, unique distractor sets, a prompt for every option term, readable images ≥ 512 px, and duplicate / near-duplicate flags.

Manual review is still required: inspect `distractor_plan.csv` and the PNGs for text leakage, cultural specificity, ambiguity, and photo realism.

The 100-item pilot report records 100 unique distractor sets, 400 unique image terms, 12 animal / 88 object targets, and age-band counts spanning 3–11.

---

## Getting started

### 1. Check out `v2`

```bash
git clone https://github.com/langcog/levante-bench.git
cd levante-bench
git checkout v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Use Python 3.10–3.13. CLIP similarity additionally needs `pip install git+https://github.com/openai/CLIP.git` (and typically `requirements-transformers.txt` / a working `torch`).

### 2. Use the existing corpus (recommended)

If you only need to **run** the task, download the already-generated assets (requires `gcloud` auth to the LEVANTE bucket):

```bash
python scripts/data_prep/download_synth_vocab_assets.py
```

That rsyncs `gs://levante-bench/corpus_data/synth_vocab` into `data/assets/synth_vocab/`. Then:

```bash
levante-bench list-tasks
levante-bench run-eval --task synthetic-vocab --model clip_base --device auto
```

Swap in any registered VLM (`levante-bench list-models`). Use `--prompt-language de` or `es` for the translated prompts.

### 3. Regenerate a corpus (optional)

Put `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `.env`. Confirm the three input CSVs exist under `data/assets/synth_vocab/inputs/`.

**Dry run** (manifest + prompts only, no image API cost):

```bash
python scripts/experimental/generate_photo_vocab_task.py \
  --version new-vocab-photo-pilot-100-v1 \
  --n-items 100 \
  --seed-items-manifest data/assets/synth_vocab/assets/new-vocab-photo-pilot-50-v1/manifest.csv \
  --provider gemini
```

**Generate images** and validate:

```bash
python scripts/experimental/generate_photo_vocab_task.py \
  --version new-vocab-photo-pilot-100-v1 \
  --n-items 100 \
  --seed-items-manifest data/assets/synth_vocab/assets/new-vocab-photo-pilot-50-v1/manifest.csv \
  --provider gemini \
  --generate-images \
  --validate
```

Useful flags:

- `--n-items 170` for the full design size
- `--limit-images N` for a cheap image-generation smoke test
- `--continue-on-error` to skip failed API calls during a pilot
- `--overwrite-images` to regenerate existing PNGs
- `--no-use-clip-similarity` if CLIP is not installed
- `--provider openai --model gpt-image-1` for the OpenAI Images path

Point `configs/additional_tasks/synthetic_vocab.yaml` `corpus_file` at the new `manifest.csv` before evaluating.

The older PIL-icon generator (`scripts/experimental/generate_vocab_task.py`) is **not** the current corpus. Do not mix icon and photo assets in one eval.

---

## Related code

| Path | Role |
|---|---|
| `scripts/experimental/generate_photo_vocab_task.py` | Photo corpus + Imagen / OpenAI image generation |
| `scripts/experimental/generate_vocab_task.py` | Legacy PIL-icon generator (superseded) |
| `scripts/data_prep/download_synth_vocab_assets.py` | Download generated assets from GCS |
| `src/levante_bench/tasks/synthetic_vocab.py` | Task loader |
| `configs/additional_tasks/synthetic_vocab.yaml` | Active 100-item pilot config |
| `data/assets/synth_vocab/README.md` | Short asset-directory notes |
| `data/assets/synth_vocab/inputs/PROVENANCE.md` | Input-file provenance |

---

## References

**THINGS / THINGSplus**

- Hebart, M. N., Dickter, A. H., Kidder, A., Kwok, W. Y., Corriveau, A., Van Wicklin, C., & Baker, C. I. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PLOS ONE, 14*(10), e0223792. https://doi.org/10.1371/journal.pone.0223792
- Stoinski, L. M., Perkuhn, J., & Hebart, M. N. (2024). THINGSplus: New norms and metadata for the THINGS database of 1,854 object concepts and 26,107 natural object images. *Behavior Research Methods, 56*, 1583–1603. https://doi.org/10.3758/s13428-023-02102-8
- OSF project: https://osf.io/jum2f/

**Age of acquisition**

- Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-acquisition ratings for 30,000 English words. *Behavior Research Methods, 44*, 978–990. https://doi.org/10.3758/s13428-012-0210-4
- Hugging Face mirror used here: https://huggingface.co/datasets/StephanAkkerman/English-Age-of-Acquisition

**Visual vocabulary / DevBench (items we deliberately do not reuse)**

- Long, B., Fan, J. E., Huey, H., Chai, Z., & Frank, M. C. (2024). Parallel developmental changes in children’s production and recognition of line drawings of visual concepts. *Nature Communications, 15*, 703. https://doi.org/10.1038/s41467-023-44529-9
- DevBench Visual Vocabulary manifest: https://github.com/alvinwmtan/dev-bench (`assets/lex-viz_vocab/manifest.csv`)

**Similarity and image models**

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of ICML*. CLIP code: https://github.com/openai/CLIP (ViT-B/32)
- Google DeepMind Imagen 4 (`imagen-4.0-generate-001`) via the Gemini API: https://ai.google.dev/gemini-api/docs/imagen
- Optional alternative: OpenAI Images API, model `gpt-image-1`

**This repository**

- Code and configs: https://github.com/langcog/levante-bench/tree/v2
- Generated assets: `gs://levante-bench/corpus_data/synth_vocab`

---

## Appendix A. Item bank (100 items, 400 photographs)

The 100-item pilot (`new-vocab-photo-pilot-100-v1`) uses 400 unique option terms: one target plus three distractors per item. Distractors are ordered high / medium / low CLIP-text similarity.

Thumbnails are 96×96 JPEG previews of the Imagen 4 photographs, committed under `docs/assets/synth_vocab_thumbs/` and linked in the appendix via GitHub raw URLs on `v2` so they render outside the repo. Original PNGs live in `data/assets/synth_vocab/assets/new-vocab-photo-pilot-100-v1/visual/vocab/` (not stored in git; download with `scripts/data_prep/download_synth_vocab_assets.py`).

**Full table with all 400 thumbnails:** [synthetic_vocab_appendix.md](synthetic_vocab_appendix.md)
