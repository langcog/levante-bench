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

Thumbnails are 96×96 JPEG previews of the Imagen 4 photographs, loaded from GitHub raw URLs on `v2` so they render if this file is opened outside the repo. Original PNGs live in `data/assets/synth_vocab/assets/new-vocab-photo-pilot-100-v1/visual/vocab/` (not stored in git; download with `scripts/data_prep/download_synth_vocab_assets.py`).

GitHub-rendered copy: [https://github.com/langcog/levante-bench/blob/v2/docs/synthetic_vocab_appendix.md](https://github.com/langcog/levante-bench/blob/v2/docs/synthetic_vocab_appendix.md)

| # | Age | Category | Target | High-similarity distractor | Medium-similarity distractor | Low-similarity distractor |
| ---: | ---: | --- | :---: | :---: | :---: | :---: |
| 1 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/key.jpg" alt="key" width="96"><br>`key` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/rag.jpg" alt="rag" width="96"><br>`rag` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dollhouse.jpg" alt="dollhouse" width="96"><br>`dollhouse` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lava.jpg" alt="lava" width="96"><br>`lava` |
| 2 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/banana.jpg" alt="banana" width="96"><br>`banana` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lemon.jpg" alt="lemon" width="96"><br>`lemon` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crib.jpg" alt="crib" width="96"><br>`crib` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hyena.jpg" alt="hyena" width="96"><br>`hyena` |
| 3 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/button.jpg" alt="button" width="96"><br>`button` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/knob.jpg" alt="knob" width="96"><br>`knob` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/coal.jpg" alt="coal" width="96"><br>`coal` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/floss.jpg" alt="floss" width="96"><br>`floss` |
| 4 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/umbrella.jpg" alt="umbrella" width="96"><br>`umbrella` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/raincoat.jpg" alt="raincoat" width="96"><br>`raincoat` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chip.jpg" alt="chip" width="96"><br>`chip` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/noodle.jpg" alt="noodle" width="96"><br>`noodle` |
| 5 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/celery.jpg" alt="celery" width="96"><br>`celery` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lettuce.jpg" alt="lettuce" width="96"><br>`lettuce` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cooler.jpg" alt="cooler" width="96"><br>`cooler` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/anteater.jpg" alt="anteater" width="96"><br>`anteater` |
| 6 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hydrant.jpg" alt="hydrant" width="96"><br>`hydrant` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hose.jpg" alt="hose" width="96"><br>`hose` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/throne.jpg" alt="throne" width="96"><br>`throne` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cable.jpg" alt="cable" width="96"><br>`cable` |
| 7 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/microwave.jpg" alt="microwave" width="96"><br>`microwave` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/oven.jpg" alt="oven" width="96"><br>`oven` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/eggnog.jpg" alt="eggnog" width="96"><br>`eggnog` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/suspenders.jpg" alt="suspenders" width="96"><br>`suspenders` |
| 8 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stethoscope.jpg" alt="stethoscope" width="96"><br>`stethoscope` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/headset.jpg" alt="headset" width="96"><br>`headset` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hinge.jpg" alt="hinge" width="96"><br>`hinge` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/aardvark.jpg" alt="aardvark" width="96"><br>`aardvark` |
| 9 | 3 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sock.jpg" alt="sock" width="96"><br>`sock` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shoe.jpg" alt="shoe" width="96"><br>`shoe` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/snowball.jpg" alt="snowball" width="96"><br>`snowball` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ladybug.jpg" alt="ladybug" width="96"><br>`ladybug` |
| 10 | 3 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chair.jpg" alt="chair" width="96"><br>`chair` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/recliner.jpg" alt="recliner" width="96"><br>`recliner` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/streetlight.jpg" alt="streetlight" width="96"><br>`streetlight` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dalmatian.jpg" alt="dalmatian" width="96"><br>`dalmatian` |
| 11 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bell.jpg" alt="bell" width="96"><br>`bell` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/van.jpg" alt="van" width="96"><br>`van` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/suit.jpg" alt="suit" width="96"><br>`suit` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/jellyfish.jpg" alt="jellyfish" width="96"><br>`jellyfish` |
| 12 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/phone.jpg" alt="phone" width="96"><br>`phone` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/game.jpg" alt="game" width="96"><br>`game` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dynamite.jpg" alt="dynamite" width="96"><br>`dynamite` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chipmunk.jpg" alt="chipmunk" width="96"><br>`chipmunk` |
| 13 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/apple.jpg" alt="apple" width="96"><br>`apple` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/candy.jpg" alt="candy" width="96"><br>`candy` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cardboard.jpg" alt="cardboard" width="96"><br>`cardboard` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/porcupine.jpg" alt="porcupine" width="96"><br>`porcupine` |
| 14 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/balloon.jpg" alt="balloon" width="96"><br>`balloon` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ball.jpg" alt="ball" width="96"><br>`ball` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tape.jpg" alt="tape" width="96"><br>`tape` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/asparagus.jpg" alt="asparagus" width="96"><br>`asparagus` |
| 15 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/scissors.jpg" alt="scissors" width="96"><br>`scissors` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pliers.jpg" alt="pliers" width="96"><br>`pliers` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/coin.jpg" alt="coin" width="96"><br>`coin` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/meatloaf.jpg" alt="meatloaf" width="96"><br>`meatloaf` |
| 16 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/leaf.jpg" alt="leaf" width="96"><br>`leaf` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/plant.jpg" alt="plant" width="96"><br>`plant` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/brownie.jpg" alt="brownie" width="96"><br>`brownie` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/gumball.jpg" alt="gumball" width="96"><br>`gumball` |
| 17 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tomato.jpg" alt="tomato" width="96"><br>`tomato` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ketchup.jpg" alt="ketchup" width="96"><br>`ketchup` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lumber.jpg" alt="lumber" width="96"><br>`lumber` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/perfume.jpg" alt="perfume" width="96"><br>`perfume` |
| 18 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/elephant.jpg" alt="elephant" width="96"><br>`elephant` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/horse.jpg" alt="horse" width="96"><br>`horse` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/penguin.jpg" alt="penguin" width="96"><br>`penguin` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/xylophone.jpg" alt="xylophone" width="96"><br>`xylophone` |
| 19 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/camel.jpg" alt="camel" width="96"><br>`camel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/donkey.jpg" alt="donkey" width="96"><br>`donkey` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/oyster.jpg" alt="oyster" width="96"><br>`oyster` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/screwdriver.jpg" alt="screwdriver" width="96"><br>`screwdriver` |
| 20 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/broccoli.jpg" alt="broccoli" width="96"><br>`broccoli` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/vegetable.jpg" alt="vegetable" width="96"><br>`vegetable` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ring.jpg" alt="ring" width="96"><br>`ring` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/splinter.jpg" alt="splinter" width="96"><br>`splinter` |
| 21 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/helicopter.jpg" alt="helicopter" width="96"><br>`helicopter` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bus.jpg" alt="bus" width="96"><br>`bus` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/rim.jpg" alt="rim" width="96"><br>`rim` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/platypus.jpg" alt="platypus" width="96"><br>`platypus` |
| 22 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/nail.jpg" alt="nail" width="96"><br>`nail` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pill.jpg" alt="pill" width="96"><br>`pill` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bongo.jpg" alt="bongo" width="96"><br>`bongo` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/unicycle.jpg" alt="unicycle" width="96"><br>`unicycle` |
| 23 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/anchor.jpg" alt="anchor" width="96"><br>`anchor` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cross.jpg" alt="cross" width="96"><br>`cross` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/flipper.jpg" alt="flipper" width="96"><br>`flipper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/toucan.jpg" alt="toucan" width="96"><br>`toucan` |
| 24 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/punch.jpg" alt="punch" width="96"><br>`punch` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/plug.jpg" alt="plug" width="96"><br>`plug` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/checkbook.jpg" alt="checkbook" width="96"><br>`checkbook` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tarantula.jpg" alt="tarantula" width="96"><br>`tarantula` |
| 25 | 6 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/peacock.jpg" alt="peacock" width="96"><br>`peacock` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bird.jpg" alt="bird" width="96"><br>`bird` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/beetle.jpg" alt="beetle" width="96"><br>`beetle` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/clothesline.jpg" alt="clothesline" width="96"><br>`clothesline` |
| 26 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stapler.jpg" alt="stapler" width="96"><br>`stapler` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/staple.jpg" alt="staple" width="96"><br>`staple` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/earring.jpg" alt="earring" width="96"><br>`earring` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/orangutan.jpg" alt="orangutan" width="96"><br>`orangutan` |
| 27 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/camera.jpg" alt="camera" width="96"><br>`camera` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/photograph.jpg" alt="photograph" width="96"><br>`photograph` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/vacuum.jpg" alt="vacuum" width="96"><br>`vacuum` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/poinsettia.jpg" alt="poinsettia" width="96"><br>`poinsettia` |
| 28 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/onion.jpg" alt="onion" width="96"><br>`onion` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/garlic.jpg" alt="garlic" width="96"><br>`garlic` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/breakfast.jpg" alt="breakfast" width="96"><br>`breakfast` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/corkboard.jpg" alt="corkboard" width="96"><br>`corkboard` |
| 29 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mailbox.jpg" alt="mailbox" width="96"><br>`mailbox` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mail.jpg" alt="mail" width="96"><br>`mail` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pancake.jpg" alt="pancake" width="96"><br>`pancake` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/iguana.jpg" alt="iguana" width="96"><br>`iguana` |
| 30 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cauliflower.jpg" alt="cauliflower" width="96"><br>`cauliflower` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cabbage.jpg" alt="cabbage" width="96"><br>`cabbage` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/outfit.jpg" alt="outfit" width="96"><br>`outfit` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sundial.jpg" alt="sundial" width="96"><br>`sundial` |
| 31 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/card.jpg" alt="card" width="96"><br>`card` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/file.jpg" alt="file" width="96"><br>`file` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/swimsuit.jpg" alt="swimsuit" width="96"><br>`swimsuit` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tool.jpg" alt="tool" width="96"><br>`tool` |
| 32 | 6 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/owl.jpg" alt="owl" width="96"><br>`owl` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/eagle.jpg" alt="eagle" width="96"><br>`eagle` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/blowfish.jpg" alt="blowfish" width="96"><br>`blowfish` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/boulder.jpg" alt="boulder" width="96"><br>`boulder` |
| 33 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cactus.jpg" alt="cactus" width="96"><br>`cactus` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cucumber.jpg" alt="cucumber" width="96"><br>`cucumber` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sailboat.jpg" alt="sailboat" width="96"><br>`sailboat` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bassinet.jpg" alt="bassinet" width="96"><br>`bassinet` |
| 34 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hanger.jpg" alt="hanger" width="96"><br>`hanger` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hammer.jpg" alt="hammer" width="96"><br>`hammer` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/piano.jpg" alt="piano" width="96"><br>`piano` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/seal.jpg" alt="seal" width="96"><br>`seal` |
| 35 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/binoculars.jpg" alt="binoculars" width="96"><br>`binoculars` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/eyepiece.jpg" alt="eyepiece" width="96"><br>`eyepiece` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lamppost.jpg" alt="lamppost" width="96"><br>`lamppost` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tumbleweed.jpg" alt="tumbleweed" width="96"><br>`tumbleweed` |
| 36 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/iron.jpg" alt="iron" width="96"><br>`iron` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/robot.jpg" alt="robot" width="96"><br>`robot` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chalk.jpg" alt="chalk" width="96"><br>`chalk` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/backgammon.jpg" alt="backgammon" width="96"><br>`backgammon` |
| 37 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/taco.jpg" alt="taco" width="96"><br>`taco` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tee.jpg" alt="tee" width="96"><br>`tee` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/battery.jpg" alt="battery" width="96"><br>`battery` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/seismograph.jpg" alt="seismograph" width="96"><br>`seismograph` |
| 38 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tongs.jpg" alt="tongs" width="96"><br>`tongs` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shoehorn.jpg" alt="shoehorn" width="96"><br>`shoehorn` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wheat.jpg" alt="wheat" width="96"><br>`wheat` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/airboat.jpg" alt="airboat" width="96"><br>`airboat` |
| 39 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crown.jpg" alt="crown" width="96"><br>`crown` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bow.jpg" alt="bow" width="96"><br>`bow` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/veil.jpg" alt="veil" width="96"><br>`veil` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hummingbird.jpg" alt="hummingbird" width="96"><br>`hummingbird` |
| 40 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wrench.jpg" alt="wrench" width="96"><br>`wrench` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/icepick.jpg" alt="icepick" width="96"><br>`icepick` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/faucet.jpg" alt="faucet" width="96"><br>`faucet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mistletoe.jpg" alt="mistletoe" width="96"><br>`mistletoe` |
| 41 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bowtie.jpg" alt="bowtie" width="96"><br>`bowtie` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bonnet.jpg" alt="bonnet" width="96"><br>`bonnet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cardigan.jpg" alt="cardigan" width="96"><br>`cardigan` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/jumpsuit.jpg" alt="jumpsuit" width="96"><br>`jumpsuit` |
| 42 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/calculator.jpg" alt="calculator" width="96"><br>`calculator` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/computer.jpg" alt="computer" width="96"><br>`computer` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/fudge.jpg" alt="fudge" width="96"><br>`fudge` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/anvil.jpg" alt="anvil" width="96"><br>`anvil` |
| 43 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/accordion.jpg" alt="accordion" width="96"><br>`accordion` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/washboard.jpg" alt="washboard" width="96"><br>`washboard` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pulley.jpg" alt="pulley" width="96"><br>`pulley` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cheetah.jpg" alt="cheetah" width="96"><br>`cheetah` |
| 44 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chandelier.jpg" alt="chandelier" width="96"><br>`chandelier` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lightbulb.jpg" alt="lightbulb" width="96"><br>`lightbulb` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/limousine.jpg" alt="limousine" width="96"><br>`limousine` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/manatee.jpg" alt="manatee" width="96"><br>`manatee` |
| 45 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/microscope.jpg" alt="microscope" width="96"><br>`microscope` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stem.jpg" alt="stem" width="96"><br>`stem` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crossbow.jpg" alt="crossbow" width="96"><br>`crossbow` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/domino.jpg" alt="domino" width="96"><br>`domino` |
| 46 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/avocado.jpg" alt="avocado" width="96"><br>`avocado` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/guacamole.jpg" alt="guacamole" width="96"><br>`guacamole` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dial.jpg" alt="dial" width="96"><br>`dial` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pillbox.jpg" alt="pillbox" width="96"><br>`pillbox` |
| 47 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/clipper.jpg" alt="clipper" width="96"><br>`clipper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/riser.jpg" alt="riser" width="96"><br>`riser` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/corkscrew.jpg" alt="corkscrew" width="96"><br>`corkscrew` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sawhorse.jpg" alt="sawhorse" width="96"><br>`sawhorse` |
| 48 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/plunger.jpg" alt="plunger" width="96"><br>`plunger` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mallet.jpg" alt="mallet" width="96"><br>`mallet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/charcoal.jpg" alt="charcoal" width="96"><br>`charcoal` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cockatoo.jpg" alt="cockatoo" width="96"><br>`cockatoo` |
| 49 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/croissant.jpg" alt="croissant" width="96"><br>`croissant` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pastry.jpg" alt="pastry" width="96"><br>`pastry` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sheath.jpg" alt="sheath" width="96"><br>`sheath` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/landmine.jpg" alt="landmine" width="96"><br>`landmine` |
| 50 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/inhaler.jpg" alt="inhaler" width="96"><br>`inhaler` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/vial.jpg" alt="vial" width="96"><br>`vial` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ramp.jpg" alt="ramp" width="96"><br>`ramp` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/boa.jpg" alt="boa" width="96"><br>`boa` |
| 51 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/toilet.jpg" alt="toilet" width="96"><br>`toilet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bathtub.jpg" alt="bathtub" width="96"><br>`bathtub` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cracker.jpg" alt="cracker" width="96"><br>`cracker` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/walrus.jpg" alt="walrus" width="96"><br>`walrus` |
| 52 | 4 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bear.jpg" alt="bear" width="96"><br>`bear` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wolf.jpg" alt="wolf" width="96"><br>`wolf` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crab.jpg" alt="crab" width="96"><br>`crab` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dustpan.jpg" alt="dustpan" width="96"><br>`dustpan` |
| 53 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/book.jpg" alt="book" width="96"><br>`book` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/paper.jpg" alt="paper" width="96"><br>`paper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dice.jpg" alt="dice" width="96"><br>`dice` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/caterpillar.jpg" alt="caterpillar" width="96"><br>`caterpillar` |
| 54 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/egg.jpg" alt="egg" width="96"><br>`egg` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/yolk.jpg" alt="yolk" width="96"><br>`yolk` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/curtain.jpg" alt="curtain" width="96"><br>`curtain` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/seatbelt.jpg" alt="seatbelt" width="96"><br>`seatbelt` |
| 55 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pencil.jpg" alt="pencil" width="96"><br>`pencil` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pen.jpg" alt="pen" width="96"><br>`pen` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mud.jpg" alt="mud" width="96"><br>`mud` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hippopotamus.jpg" alt="hippopotamus" width="96"><br>`hippopotamus` |
| 56 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bench.jpg" alt="bench" width="96"><br>`bench` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/couch.jpg" alt="couch" width="96"><br>`couch` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/jar.jpg" alt="jar" width="96"><br>`jar` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/octopus.jpg" alt="octopus" width="96"><br>`octopus` |
| 57 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/watch.jpg" alt="watch" width="96"><br>`watch` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bracelet.jpg" alt="bracelet" width="96"><br>`bracelet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pacifier.jpg" alt="pacifier" width="96"><br>`pacifier` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/reindeer.jpg" alt="reindeer" width="96"><br>`reindeer` |
| 58 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wheel.jpg" alt="wheel" width="96"><br>`wheel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/drum.jpg" alt="drum" width="96"><br>`drum` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/coconut.jpg" alt="coconut" width="96"><br>`coconut` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hook.jpg" alt="hook" width="96"><br>`hook` |
| 59 | 4 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/clock.jpg" alt="clock" width="96"><br>`clock` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/screen.jpg" alt="screen" width="96"><br>`screen` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wing.jpg" alt="wing" width="96"><br>`wing` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/peppermint.jpg" alt="peppermint" width="96"><br>`peppermint` |
| 60 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/corn.jpg" alt="corn" width="96"><br>`corn` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/grain.jpg" alt="grain" width="96"><br>`grain` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/fireworks.jpg" alt="fireworks" width="96"><br>`fireworks` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/doorknocker.jpg" alt="doorknocker" width="96"><br>`doorknocker` |
| 61 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/feather.jpg" alt="feather" width="96"><br>`feather` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/rooster.jpg" alt="rooster" width="96"><br>`rooster` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lobster.jpg" alt="lobster" width="96"><br>`lobster` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/chalkboard.jpg" alt="chalkboard" width="96"><br>`chalkboard` |
| 62 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tie.jpg" alt="tie" width="96"><br>`tie` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/strap.jpg" alt="strap" width="96"><br>`strap` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/carriage.jpg" alt="carriage" width="96"><br>`carriage` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/handprint.jpg" alt="handprint" width="96"><br>`handprint` |
| 63 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/zebra.jpg" alt="zebra" width="96"><br>`zebra` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/giraffe.jpg" alt="giraffe" width="96"><br>`giraffe` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/grasshopper.jpg" alt="grasshopper" width="96"><br>`grasshopper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/propeller.jpg" alt="propeller" width="96"><br>`propeller` |
| 64 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bat.jpg" alt="bat" width="96"><br>`bat` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/rat.jpg" alt="rat" width="96"><br>`rat` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/leech.jpg" alt="leech" width="96"><br>`leech` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/seaweed.jpg" alt="seaweed" width="96"><br>`seaweed` |
| 65 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pretzel.jpg" alt="pretzel" width="96"><br>`pretzel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/peanut.jpg" alt="peanut" width="96"><br>`peanut` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/clothes.jpg" alt="clothes" width="96"><br>`clothes` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/warthog.jpg" alt="warthog" width="96"><br>`warthog` |
| 66 | 5 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/snake.jpg" alt="snake" width="96"><br>`snake` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hawk.jpg" alt="hawk" width="96"><br>`hawk` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bee.jpg" alt="bee" width="96"><br>`bee` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/highchair.jpg" alt="highchair" width="96"><br>`highchair` |
| 67 | 5 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/guitar.jpg" alt="guitar" width="96"><br>`guitar` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/violin.jpg" alt="violin" width="96"><br>`violin` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/closet.jpg" alt="closet" width="96"><br>`closet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pearl.jpg" alt="pearl" width="96"><br>`pearl` |
| 68 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/meatball.jpg" alt="meatball" width="96"><br>`meatball` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/muffin.jpg" alt="muffin" width="96"><br>`muffin` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wineglass.jpg" alt="wineglass" width="96"><br>`wineglass` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/doormat.jpg" alt="doormat" width="96"><br>`doormat` |
| 69 | 6 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/dolphin.jpg" alt="dolphin" width="96"><br>`dolphin` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shark.jpg" alt="shark" width="96"><br>`shark` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/flamingo.jpg" alt="flamingo" width="96"><br>`flamingo` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/knitting.jpg" alt="knitting" width="96"><br>`knitting` |
| 70 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ambulance.jpg" alt="ambulance" width="96"><br>`ambulance` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/truck.jpg" alt="truck" width="96"><br>`truck` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/locket.jpg" alt="locket" width="96"><br>`locket` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/backscratcher.jpg" alt="backscratcher" width="96"><br>`backscratcher` |
| 71 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stool.jpg" alt="stool" width="96"><br>`stool` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/table.jpg" alt="table" width="96"><br>`table` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tablecloth.jpg" alt="tablecloth" width="96"><br>`tablecloth` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/melon.jpg" alt="melon" width="96"><br>`melon` |
| 72 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/blueberry.jpg" alt="blueberry" width="96"><br>`blueberry` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/berry.jpg" alt="berry" width="96"><br>`berry` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cabinet.jpg" alt="cabinet" width="96"><br>`cabinet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/goldfish.jpg" alt="goldfish" width="96"><br>`goldfish` |
| 73 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/fence.jpg" alt="fence" width="96"><br>`fence` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/railing.jpg" alt="railing" width="96"><br>`railing` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crane.jpg" alt="crane" width="96"><br>`crane` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cashew.jpg" alt="cashew" width="96"><br>`cashew` |
| 74 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/timer.jpg" alt="timer" width="96"><br>`timer` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tag.jpg" alt="tag" width="96"><br>`tag` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/motorcycle.jpg" alt="motorcycle" width="96"><br>`motorcycle` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/starfish.jpg" alt="starfish" width="96"><br>`starfish` |
| 75 | 6 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/swan.jpg" alt="swan" width="96"><br>`swan` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/goose.jpg" alt="goose" width="96"><br>`goose` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/yak.jpg" alt="yak" width="96"><br>`yak` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mascara.jpg" alt="mascara" width="96"><br>`mascara` |
| 76 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/brick.jpg" alt="brick" width="96"><br>`brick` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/block.jpg" alt="block" width="96"><br>`block` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cranberry.jpg" alt="cranberry" width="96"><br>`cranberry` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hairpin.jpg" alt="hairpin" width="96"><br>`hairpin` |
| 77 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mushroom.jpg" alt="mushroom" width="96"><br>`mushroom` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/fungus.jpg" alt="fungus" width="96"><br>`fungus` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tortilla.jpg" alt="tortilla" width="96"><br>`tortilla` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/overalls.jpg" alt="overalls" width="96"><br>`overalls` |
| 78 | 6 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/globe.jpg" alt="globe" width="96"><br>`globe` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bubble.jpg" alt="bubble" width="96"><br>`bubble` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mitten.jpg" alt="mitten" width="96"><br>`mitten` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stockings.jpg" alt="stockings" width="96"><br>`stockings` |
| 79 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/scooter.jpg" alt="scooter" width="96"><br>`scooter` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/coop.jpg" alt="coop" width="96"><br>`coop` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/trombone.jpg" alt="trombone" width="96"><br>`trombone` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/windowsill.jpg" alt="windowsill" width="96"><br>`windowsill` |
| 80 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/toaster.jpg" alt="toaster" width="96"><br>`toaster` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/heater.jpg" alt="heater" width="96"><br>`heater` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/jewel.jpg" alt="jewel" width="96"><br>`jewel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wreath.jpg" alt="wreath" width="96"><br>`wreath` |
| 81 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/marble.jpg" alt="marble" width="96"><br>`marble` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/crystal.jpg" alt="crystal" width="96"><br>`crystal` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shield.jpg" alt="shield" width="96"><br>`shield` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/pheasant.jpg" alt="pheasant" width="96"><br>`pheasant` |
| 82 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/extinguisher.jpg" alt="extinguisher" width="96"><br>`extinguisher` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/lighter.jpg" alt="lighter" width="96"><br>`lighter` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bead.jpg" alt="bead" width="96"><br>`bead` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ravioli.jpg" alt="ravioli" width="96"><br>`ravioli` |
| 83 | 7 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/fireplace.jpg" alt="fireplace" width="96"><br>`fireplace` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/altar.jpg" alt="altar" width="96"><br>`altar` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/reel.jpg" alt="reel" width="96"><br>`reel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ratchet.jpg" alt="ratchet" width="96"><br>`ratchet` |
| 84 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/headrest.jpg" alt="headrest" width="96"><br>`headrest` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/footrest.jpg" alt="footrest" width="96"><br>`footrest` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/birdbath.jpg" alt="birdbath" width="96"><br>`birdbath` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/origami.jpg" alt="origami" width="96"><br>`origami` |
| 85 | 8 | animal | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/scorpion.jpg" alt="scorpion" width="96"><br>`scorpion` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cobra.jpg" alt="cobra" width="96"><br>`cobra` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bison.jpg" alt="bison" width="96"><br>`bison` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/denture.jpg" alt="denture" width="96"><br>`denture` |
| 86 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/torch.jpg" alt="torch" width="96"><br>`torch` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/blowtorch.jpg" alt="blowtorch" width="96"><br>`blowtorch` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/leash.jpg" alt="leash" width="96"><br>`leash` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/gravestone.jpg" alt="gravestone" width="96"><br>`gravestone` |
| 87 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/makeup.jpg" alt="makeup" width="96"><br>`makeup` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/peg.jpg" alt="peg" width="96"><br>`peg` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bracket.jpg" alt="bracket" width="96"><br>`bracket` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sprouts.jpg" alt="sprouts" width="96"><br>`sprouts` |
| 88 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/banjo.jpg" alt="banjo" width="96"><br>`banjo` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/basil.jpg" alt="basil" width="96"><br>`basil` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/visor.jpg" alt="visor" width="96"><br>`visor` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/cockroach.jpg" alt="cockroach" width="96"><br>`cockroach` |
| 89 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/compass.jpg" alt="compass" width="96"><br>`compass` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bomb.jpg" alt="bomb" width="96"><br>`bomb` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mousetrap.jpg" alt="mousetrap" width="96"><br>`mousetrap` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/forklift.jpg" alt="forklift" width="96"><br>`forklift` |
| 90 | 8 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/funnel.jpg" alt="funnel" width="96"><br>`funnel` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/spout.jpg" alt="spout" width="96"><br>`spout` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stirrup.jpg" alt="stirrup" width="96"><br>`stirrup` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/valve.jpg" alt="valve" width="96"><br>`valve` |
| 91 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sling.jpg" alt="sling" width="96"><br>`sling` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/brace.jpg" alt="brace" width="96"><br>`brace` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/stamp.jpg" alt="stamp" width="96"><br>`stamp` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/penlight.jpg" alt="penlight" width="96"><br>`penlight` |
| 92 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/whisk.jpg" alt="whisk" width="96"><br>`whisk` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/whip.jpg" alt="whip" width="96"><br>`whip` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/grinder.jpg" alt="grinder" width="96"><br>`grinder` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/kaleidoscope.jpg" alt="kaleidoscope" width="96"><br>`kaleidoscope` |
| 93 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shredder.jpg" alt="shredder" width="96"><br>`shredder` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/sweeper.jpg" alt="sweeper" width="96"><br>`sweeper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/shears.jpg" alt="shears" width="96"><br>`shears` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/trunk.jpg" alt="trunk" width="96"><br>`trunk` |
| 94 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/saxophone.jpg" alt="saxophone" width="96"><br>`saxophone` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/clarinet.jpg" alt="clarinet" width="96"><br>`clarinet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/firewood.jpg" alt="firewood" width="96"><br>`firewood` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/hail.jpg" alt="hail" width="96"><br>`hail` |
| 95 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/keyboard.jpg" alt="keyboard" width="96"><br>`keyboard` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/tablet.jpg" alt="tablet" width="96"><br>`tablet` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/urinal.jpg" alt="urinal" width="96"><br>`urinal` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/gazelle.jpg" alt="gazelle" width="96"><br>`gazelle` |
| 96 | 9 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/spur.jpg" alt="spur" width="96"><br>`spur` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/spear.jpg" alt="spear" width="96"><br>`spear` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/wheelbarrow.jpg" alt="wheelbarrow" width="96"><br>`wheelbarrow` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/videocassette.jpg" alt="videocassette" width="96"><br>`videocassette` |
| 97 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/gong.jpg" alt="gong" width="96"><br>`gong` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/gem.jpg" alt="gem" width="96"><br>`gem` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/grater.jpg" alt="grater" width="96"><br>`grater` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/leopard.jpg" alt="leopard" width="96"><br>`leopard` |
| 98 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/boomerang.jpg" alt="boomerang" width="96"><br>`boomerang` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/bumper.jpg" alt="bumper" width="96"><br>`bumper` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/charger.jpg" alt="charger" width="96"><br>`charger` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/incense.jpg" alt="incense" width="96"><br>`incense` |
| 99 | 10 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/ladle.jpg" alt="ladle" width="96"><br>`ladle` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/penholder.jpg" alt="penholder" width="96"><br>`penholder` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/mold.jpg" alt="mold" width="96"><br>`mold` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/organ.jpg" alt="organ" width="96"><br>`organ` |
| 100 | 11 | object | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/flask.jpg" alt="flask" width="96"><br>`flask` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/filter.jpg" alt="filter" width="96"><br>`filter` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/beanie.jpg" alt="beanie" width="96"><br>`beanie` | <img src="https://raw.githubusercontent.com/langcog/levante-bench/v2/docs/assets/synth_vocab_thumbs/poppy.jpg" alt="poppy" width="96"><br>`poppy` |

*100 items, 400 unique terms.*

---

## Appendix B. Relation to the LEVANTE core-tasks manuscript

We did **not** use a paper supplement. The December 2025 LEVANTE core-tasks manuscript (`references/LEVANTE_manuscript_dec16.pdf`) has no supplement attached, and the generator never read SI tables.

That paper’s Vocabulary section (main text) describes the **child** 4-choice picture-vocabulary task: a 108-item THINGS+ core with close / far / unrelated distractors (Long et al., 2025), then extra DAIVT and hand-built items to reach 170. Data and code are pointed at [`levante-framework/levante-pilots`](https://github.com/levante-framework/levante-pilots), not an SI file.

This synthetic VLM bank is a separate construction. It followed the same main-text design (4-choice, ~170 items, similarity-tiered distractors, THINGS/THINGSplus) but:

- took the original 108 labels from the DevBench GitHub manifest and **excluded** them so the banks do not overlap
- added Kuperman age-of-acquisition norms (not in that vocab paragraph)
- did **not** use DAIVT or the pilots item list
- generated new Imagen photographs

In short: we used the same public resources and design idea as the manuscript main text, not a supplement.
