# Synthetic Vocab Input Provenance

- `things_meta.csv` is derived from THINGS/THINGSplus OSF project `jum2f`: `02_object-level/_concepts-metadata_things.tsv` and `02_object-level/_property-ratings.tsv`. `child_safe` is a local conservative heuristic; `nameability` uses `image-label_nameability_mean`; `animacy` normalizes `property_lives_mean` from the 1-7 THINGSplus scale to 0-1.
- `aoa_kuperman.csv` is derived from the Hugging Face mirror `StephanAkkerman/English-Age-of-Acquisition`, file `en.aoa.csv`; `aoa` uses `AoA_Kup_lem` with `AoA_Kup` fallback.
- `original_108.csv` is derived from the DevBench Visual Vocabulary manifest at `https://raw.githubusercontent.com/alvinwmtan/dev-bench/master/assets/lex-viz_vocab/manifest.csv`. That manifest currently has 119 rows; this file uses the first 108 labels to match the original Long et al. VV target count.
