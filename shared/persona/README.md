# Child-age persona (shared with levante-qa)

This folder holds the **single source of truth** for the child-age persona used to
make a VLM answer *as a typical child of a target age would*, calibrated to real
LEVANTE accuracy-by-age data. The same two artifacts are consumed by both repos:

| File | Purpose |
|------|---------|
| `age_task_accuracy.json` | Per-task mean accuracy by age (`{task_id: {ageYears: meanAccuracy}}`) — the empirical "difficulty" target (always used when persona is on). |
| `age_task_ability.json` | Per-task mean IRT ability θ by age (`{task_id: {ageYears: {theta, n}}}`) — optional add-on when `QA_PERSONA_ABILITY=irt`. |
| `age_task_accuracy_by_country.json` | Same as accuracy, stratified by country (`de`/`co`/`ca` from pilot sites). Used when `QA_SIM_COUNTRY` / `QA_PERSONA_COUNTRY` is set. |
| `age_task_ability_by_country.json` | Same as ability, stratified by country. |
| `persona_template.txt` | The persona prompt wording, with `{age_phrase}`, `{difficulty_block}`, and `{ability_block}` placeholders. |

- **Canonical copies live here** (`levante-bench/shared/persona/`), because the
  child trial data (`data/responses/v2/trials.csv` by default; use `--version v1`
  for the frozen April tree) and the generator live in this repo.
- **levante-qa vendors a copy** at `cypress/support/persona/` and pulls updates
  with `pnpm persona:sync` (from levante-qa). Re-sync whenever the profile or
  template changes here so both repos render identical persona prompts.

## Regenerating the profile

```bash
python scripts/build_age_accuracy_profile.py              # default: data/responses/v2
python scripts/build_age_ability_profile.py
python scripts/build_age_accuracy_profile.py --version v1 # frozen April tree
# options: --trials <path> --out <path> --out-by-country <path>
#          --min-samples 30 (accuracy) / --min-runs 15 (ability)
```

**Accuracy profile:** bins trials by `task_id` and rounded age (whole years), drops cells with fewer
than `--min-samples` trials, and writes `age_task_accuracy.json` plus
`age_task_accuracy_by_country.json` (site → `de`/`co`/`ca` via `scripts/site_country.py`).

**Ability profile:** joins `trials.csv` (age + site per `run_id`) with `irt_models/*_ability_scores.csv`,
bins by rounded age, and writes `age_task_ability.json` plus `age_task_ability_by_country.json`
(tasks that have IRT ability only).

Included tasks for accuracy:
`egma-math, matrix-reasoning, mental-rotation, theory-of-mind, trog, vocab,
hearts-and-flowers, same-different-selection, memory-game`.

After regenerating, in levante-qa: `LEVANTE_BENCH_DIR=/path/to/levante-bench pnpm persona:sync`.

Check Child Twins grid coverage (flags country cells that would fall back to global):

```bash
python scripts/validate_country_age_profiles.py
python scripts/validate_country_age_profiles.py --ages 6,8,10 --countries de,co,ca
```

---

## TODO: wire the persona into the Gemini model (not yet implemented in this repo)

> levante-qa already implements the persona end-to-end (it prepends the preamble
> to each task's system prompt inside its `askVLM` task). The steps below add the
> equivalent to levante-bench's Gemini model. Keep the wording and difficulty
> thresholds **identical** to levante-qa's `cypress/support/persona/childPersona.ts`.

### 1. Python persona builder

Create `src/levante_bench/prompts/child_persona.py`:

```python
def make_child_persona_prompt(
    age_years: float,
    age_months: int = 0,
    task_id: str | None = None,
    profile: dict | None = None,
) -> str:
    ...
```

- Load `shared/persona/age_task_accuracy.json` + `shared/persona/persona_template.txt`.
- Difficulty thresholds **must match** levante-qa: `> 0.75` easy / `> 0.45` moderate / else hard.
- Nearest-age lookup, clamped to the profile's age range.
- Age phrase: `"{y}-year-old"`, or `"{y}-year-{m}-month-old"` when months > 0.
- `difficulty_block` (only when a task profile exists for the age):
  `"\n\nFor a child this age, this task is typically {label} (about {pct}% of items answered correctly by children this age)."`
- bench `task_id`s are already canonical (`egma-math`, `theory-of-mind`, ...), so no slug map is needed.

### 2. Wire system instruction into `GeminiProModel`

In `src/levante_bench/models/gemini.py`:

- Add to `__init__`: `child_age_years: float | None = None`, `child_age_months: int = 0`,
  `system_instruction: str | None = None` (store them).
- Add a helper:

```python
def _build_system_instruction(self, task_id: str | None = None) -> str | None:
    if self.system_instruction:
        return self.system_instruction
    if self.child_age_years is not None:
        from levante_bench.prompts.child_persona import make_child_persona_prompt
        return make_child_persona_prompt(self.child_age_years, self.child_age_months, task_id)
    return None
```

- In `generate()`, after building `payload`, inject it (the REST API at
  `API_BASE` supports the top-level field):

```python
sys_instr = self._build_system_instruction(task_id)  # thread task_id in if available
if sys_instr:
    payload["system_instruction"] = {"parts": [{"text": sys_instr}]}
```

  If the eval loop can pass the current `task_id` into `generate()`, do so for
  the per-task difficulty hint; otherwise the global age persona still applies
  (no per-task hint).

### 3. Model config

Add `configs/models/gemini_pro_child_6yr.yaml`, mirroring `gemini25_pro.yaml` plus:

```yaml
child_age_years: 6.0
```

Confirm the registry/loader forwards unknown YAML keys to `__init__` (the
`@register` decorator on `GeminiProModel` already does).

### 4. Run + analyze

```bash
levante-bench run-eval --task trog --model gemini_pro_child_6yr
```

Then feed outputs through the existing age-equivalency pipeline
(`estimate_model_age_equivalency.py` + the KL comparison) to check whether the
simulated child's age-equivalency matches the target age. Use
`prompt_robustness_sweep.py` to sweep ages / prompt phrasings.

### Acceptance

- For a given age + task, `make_child_persona_prompt` returns text identical
  (modulo whitespace) to levante-qa's `makeChildPersonaPrompt`.
- A non-persona model config is unchanged.
- The `gemini_pro_child_6yr` config injects the persona into the Gemini payload.

### Notes

- The persona is wording-based ("answer as a child would"), so models won't
  perfectly emulate a developmental level — the KL / age-equivalency analysis is
  what quantifies the fit.
- The same pattern extends to other adapters (GPT, etc.): add `system_instruction`
  support and the `child_age_*` params, reusing `make_child_persona_prompt`.
