# Prompting and answer parsing

Templates and parser: `src/levante_bench/models/base.py`.

## Two paths, picked by `use_json_format` in each model YAML

**JSON (default):**

```
<task prompt>

Respond in JSON format: {"answer": "<one of A, B, C, or D>", "reason": "<short reason>"}
```

**Simple** (`use_json_format: false`, for tiny models that echo JSON
placeholders):

```
<task prompt>

Answer with only the letter A, B, C, or D.
```

Both paths use the same parser: JSON outputs are handled by `json-repair`,
single-letter replies by exact/prefix-label fallbacks.

## Where the letters come from

Each dataset sets `trial["option_labels"]` to the full set of valid letters
for that trial (e.g. `["A", "B"]` for mental-rotation, `["A", "B", "C", "D"]`
for vocab). The correct answer is stored separately in `trial["correct_label"]`
and is always one of `option_labels`, but the prompt never reveals which —
the model only sees the full enumeration. At prompt-build time,
`_format_labels_prose` renders the list as `A or B` / `A, B, C, or D` and
substitutes it into the template, so the instruction always matches the
trial's actual option count.

## Per-model config

Set `use_json_format` in the per-size YAML (`configs/models/<name>.yaml`).
No YAML inheritance — each file stands alone, so one family can mix capable
(JSON) and tiny (simple) variants.
