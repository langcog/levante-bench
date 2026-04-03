# Environment Split: Benchmark vs Aquila

Use separate virtual environments to avoid dependency conflicts between:

- **Core benchmark models** (`smolvlm2`, `qwen35`, `internvl35`, `tinyllava`)
- **Aquila intermediate checkpoints** (`aquila_vl` via LLaVA-NeXT)

## 1) Core benchmark environment

- Path: `.venv`
- Key stack: `transformers>=4.57`, `tokenizers 0.22+`

Activate:

```bash
source .venv/bin/activate
```

Sanity check:

```bash
python - <<'PY'
import transformers
from transformers import AutoModelForImageTextToText
print(transformers.__version__)
print("AutoModelForImageTextToText OK")
PY
```

## 2) Aquila / LLaVA environment

- Path: `.venv-aquila`
- Key stack: `llava`, `transformers==4.40.0.dev0` (LLaVA-compatible), `tokenizers~=0.15`

Activate:

```bash
source .venv-aquila/bin/activate
```

Run Aquila experiment:

```bash
PYTHONPATH=src python -m levante_bench.cli experiment=configs/experiment_aquila_vl.yaml device=cuda
```

## Notes

- Do **not** run standard benchmark models inside `.venv-aquila`.
- Do **not** run `aquila_vl` inside `.venv`.
- If you reinstall dependencies in one env, re-check with `python -m levante_bench.cli list-models` and a one-trial smoke test.
