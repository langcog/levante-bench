"""Run EN vs DE TROG comparison for SmolVLM2 with checkpoints."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from levante_bench.config import get_task_def, load_model_config
from levante_bench.models import get_model_class
from levante_bench.tasks import get_task_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="hackathon")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        default="results/multi-language/trog_en_vs_de_30.json",
    )
    args = parser.parse_args()

    data_root = Path("data")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        report["resumed_at"] = datetime.now(UTC).isoformat()
    else:
        report = {
            "started_at": datetime.now(UTC).isoformat(),
            "task": "trog",
            "version": args.version,
            "model": "smolvlm2",
            "n_requested": args.n,
            "languages": {},
            "results_by_language": {"en": {}, "de": {}},
            "items": [],
            "status": "running",
        }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        model_cfg = OmegaConf.to_container(load_model_config("smolvlm2"), resolve=True)
        model_cls = get_model_class("smolvlm2")
        model = model_cls(model_name=model_cfg["hf_name"], device=args.device)
        model.use_json_format = bool(model_cfg.get("use_json_format", True))
        model.load()

        per_lang_results: dict[str, dict[str, dict]] = report.setdefault(
            "results_by_language", {"en": {}, "de": {}}
        )
        item_uids: list[str] | None = None

        for lang in ["en", "de"]:
            task_def = get_task_def(
                "trog",
                args.version,
                data_root=data_root,
                task_overrides={"prompt_language": lang},
            )
            ds_cls = get_task_dataset("trog")
            ds = ds_cls(task_def=task_def, version=args.version, data_root=data_root)
            n = min(args.n, len(ds))
            if item_uids is None:
                item_uids = [str(ds[i]["item_uid"]) for i in range(n)]

            lang_rows = per_lang_results.setdefault(lang, {})
            correct = sum(1 for r in lang_rows.values() if bool(r.get("is_correct")))
            completed = len(lang_rows)
            report["languages"][lang] = {
                "correct": correct,
                "total": n,
                "accuracy": (correct / completed) if completed else 0.0,
                "completed": completed,
            }

            for i in range(n):
                trial = ds[i]
                uid = str(trial["item_uid"])
                if uid in lang_rows:
                    continue
                trial["max_new_tokens"] = int(model_cfg.get("max_new_tokens", 64))
                result = model.evaluate_trial(trial)
                ok = bool(result.get("is_correct"))
                correct += int(ok)
                lang_rows[uid] = {
                    "prompt": trial.get("prompt", ""),
                    "pred_label": result.get("pred_label"),
                    "correct_label": result.get("correct_label"),
                    "is_correct": ok,
                }
                report["languages"][lang]["correct"] = correct
                report["languages"][lang]["completed"] = len(lang_rows)
                report["languages"][lang]["accuracy"] = correct / len(lang_rows)
                report["status"] = "running"
                out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"lang={lang} {i+1}/{n} uid={uid} pred={result.get('pred_label')} "
                    f"gold={result.get('correct_label')} ok={ok}",
                    flush=True,
                )

            report["languages"][lang]["accuracy"] = correct / n if n else 0.0
            print(f"LANG={lang} ACC={report['languages'][lang]['accuracy']:.4f} ({correct}/{n})", flush=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        report["items"] = []
        for uid in item_uids or []:
            en_row = per_lang_results["en"][uid]
            de_row = per_lang_results["de"][uid]
            report["items"].append(
                {
                    "item_uid": uid,
                    "en_is_correct": en_row["is_correct"],
                    "de_is_correct": de_row["is_correct"],
                    "en_pred_label": en_row["pred_label"],
                    "de_pred_label": de_row["pred_label"],
                    "gold_label": en_row["correct_label"],
                    "changed_correctness": en_row["is_correct"] != de_row["is_correct"],
                    "changed_prediction": en_row["pred_label"] != de_row["pred_label"],
                }
            )

        report["status"] = "completed"
        report["finished_at"] = datetime.now(UTC).isoformat()
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE {out_path}", flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["finished_at"] = datetime.now(UTC).isoformat()
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"FAILED: {exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
