"""
backfill_wandb.py
=================
Ingest completed training runs (stored as text files) into wandb so they
appear in the same project as live runs for comparison.

Usage
-----
    python backfill_wandb.py
    python backfill_wandb.py --dry_run        # print what would be logged, no API calls
    python backfill_wandb.py --project NAME   # override project name
"""

import argparse
import re
from pathlib import Path

import wandb

WANDB_PROJECT = "mfcad-feature-classification"
WANDB_ENTITY  = None

RUNS_DIR = Path("./runs")

# ── Run registry ─────────────────────────────────────────────────────────────
# Each entry maps to one wandb run. The 'summary_path' and 'report_path' point
# to the text files already on disk. config_overrides fills in fields not
# present in the text files (e.g. feature_set, which was added later).

HISTORICAL_RUNS = [
    # ── run_20260410_both / gcn  (V_1 only, no weighting) ────────────────────
    {
        "run_name":      "gcn_run_20260410_both",
        "run_group":     "run_20260410_both",
        "summary_path":  RUNS_DIR / "run_20260410_both/gcn/summary.txt",
        "report_path":   RUNS_DIR / "run_20260410_both/gcn/classification_report.txt",
        "tags":          ["arch:gcn", "feature_set:v1_only",
                          "weighted_loss:no", "backfilled"],
        "config_overrides": {
            "feature_set":   "v1_only",
            "arch":          "gcn",
            "weighted_loss": False,
        },
    },
    # ── run_20260410_both / graphsage  (V_1 only, no weighting) ──────────────
    {
        "run_name":      "graphsage_run_20260410_both",
        "run_group":     "run_20260410_both",
        "summary_path":  RUNS_DIR / "run_20260410_both/graphsage/summary.txt",
        "report_path":   RUNS_DIR / "run_20260410_both/graphsage/classification_report.txt",
        "tags":          ["arch:graphsage", "feature_set:v1_only",
                          "weighted_loss:no", "backfilled"],
        "config_overrides": {
            "feature_set":   "v1_only",
            "arch":          "graphsage",
            "weighted_loss": False,
        },
    },
    # ── run_20260410_093940_both / gcn  (hierarchical V1+V2, weighted) ────────
    {
        "run_name":      "gcn_run_20260410_093940_both",
        "run_group":     "run_20260410_093940_both",
        "summary_path":  RUNS_DIR / "run_20260410_093940_both/gcn/summary.txt",
        "report_path":   RUNS_DIR / "run_20260410_093940_both/gcn/classification_report.txt",
        "tags":          ["arch:gcn", "feature_set:hierarchical_v1_v2",
                          "weighted_loss:yes", "backfilled"],
        "config_overrides": {
            "feature_set":   "hierarchical_v1_v2",
            "arch":          "gcn",
            "weighted_loss": True,
        },
    },
    # ── run_20260410_093940_both / graphsage  (hierarchical V1+V2, weighted) ──
    {
        "run_name":      "graphsage_run_20260410_093940_both",
        "run_group":     "run_20260410_093940_both",
        "summary_path":  RUNS_DIR / "run_20260410_093940_both/graphsage/summary.txt",
        "report_path":   RUNS_DIR / "run_20260410_093940_both/graphsage/classification_report.txt",
        "tags":          ["arch:graphsage", "feature_set:hierarchical_v1_v2",
                          "weighted_loss:yes", "backfilled"],
        "config_overrides": {
            "feature_set":   "hierarchical_v1_v2",
            "arch":          "graphsage",
            "weighted_loss": True,
        },
    },
]

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_summary(path: Path) -> dict:
    """
    Parse a summary.txt file produced by save_run_summary().
    Returns dict with keys: config (dict), results (dict).
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {"config": {}, "results": {}}

    # Config block — lines like "  hidden_dim           128"
    cfg_block = re.search(r"-- Config -+\n(.*?)(?=\n-- )", text, re.DOTALL)
    if cfg_block:
        for line in cfg_block.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                k, v = parts
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                out["config"][k] = v

    # Results
    for key, pat in [
        ("test_acc",    r"Test accuracy\s+([\d.]+)"),
        ("test_f1",     r"Macro F1\s+([\d.]+)"),
        ("best_val_f1", r"Best val F1\s+([\d.]+)"),
        ("epochs_run",  r"Epochs run\s+(\d+)"),
    ]:
        m = re.search(pat, text)
        if m:
            out["results"][key] = float(m.group(1))

    m = re.search(r"Early stopped\s+(\w+)", text)
    if m:
        out["results"]["stopped_early"] = m.group(1).lower() == "yes"

    m = re.search(r"Training time\s+(.+)", text)
    if m:
        out["results"]["training_time_str"] = m.group(1).strip()

    return out


def parse_report(path: Path) -> tuple:
    """
    Parse classification_report.txt.
    Returns (header_dict, rows) where rows is list of per-class dicts.
    """
    if not path.exists():
        return {}, []
    text = path.read_text(encoding="utf-8", errors="replace")

    header = {}
    m = re.match(
        r"(\w+)\s+[—-]+\s+acc=([\d.]+)\s+macro-F1=([\d.]+)\s+weighted_loss=(\w+)",
        text.strip(),
    )
    if m:
        header = {
            "arch":          m.group(1).lower(),
            "test_acc":      float(m.group(2)),
            "test_f1":       float(m.group(3)),
            "weighted_loss": m.group(4) == "yes",
        }

    rows = []
    # Match lines with 4 numeric columns (precision recall f1 support)
    pat = re.compile(r"^\s{2,}(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$")
    for line in text.splitlines():
        mo = pat.match(line)
        if mo:
            rows.append({
                "class_name": mo.group(1).strip(),
                "precision":  float(mo.group(2)),
                "recall":     float(mo.group(3)),
                "f1":         float(mo.group(4)),
                "support":    int(mo.group(5)),
            })
    return header, rows


def parse_training_time(s: str) -> int:
    """Convert '75m 00s' to seconds."""
    m = re.match(r"(?:(\d+)m\s*)?(\d+)s", s or "")
    if m:
        return int(m.group(1) or 0) * 60 + int(m.group(2))
    return 0


# ── Backfill ─────────────────────────────────────────────────────────────────

def backfill_run(run_def: dict, dry_run: bool, project: str, entity):
    summary = parse_summary(run_def["summary_path"])
    header, class_rows = parse_report(run_def["report_path"])

    # Merge config: defaults → summary.config → header → explicit overrides
    config = {
        "hidden_dim":    128,
        "num_layers":    3,
        "batch_size":    32,
        "lr":            0.001,
        "dropout":       0.5,
        "patience":      15,
        "num_classes":   25,
    }
    config.update(summary.get("config", {}))
    if header.get("weighted_loss") is not None:
        config["weighted_loss"] = header["weighted_loss"]
    config.update(run_def["config_overrides"])

    results = summary.get("results", {})
    test_acc = results.get("test_acc") or header.get("test_acc")
    test_f1  = results.get("test_f1")  or header.get("test_f1")

    if dry_run:
        print(f"[DRY RUN] {run_def['run_name']}")
        print(f"  test_acc={test_acc}  test_f1={test_f1}")
        print(f"  config={config}")
        print(f"  tags={run_def['tags']}")
        print(f"  class_rows={len(class_rows)}")
        return

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_def["run_name"],
        group=run_def["run_group"],
        tags=run_def["tags"],
        config=config,
        reinit="finish_previous",
    )

    run.log({"test/acc": test_acc, "test/f1": test_f1,
             "best_val_f1": results.get("best_val_f1", test_f1)})

    run.summary["test/acc"]      = test_acc
    run.summary["test/f1"]       = test_f1
    run.summary["best_val_f1"]   = results.get("best_val_f1", test_f1)
    run.summary["epochs_run"]    = results.get("epochs_run")
    run.summary["stopped_early"] = results.get("stopped_early")
    if "training_time_str" in results:
        run.summary["training_time_sec"] = parse_training_time(
            results["training_time_str"]
        )

    # Per-class F1 table
    class_data = [
        [r["class_name"], r["precision"], r["recall"], r["f1"], r["support"]]
        for r in class_rows
        if r["class_name"] not in ("accuracy", "macro avg", "weighted avg")
    ]
    if class_data:
        table = wandb.Table(
            columns=["class_name", "precision", "recall", "f1", "support"],
            data=class_data,
        )
        run.log({"per_class_f1_table": table})

    # Attach text files as artifact
    artifact = wandb.Artifact(
        name=f"run_files_{run_def['run_name']}",
        type="run_outputs",
    )
    for key in ("summary_path", "report_path"):
        p = run_def[key]
        if Path(p).exists():
            artifact.add_file(str(p), name=Path(p).name)
    run.log_artifact(artifact)

    run.finish()
    print(f"  [done] {run_def['run_name']}  (test_f1={test_f1:.4f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backfill historical runs into wandb")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be logged without calling wandb API")
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--entity",  default=WANDB_ENTITY)
    args = parser.parse_args()

    print(f"Project: {args.project}  |  Dry run: {args.dry_run}")
    print(f"Runs to backfill: {len(HISTORICAL_RUNS)}\n")

    for run_def in HISTORICAL_RUNS:
        backfill_run(run_def, dry_run=args.dry_run,
                     project=args.project, entity=args.entity)

    print("\nDone.")


if __name__ == "__main__":
    main()
