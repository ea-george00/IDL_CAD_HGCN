"""
reproduce_results.py
--------------------
Loads all six trained checkpoints and regenerates the key figures and
metrics from the paper without retraining.

Usage:
    python reproduce_results.py --data_dir /path/to/MFCAD++_dataset
    python reproduce_results.py --data_dir /path/to/MFCAD++_dataset --uv_cache_dir ./uv_cache

Arguments:
    --data_dir      Path to the MFCAD++ dataset folder or zip file.
    --uv_cache_dir  Path to the directory containing uv_cache/{train,val,test}.h5
                    (produced by extract_uv_features.py).
                    Required to evaluate GCN UV-net and GraphSAGE UV-net.
                    V1 and V1+V2 runs do not require this.
    --out_dir       Where to save output figures (default: ./paper_figures)
    --no_uvnet      Skip UV-net runs (useful if uv_cache is not available)

Outputs (saved to --out_dir):
    table1.csv                  All 6 runs: acc, macro-F1, precision, recall
    confusion_matrix.png        Normalized confusion matrix (GraphSAGE UV-net)
    per_class_f1.png            Per-class F1 bar chart (GraphSAGE UV-net)
    classification_report.txt   Full sklearn report (GraphSAGE UV-net)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from torch_geometric.loader import DataLoader

from train_mfcad import (BATCH_SIZE, DROPOUT, HIDDEN_DIM, NUM_LAYERS,
                         HierarchicalGCN, HierarchicalGraphSAGE, evaluate,
                         h5_to_pyg, load_label_map)

REPO_ROOT = Path(__file__).parent

EXPERIMENTS = [
    {
        "name":    "GCN V1 flat",
        "arch":    "gcn",
        "face_in": 5,
        "use_v2":  False,
        "uv":      False,
        "ckpt":    "checkpoints/gcn_v1_flat/gcn_mfcad.pt",
        "wandb":   "fw8zrpy4",
    },
    {
        "name":    "GCN V1+V2 hierarchical",
        "arch":    "gcn",
        "face_in": 5,
        "use_v2":  True,
        "uv":      False,
        "ckpt":    "checkpoints/gcn_v1v2_hierarchical/gcn_mfcad.pt",
        "wandb":   "89g2g8my",
    },
    {
        "name":    "GCN UV-net",
        "arch":    "gcn",
        "face_in": 12,
        "use_v2":  True,
        "uv":      True,
        "ckpt":    "checkpoints/gcn_uvnet/gcn_mfcad.pt",
        "wandb":   "pkgqwb7b",
    },
    {
        "name":    "GraphSAGE V1 flat",
        "arch":    "graphsage",
        "face_in": 5,
        "use_v2":  False,
        "uv":      False,
        "ckpt":    "checkpoints/graphsage_v1_flat/graphsage_mfcad.pt",
        "wandb":   "024839ck",
    },
    {
        "name":    "GraphSAGE V1+V2 hierarchical",
        "arch":    "graphsage",
        "face_in": 5,
        "use_v2":  True,
        "uv":      False,
        "ckpt":    "checkpoints/graphsage_v1v2_hierarchical/graphsage_mfcad.pt",
        "wandb":   "mfl4y8ti",
    },
    {
        "name":    "GraphSAGE UV-net  [best]",
        "arch":    "graphsage",
        "face_in": 12,
        "use_v2":  True,
        "uv":      True,
        "ckpt":    "checkpoints/graphsage_uvnet/graphsage_mfcad.pt",
        "wandb":   "yfm13ito",
    },
]


def load_model(exp: dict, num_classes: int, device):
    ModelClass = HierarchicalGCN if exp["arch"] == "gcn" else HierarchicalGraphSAGE
    model = ModelClass(
        face_in=exp["face_in"],
        facet_in=4,
        hidden=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        use_v2=exp["use_v2"],
    ).to(device)
    ckpt = REPO_ROOT / exp["ckpt"]
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def plot_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45, values_format=".2f",
              cmap="Blues")
    ax.set_title("Normalized Confusion Matrix — GraphSAGE UV-net", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_class_f1(labels, preds, class_names, out_path):
    per_class = f1_score(labels, preds, average=None, zero_division=0)
    order = np.argsort(per_class)
    sorted_names = [class_names[i] for i in order]
    sorted_f1 = per_class[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(sorted_names)), sorted_f1, color="#4C72B0")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-class F1 — GraphSAGE UV-net", fontsize=13)
    ax.axvline(sorted_f1.mean(), color="red", linestyle="--", linewidth=1,
               label=f"Macro avg = {sorted_f1.mean():.3f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--uv_cache_dir", default=None)
    parser.add_argument("--out_dir",     default="paper_figures")
    parser.add_argument("--no_uvnet",    action="store_true",
                        help="Skip UV-net experiments (if uv_cache not available)")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    uv_cache    = Path(args.uv_cache_dir) if args.uv_cache_dir else None
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    label_map   = load_label_map(data_dir)
    num_classes = len(label_map)
    class_names = [label_map[i] for i in range(num_classes)]

    # Load test graphs — two variants: flat (face_in=5) and UV-net (face_in=12)
    print("Loading test set (flat features)...")
    test_flat = h5_to_pyg(data_dir, "test", label_map)
    loader_flat = DataLoader(test_flat, batch_size=BATCH_SIZE)

    loader_uv = None
    if uv_cache is not None:
        print("Loading test set (UV-net features)...")
        test_uv   = h5_to_pyg(data_dir, "test", label_map, uv_cache_dir=uv_cache)
        loader_uv = DataLoader(test_uv, batch_size=BATCH_SIZE)
    elif not args.no_uvnet:
        print("[warning] --uv_cache_dir not provided. UV-net runs will be skipped.")
        print("         Run extract_uv_features.py first, then pass --uv_cache_dir.\n")

    # ── Evaluate all experiments ────────────────────────────────────────────
    results = []
    best_exp   = None
    best_preds = None
    best_labels = None

    for exp in EXPERIMENTS:
        if exp["uv"] and loader_uv is None:
            print(f"  [skip] {exp['name']}  (no UV cache)")
            continue

        loader = loader_uv if exp["uv"] else loader_flat
        print(f"\nEvaluating: {exp['name']}  (wandb: {exp['wandb']})")

        model = load_model(exp, num_classes, device)
        acc, f1, prec, rec, loss, preds, labels = evaluate(
            model, loader, device, return_preds=True
        )

        row = {
            "experiment": exp["name"],
            "wandb_id":   exp["wandb"],
            "acc":        round(float(acc),  4),
            "macro_f1":   round(float(f1),   4),
            "precision":  round(float(prec), 4),
            "recall":     round(float(rec),  4),
        }
        results.append(row)
        print(f"  acc={acc:.4f}  macro-F1={f1:.4f}  prec={prec:.4f}  rec={rec:.4f}")

        if "best" in exp["name"]:
            best_exp    = exp
            best_preds  = preds
            best_labels = labels

    # ── Table 1 ─────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'Experiment':<35} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}  wandb")
    print("-"*72)
    for r in results:
        print(f"{r['experiment']:<35} {r['acc']:>6.4f} {r['macro_f1']:>6.4f} "
              f"{r['precision']:>6.4f} {r['recall']:>6.4f}  {r['wandb_id']}")
    print("="*72)

    # Save as CSV
    csv_path = out_dir / "table1.csv"
    with open(csv_path, "w") as f:
        f.write("experiment,wandb_id,acc,macro_f1,precision,recall\n")
        for r in results:
            f.write(f"{r['experiment']},{r['wandb_id']},{r['acc']},"
                    f"{r['macro_f1']},{r['precision']},{r['recall']}\n")
    print(f"\nTable 1 saved to {csv_path}")

    # ── Figures for best model ───────────────────────────────────────────────
    if best_preds is not None:
        print(f"\nGenerating figures for {best_exp['name']}...")

        report = classification_report(best_labels, best_preds,
                                       target_names=class_names, zero_division=0)
        rep_path = out_dir / "classification_report.txt"
        with open(rep_path, "w") as f:
            f.write(f"GraphSAGE UV-net (wandb: {best_exp['wandb']})\n\n{report}")
        print(f"  saved {rep_path}")

        plot_confusion_matrix(best_labels, best_preds, class_names,
                              out_dir / "confusion_matrix.png")
        plot_per_class_f1(best_labels, best_preds, class_names,
                          out_dir / "per_class_f1.png")
    else:
        print("\n[note] Best model (GraphSAGE UV-net) was skipped — "
              "provide --uv_cache_dir to generate figures.")

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
