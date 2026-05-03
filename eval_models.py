"""
eval_models.py — run per-class classification report on saved models
without retraining. Uses the already-saved .pt weights.

Usage:
    python eval_models.py --data_dir "C:\...\MFCAD_dataset.zip"
    python eval_models.py --data_dir "C:\...\MFCAD++_dataset"
"""
import argparse, json, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.loader import DataLoader
from train_mfcad import (load_label_map, h5_to_pyg, GCN, GraphSAGE,
                          HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE, OUT_DIR)

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        all_preds.append(out.argmax(dim=-1).cpu())
        all_labels.append(batch.y.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    label_map = load_label_map(data_dir)
    num_classes = len(label_map)
    class_names = [label_map[i] for i in range(num_classes)]

    with open(OUT_DIR / "model_config.json") as f:
        cfg = json.load(f)
    in_ch = cfg["in_channels"]

    print("Loading test split...")
    test_graphs = h5_to_pyg(data_dir, "test", label_map)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

    for model_name, ModelClass in [("gcn", GCN), ("graphsage", GraphSAGE)]:
        pt = OUT_DIR / f"{model_name}_mfcad.pt"
        if not pt.exists():
            print(f"  {pt} not found, skipping.")
            continue

        model = ModelClass(in_ch, HIDDEN_DIM, num_classes).to(device)
        model.load_state_dict(torch.load(pt, map_location=device))

        preds, labels = predict(model, test_loader, device)
        acc = (preds == labels).mean()

        print(f"\n{'='*64}")
        print(f"{model_name.upper()}  —  test acc={acc:.4f}")
        print('='*64)
        report = classification_report(labels, preds, target_names=class_names,
                                       zero_division=0)
        print(report)

        # Top-5 best and worst classes by F1
        from sklearn.metrics import f1_score
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        order = np.argsort(per_class_f1)
        print("  Worst 5 classes:")
        for i in order[:5]:
            print(f"    [{i:2d}] {class_names[i]:<35} F1={per_class_f1[i]:.3f}")
        print("  Best 5 classes:")
        for i in order[-5:][::-1]:
            print(f"    [{i:2d}] {class_names[i]:<35} F1={per_class_f1[i]:.3f}")

        # Save report
        out = OUT_DIR / f"{model_name}_classification_report.txt"
        with open(out, "w") as f:
            f.write(f"{model_name.upper()} — acc={acc:.4f}\n\n{report}")
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
