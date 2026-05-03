"""
train_mfcad.py
==============
Train GCN (baseline) and GraphSAGE (contribution) on the MFCAD++ dataset
for face-level machining feature classification.

Usage
-----
    python train_mfcad.py --data_dir /path/to/MFCAD_dataset.zip
    python train_mfcad.py --data_dir /path/to/MFCAD_dataset.zip --model gcn
    python train_mfcad.py --data_dir /path/to/MFCAD_dataset.zip --model graphsage
    python train_mfcad.py --data_dir /path/to/MFCAD_dataset.zip --model both   (default)

Outputs (written to ./models/)
-------------------------------
    gcn_mfcad.pt          GCN weights (baseline)
    graphsage_mfcad.pt    GraphSAGE weights (contribution)
    label_map.json        {int -> feature_class_name}
    model_config.json     face_in, facet_in, hidden_dim, num_classes  (needed for inference)

Requirements
------------
    pip install torch torch-geometric h5py scikit-learn tqdm

Dataset layout (MFCAD_dataset.zip)
-----------------------------------
    MFCAD++_dataset/
        feature_labels.txt
        hierarchical_graphs/
            training_MFCAD++.h5
            val_MFCAD++.h5
            test_MFCAD++.h5

    HDF5 structure per graph group (full hierarchical B-AAG):
        V_1          float32  (N_faces,  5)   face-level node features
        V_2          float32  (N_facets, 4)   facet-level node features
        A_1_idx/values        face-face adjacency  (COO + edge weights)
        A_2_idx/values        facet-facet adjacency
        A_3_idx/shape         cross-level assignment: facet → face
        labels       float32  (N_faces,)       per-face class label (prediction target)
        facet_labels int16    (N_facets,)       per-facet class label (used for weighting)
"""

import argparse
import io
import json
import os
import platform
import time
import zipfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from sklearn.metrics import (f1_score, classification_report,
                              precision_score, recall_score)
from tqdm import tqdm
import wandb


# ── Hierarchical data container ───────────────────────────────────────────────

class HierarchicalData(Data):
    """
    PyG Data extended for the MFCAD++ two-level B-AAG.

    Face level  (coarse): x, edge_index, edge_attr, y
    Facet level (fine):   x_facet, edge_index_facet, edge_attr_facet
    Cross-level:          facet_to_face  LongTensor (N_facets,)
                          facet_to_face[i] = index of the face that owns facet i

    __inc__ / __cat_dim__ teach PyG how to offset these tensors correctly
    when multiple graphs are batched together by DataLoader.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_facet":
            return self.x_facet.size(0)
        if key == "facet_to_face":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "edge_index_facet":
            return 1          # edges are concatenated along the column dimension
        return super().__cat_dim__(key, value, *args, **kwargs)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean-aggregate rows of src into dim_size output rows, grouped by index."""
    out   = src.new_zeros(dim_size, src.size(-1))
    count = src.new_zeros(dim_size, 1)
    idx   = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(0, index.unsqueeze(-1),
                       torch.ones(src.size(0), 1, dtype=src.dtype, device=src.device))
    return out / count.clamp(min=1)


# -- Data-source helpers (supports both zip and extracted folder) --------------

def _open_zip(data_path: Path) -> zipfile.ZipFile:
    p = str(data_path)
    if p.endswith(".zip") and os.path.isfile(p):
        return zipfile.ZipFile(p, "r")
    raise FileNotFoundError(
        f"Expected a .zip file, got: {data_path}. "
        "Pass the path to MFCAD_dataset.zip or the extracted MFCAD_dataset/ folder."
    )


def _read_zip_text(zf: zipfile.ZipFile, inner_path: str) -> str:
    return zf.read(inner_path).decode("utf-8", errors="replace")


def _open_h5_from_zip(zf: zipfile.ZipFile, inner_path: str) -> h5py.File:
    data = zf.read(inner_path)
    return h5py.File(io.BytesIO(data), "r")


def _is_folder(data_path: Path) -> bool:
    return data_path.is_dir()

# -- Configuration ------------------------------------------------------------

HIDDEN_DIM    = 128   # embedding dimension (used at inference in AgentsCAD)
NUM_LAYERS    = 3
DROPOUT       = 0.5
EPOCHS        = 100
LR            = 1e-3
BATCH_SIZE    = 32
PATIENCE      = 15    # early stopping patience

OUT_DIR  = Path("./models")
RUNS_DIR = Path("./runs")
OUT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

WANDB_PROJECT = "mfcad-feature-classification"
WANDB_ENTITY  = None   # set to your wandb username/org or leave None
FEATURE_SET   = "hierarchical_v1_v2"   # update to "uv_net" when UV cache is ready

# Surface type string → integer (must match AgentsCAD's GeometryParserAgent output)
SURFACE_TYPE_MAP = {
    "Plane":     0,
    "Planar":    0,
    "Cylinder":  1,
    "Cylindrical": 1,
    "Cone":      2,
    "Sphere":    3,
    "Torus":     4,
    "BSpline":   5,
    "Other":     5,
    "Unknown":   5,
}

# -- HDF5 loading -------------------------------------------------------------

# HDF5 key names — MFCAD++ hierarchical_graphs files

# Inner zip paths
_ZIP_LABELS = "MFCAD++_dataset/feature_labels.txt"
_ZIP_H5 = {
    "train": "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5",
    "val":   "MFCAD++_dataset/hierarchical_graphs/val_MFCAD++.h5",
    "test":  "MFCAD++_dataset/hierarchical_graphs/test_MFCAD++.h5",
}
_ZIP_STEP = "MFCAD++_dataset/step/{split}/{cad_id}.step"
# Flat filenames when using an extracted folder
_FOLDER_LABELS = "feature_labels.txt"
_FOLDER_H5 = {
    "train": "training_MFCAD++.h5",
    "val":   "val_MFCAD++.h5",
    "test":  "test_MFCAD++.h5",
}


def _parse_label_text(text: str) -> dict:
    label_map = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if " - " in line:
            idx_str, name = line.split(" - ", 1)
            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            label_map[idx] = name.strip()
        else:
            label_map[len(label_map)] = line
    return label_map


def _coo_to_edge_index(idx: np.ndarray, values: np.ndarray):
    """Convert COO sparse matrix arrays to undirected PyG edge_index + edge_attr."""
    if idx.ndim == 1:
        idx = idx.reshape(-1, 2)
    if len(idx) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float)
    src = torch.tensor(idx[:, 0], dtype=torch.long)
    dst = torch.tensor(idx[:, 1], dtype=torch.long)
    val = torch.tensor(values, dtype=torch.float)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_attr  = torch.cat([val, val])
    return edge_index, edge_attr


def _graphs_from_h5(f: h5py.File, split_name: str,
                    uv_cache: h5py.File = None) -> list[HierarchicalData]:
    graphs = []
    for graph_id in tqdm(list(f.keys()), desc=f"Loading {split_name}", leave=False):
        g = f[graph_id]

        # ── Face level (coarse) ───────────────────────────────────────────────
        x_face_v1 = torch.tensor(np.array(g["V_1"]), dtype=torch.float)  # (N_faces, 5)
        if uv_cache is not None and graph_id in uv_cache:
            uv = torch.tensor(np.array(uv_cache[graph_id]),
                              dtype=torch.float)                            # (N_faces, 7)
            x_face = torch.cat([x_face_v1, uv], dim=-1)                   # (N_faces, 12)
        else:
            x_face = x_face_v1
        y_face = torch.tensor(np.array(g["labels"]), dtype=torch.long)  # (N_faces,)
        edge_index_face, edge_attr_face = _coo_to_edge_index(
            np.array(g["A_1_idx"]), np.array(g["A_1_values"]))

        # ── Facet level (fine) ────────────────────────────────────────────────
        x_facet = torch.tensor(np.array(g["V_2"]), dtype=torch.float)  # (N_facets, 4)
        edge_index_facet, edge_attr_facet = _coo_to_edge_index(
            np.array(g["A_2_idx"]), np.array(g["A_2_values"]))

        # ── Cross-level mapping: facet → face (via A_3) ───────────────────────
        # A_3 is a sparse assignment matrix with exactly one entry per facet.
        # A_3_shape tells us [dim0, dim1]; from that we can tell which axis is faces.
        a3_shape = np.array(g["A_3_shape"])   # [dim0, dim1]
        a3_idx   = np.array(g["A_3_idx"])     # (N_facets, 2) COO pairs
        n_faces  = x_face.shape[0]
        n_facets = x_facet.shape[0]
        facet_to_face_np = np.zeros(n_facets, dtype=np.int64)
        if a3_shape[0] == n_faces:
            # shape = [N_faces, N_facets]: col 0 = face idx, col 1 = facet idx
            facet_to_face_np[a3_idx[:, 1]] = a3_idx[:, 0]
        else:
            # shape = [N_facets, N_faces]: col 0 = facet idx, col 1 = face idx
            facet_to_face_np[a3_idx[:, 0]] = a3_idx[:, 1]
        facet_to_face = torch.tensor(facet_to_face_np, dtype=torch.long)

        data = HierarchicalData(
            x=x_face,
            edge_index=edge_index_face,
            edge_attr=edge_attr_face,
            y=y_face,
            x_facet=x_facet,
            edge_index_facet=edge_index_facet,
            edge_attr_facet=edge_attr_facet,
            facet_to_face=facet_to_face,
        )

        # ── Viz metadata: part → CAD ID + local face range ────────────────────
        # idx[:,0] = GLOBAL cumulative end indices (inclusive) across the split.
        # Compute local (within-group) face ranges for each part.
        cad_models_list = [b.decode() for b in np.array(g["CAD_model"])]
        idx_arr = np.array(g["idx"])
        grp_global_start = int(idx_arr[-1, 0]) - n_faces + 1
        part_ranges = []
        for pi in range(len(cad_models_list)):
            g_end   = int(idx_arr[pi, 0])
            g_start = grp_global_start if pi == 0 else int(idx_arr[pi - 1, 0]) + 1
            part_ranges.append((g_start - grp_global_start,
                                 g_end   - grp_global_start + 1))
        data.graph_key       = graph_id
        data.part_cad_ids    = cad_models_list
        data.part_face_ranges = part_ranges

        graphs.append(data)
    return graphs


def inspect_h5(data_path: Path):
    """Print HDF5 structure so you can verify the key names match this script."""
    if _is_folder(data_path):
        for split, fname in _FOLDER_H5.items():
            h5_path = data_path / fname
            print(f"\n=== {h5_path} ===")
            with h5py.File(h5_path, "r") as f:
                keys = list(f.keys())
                print(f"Top-level keys ({len(keys)} total): {keys[:5]} ...")
                first = keys[0]
                print(f"\nGroup '{first}':")
                for k, v in f[first].items():
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        with _open_zip(data_path) as zf:
            for split, inner in _ZIP_H5.items():
                print(f"\n=== {inner} ===")
                with _open_h5_from_zip(zf, inner) as f:
                    keys = list(f.keys())
                    print(f"Top-level keys ({len(keys)} total): {keys[:5]} ...")
                    first = keys[0]
                    print(f"\nGroup '{first}':")
                    for k, v in f[first].items():
                        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")


def load_label_map(data_path: Path) -> dict:
    """Load label_map from feature_labels.txt. Accepts a folder or zip."""
    if _is_folder(data_path):
        with open(data_path / _FOLDER_LABELS) as f:
            text = f.read()
    else:
        with _open_zip(data_path) as zf:
            text = _read_zip_text(zf, _ZIP_LABELS)
    return _parse_label_text(text)


def h5_to_pyg(data_path: Path, split: str, label_map: dict,
              uv_cache_dir: Path = None) -> list[Data]:
    """
    Convert a MFCAD++ HDF5 split into PyG Data objects.
    Accepts either an extracted folder or the original zip.
    If uv_cache_dir is provided, UV-net features are concatenated onto V_1.
    """
    uv_cache = None
    if uv_cache_dir is not None:
        uv_path = uv_cache_dir / f"{split}.h5"
        if uv_path.exists():
            uv_cache = h5py.File(str(uv_path), "r")
        else:
            print(f"  [warning] UV cache not found at {uv_path}, using V_1 only")

    try:
        if _is_folder(data_path):
            h5_path = data_path / _FOLDER_H5[split]
            with h5py.File(h5_path, "r") as f:
                return _graphs_from_h5(f, split, uv_cache)
        else:
            with _open_zip(data_path) as zf:
                with _open_h5_from_zip(zf, _ZIP_H5[split]) as f:
                    return _graphs_from_h5(f, split, uv_cache)
    finally:
        if uv_cache is not None:
            uv_cache.close()


# ── Visualization helpers ─────────────────────────────────────────────────────


def _class_color_palette(n_classes: int) -> np.ndarray:
    """Return (n_classes, 3) uint8 RGB array with visually distinct HSV colors."""
    import colorsys
    palette = []
    for i in range(n_classes):
        h = i / n_classes
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        palette.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(palette, dtype=np.uint8)


def log_centroid_scatter(model, val_loader, device, label_map, _wrun, epoch,
                          n_graphs: int = 3):
    """
    Log 3D centroid scatter plots every 10 epochs.

    Each point = one B-Rep face.
    Position = V_1 columns 1-3 (normalised centroid x,y,z).
    Colour   = predicted class (HSV palette).

    Samples the first n_graphs from the validation loader — one forward pass only.
    """
    was_training = model.training
    model.eval()
    palette  = _class_color_palette(len(label_map))
    clouds   = {}
    n_logged = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch).argmax(dim=-1).cpu().numpy()
            for gi in range(batch.num_graphs):
                if n_logged >= n_graphs:
                    break
                mask = (batch.batch == gi).cpu().numpy().astype(bool)
                xyz  = batch.x[mask, 1:4].cpu().numpy()   # normalised centroid
                cls  = preds[mask]
                rgb  = palette[cls].astype(np.float32)
                pts  = np.column_stack([xyz, rgb])         # (N_faces, 6)
                clouds[f"centroid_scatter/g{n_logged}"] = wandb.Object3D(pts)
                n_logged += 1
            if n_logged >= n_graphs:
                break

    if clouds:
        _wrun.log({**clouds, "epoch": epoch}, step=epoch)

    if was_training:
        model.train()


def _face_pts_3d(occ_face, n_uv: int = 8) -> np.ndarray:
    """
    Sample 3D surface points on a B-Rep face via a UV grid.

    Returns (N, 3) float32 — only points that pass the trim-boundary test.
    Falls back to np.zeros((1,3)) on any OCC error.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.gp import gp_Pnt2d
    from OCP.TopAbs import TopAbs_IN
    import warnings as _w
    _w.filterwarnings("ignore")
    try:
        surf = BRepAdaptor_Surface(occ_face)
        umin, umax = surf.FirstUParameter(), surf.LastUParameter()
        vmin, vmax = surf.FirstVParameter(), surf.LastVParameter()
        clf = BRepTopAdaptor_FClass2d(occ_face, 1e-7)
        pts = []
        for u in np.linspace(umin, umax, n_uv):
            for v in np.linspace(vmin, vmax, n_uv):
                if clf.Perform(gp_Pnt2d(u, v)) == TopAbs_IN:
                    p = surf.Value(u, v)
                    pts.append([p.X(), p.Y(), p.Z()])
        return np.array(pts, dtype=np.float32) if pts else np.zeros((1, 3), dtype=np.float32)
    except Exception:
        return np.zeros((1, 3), dtype=np.float32)


def log_mesh_prediction_samples(model, test_graphs: list, data_path: Path,
                                  split: str, label_map: dict, _wrun, device,
                                  uv_cache_dir=None, n_parts: int = 4):
    """
    After training: tessellate B-Rep faces for n_parts sampled test parts,
    colour each face's 3D points by predicted class, and log as wandb.Object3D.

    Correctly predicted faces   → class colour (HSV palette).
    Mispredicted faces          → red  (#DC3232).

    Requires cadquery / OCP for STEP loading and UV-grid tessellation.
    Silently skips parts with missing STEP files or face-count mismatches.
    """
    import tempfile
    try:
        import cadquery as cq
    except ImportError:
        print("  [mesh_viz] cadquery not installed — skipping mesh visualizations.")
        return

    model.eval()
    palette     = _class_color_palette(len(label_map))
    ERROR_COLOR = np.array([220, 50, 50], dtype=np.uint8)

    # ── gather (graph_idx, part_idx) candidates with metadata ─────────────────
    candidates = [
        (gi, pi)
        for gi, g in enumerate(test_graphs)
        if hasattr(g, "part_cad_ids")
        for pi in range(len(g.part_cad_ids))
    ]
    rng = np.random.default_rng(42)
    rng.shuffle(candidates)

    # ── open data source ───────────────────────────────────────────────────────
    try:
        zf = _open_zip(data_path) if not _is_folder(data_path) else None
    except Exception:
        print("  [mesh_viz] Cannot open data zip — skipping mesh visualizations.")
        return

    # ── open UV cache if available ─────────────────────────────────────────────
    uv_h5 = None
    if uv_cache_dir is not None:
        uv_path = Path(uv_cache_dir) / f"{split}.h5"
        if uv_path.exists():
            uv_h5 = h5py.File(str(uv_path), "r")

    logged   = 0
    viz_dict = {}

    for gi, pi in candidates:
        if logged >= n_parts:
            break
        g      = test_graphs[gi]
        cad_id = g.part_cad_ids[pi]
        fs, fe = g.part_face_ranges[pi]
        n_part_faces = fe - fs

        # ── inference on single graph ─────────────────────────────────────────
        with torch.no_grad():
            from torch_geometric.data import Batch
            g_batch = Batch.from_data_list([g]).to(device)
            out     = model(g_batch)
            preds   = out.argmax(dim=-1).cpu().numpy()   # all faces in this group
            true    = g.y.numpy()   # g still on CPU
        part_preds  = preds[fs:fe]
        part_labels = true[fs:fe]

        # ── load STEP ─────────────────────────────────────────────────────────
        try:
            if zf is not None:
                step_bytes = zf.read(_ZIP_STEP.format(split=split, cad_id=cad_id))
                with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tf:
                    tf.write(step_bytes)
                    tmp = tf.name
                try:
                    faces = cq.importers.importStep(tmp).faces().vals()
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            else:
                step_path = (data_path / "MFCAD++_dataset" / "step"
                             / split / f"{cad_id}.step")
                faces = cq.importers.importStep(str(step_path)).faces().vals()
        except Exception:
            continue

        if len(faces) != n_part_faces:
            continue   # face-count mismatch — skip

        # ── tessellate: UV-sampled points per face, coloured by prediction ─────
        all_pts = []
        for fi, face in enumerate(faces):
            pts3d    = _face_pts_3d(face.wrapped)          # (N, 3)
            pred_cls = int(part_preds[fi])
            true_cls = int(part_labels[fi])
            color    = palette[pred_cls] if pred_cls == true_cls else ERROR_COLOR
            rgb      = np.broadcast_to(color, (len(pts3d), 3)).copy().astype(np.float32)
            all_pts.append(np.column_stack([pts3d, rgb]))

        if all_pts:
            cloud    = np.vstack(all_pts)
            pct_ok   = (part_preds == part_labels).mean() * 100
            key      = f"mesh_viz/part{logged}_{cad_id[:8]}_acc{pct_ok:.0f}"
            viz_dict[key] = wandb.Object3D(cloud)
            logged  += 1

    if zf is not None:
        zf.close()
    if uv_h5 is not None:
        uv_h5.close()

    if viz_dict:
        _wrun.log(viz_dict)
        print(f"  [mesh_viz] Logged {logged} mesh prediction samples to wandb.")
    else:
        print("  [mesh_viz] No samples logged "
              "(missing STEP files or face-count mismatches).")


# ── Model definitions ─────────────────────────────────────────────────────────

class HierarchicalGCN(torch.nn.Module):
    """
    Baseline: two-level Hierarchical GCN.

    1. Run GCNConv layers on the fine facet graph (V_2 + A_2).
    2. Mean-pool facet embeddings to the face level via the A_3 assignment.
    3. Concatenate pooled context with face features (V_1) and run GCNConv
       layers on the coarse face graph (A_1).
    Prediction is at the face level.
    """
    def __init__(self, face_in: int, facet_in: int,
                 hidden: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.5,
                 use_v2: bool = True):
        super().__init__()
        self.dropout = dropout
        self.use_v2  = use_v2
        face_conv_in = face_in + hidden if use_v2 else face_in
        if use_v2:
            self.facet_convs = torch.nn.ModuleList([GCNConv(facet_in, hidden)])
            for _ in range(num_layers - 1):
                self.facet_convs.append(GCNConv(hidden, hidden))
        self.face_convs = torch.nn.ModuleList([GCNConv(face_conv_in, hidden)])
        for _ in range(num_layers - 2):
            self.face_convs.append(GCNConv(hidden, hidden))
        self.face_convs.append(GCNConv(hidden, num_classes))

    def _facet_pass(self, x, edge_index):
        for i, conv in enumerate(self.facet_convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.facet_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _face_pass(self, x, edge_index):
        for i, conv in enumerate(self.face_convs):
            x = conv(x, edge_index)
            if i < len(self.face_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, batch):
        if self.use_v2:
            h_facet  = self._facet_pass(batch.x_facet, batch.edge_index_facet)
            h_pooled = _scatter_mean(h_facet, batch.facet_to_face, batch.x.size(0))
            x = torch.cat([batch.x, h_pooled], dim=-1)
        else:
            x = batch.x
        return self._face_pass(x, batch.edge_index)

    def embed(self, batch):
        """Face-level embeddings for AgentsCAD RAG / t-SNE."""
        if self.use_v2:
            h_facet  = self._facet_pass(batch.x_facet, batch.edge_index_facet)
            h_pooled = _scatter_mean(h_facet, batch.facet_to_face, batch.x.size(0))
            x = torch.cat([batch.x, h_pooled], dim=-1)
        else:
            x = batch.x
        for conv in self.face_convs[:-1]:
            x = F.relu(conv(x, batch.edge_index))
        return x


class HierarchicalGraphSAGE(torch.nn.Module):
    """
    Contribution: two-level Hierarchical GraphSAGE.

    Same two-level structure as HierarchicalGCN but uses SAGEConv — inductive,
    generalises to unseen graphs (new STEP files at AgentsCAD inference time).
    """
    def __init__(self, face_in: int, facet_in: int,
                 hidden: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.5,
                 use_v2: bool = True):
        super().__init__()
        self.dropout = dropout
        self.use_v2  = use_v2
        face_conv_in = face_in + hidden if use_v2 else face_in
        if use_v2:
            self.facet_convs = torch.nn.ModuleList([SAGEConv(facet_in, hidden)])
            for _ in range(num_layers - 1):
                self.facet_convs.append(SAGEConv(hidden, hidden))
        self.face_convs = torch.nn.ModuleList([SAGEConv(face_conv_in, hidden)])
        for _ in range(num_layers - 2):
            self.face_convs.append(SAGEConv(hidden, hidden))
        self.face_convs.append(SAGEConv(hidden, num_classes))

    def _facet_pass(self, x, edge_index):
        for i, conv in enumerate(self.facet_convs):
            x = F.relu(conv(x, edge_index))
            if i < len(self.facet_convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def _face_pass(self, x, edge_index):
        for i, conv in enumerate(self.face_convs):
            x = conv(x, edge_index)
            if i < len(self.face_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, batch):
        if self.use_v2:
            h_facet  = self._facet_pass(batch.x_facet, batch.edge_index_facet)
            h_pooled = _scatter_mean(h_facet, batch.facet_to_face, batch.x.size(0))
            x = torch.cat([batch.x, h_pooled], dim=-1)
        else:
            x = batch.x
        return self._face_pass(x, batch.edge_index)

    def embed(self, batch):
        """128-dim face embeddings for AgentsCAD RAG."""
        if self.use_v2:
            h_facet  = self._facet_pass(batch.x_facet, batch.edge_index_facet)
            h_pooled = _scatter_mean(h_facet, batch.facet_to_face, batch.x.size(0))
            x = torch.cat([batch.x, h_pooled], dim=-1)
        else:
            x = batch.x
        for conv in self.face_convs[:-1]:
            x = F.relu(conv(x, batch.edge_index))
        return x


# -- Training / evaluation -----------------------------------------------------

def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(out, batch.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, return_preds=False, class_weights=None):
    model.eval()
    all_preds, all_labels, all_loss = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)
        preds = out.argmax(dim=-1)
        loss  = F.cross_entropy(out, batch.y, weight=class_weights)
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu())
        all_loss.append(loss.item() * batch.num_graphs)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc   = (preds == labels).mean()
    f1    = f1_score(labels,    preds, average="macro", zero_division=0)
    prec  = precision_score(labels, preds, average="macro", zero_division=0)
    rec   = recall_score(labels,    preds, average="macro", zero_division=0)
    vloss = sum(all_loss) / len(loader.dataset)
    if return_preds:
        return acc, f1, prec, rec, vloss, preds, labels
    return acc, f1, prec, rec, vloss


def save_run_summary(run_dir: Path, model_name: str, config: dict,
                     results: dict, label_map: dict,
                     per_class_f1: np.ndarray, training_time: float,
                     stopped_early: bool, stopped_epoch: int,
                     class_weights=None):
    """Write a human-readable summary file for a completed training run."""
    num_classes = len(label_map)
    class_names = [label_map[i] for i in range(num_classes)]
    order = np.argsort(per_class_f1)

    lines = []
    lines.append(f"{'='*64}")
    lines.append(f"Run: {run_dir.name}")
    lines.append(f"Model: {model_name.upper()}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Platform: {platform.node()}  ({platform.system()})")
    lines.append(f"{'='*64}")
    lines.append("")

    lines.append("-- Config --------------------------------------------------")
    for k, v in config.items():
        lines.append(f"  {k:<20} {v}")
    lines.append("")

    lines.append("-- Results -------------------------------------------------")
    lines.append(f"  Test accuracy   {results['test_acc']:.4f}")
    lines.append(f"  Macro F1        {results['test_f1']:.4f}")
    lines.append(f"  Best val F1     {results['best_val_f1']:.4f}")
    lines.append(f"  Epochs run      {stopped_epoch}")
    lines.append(f"  Early stopped   {'yes' if stopped_early else 'no'}")
    lines.append(f"  Training time   {_fmt_time(training_time)}")
    lines.append("")

    lines.append("-- Best 5 classes by F1 -------------------------------------")
    for i in reversed(order[-5:]):
        sup = results['per_class_support'][i]
        lines.append(f"  [{i:2d}] {class_names[i]:<38} F1={per_class_f1[i]:.3f}  (n={sup:,})")
    lines.append("")

    lines.append("-- Worst 5 classes by F1 ------------------------------------")
    for i in order[:5]:
        sup = results['per_class_support'][i]
        lines.append(f"  [{i:2d}] {class_names[i]:<38} F1={per_class_f1[i]:.3f}  (n={sup:,})")
    lines.append("")

    lines.append("-- Insights -------------------------------------------------")
    # Auto-generate some observations
    low_f1 = [(class_names[i], per_class_f1[i]) for i in order[:5]]
    high_f1 = [(class_names[i], per_class_f1[i]) for i in reversed(order[-5:])]
    avg_low = np.mean([f for _, f in low_f1])
    avg_high = np.mean([f for _, f in high_f1])
    lines.append(f"  * Average F1 of top-5 classes:    {avg_high:.3f}")
    lines.append(f"  * Average F1 of bottom-5 classes: {avg_low:.3f}")
    zero_classes = [class_names[i] for i in range(num_classes) if per_class_f1[i] == 0.0]
    if zero_classes:
        lines.append(f"  * Classes with F1=0 (model never predicted correctly):")
        for c in zero_classes:
            lines.append(f"      - {c}")
    support = results['per_class_support']
    largest_idx  = int(np.argmax(support))
    smallest_idx = int(np.argmin(support))
    lines.append(f"  * Class imbalance: largest class "
                 f"({class_names[largest_idx]}) has "
                 f"{support[largest_idx]:,} faces vs "
                 f"smallest ({class_names[smallest_idx]}) "
                 f"with {support[smallest_idx]:,} "
                 f"(ratio {support[largest_idx]/max(support[smallest_idx],1):.0f}x)")
    lines.append("")

    lines.append("-- Class Weighting ------------------------------------------")
    if class_weights is not None:
        w = class_weights.cpu().numpy()
        w_order = np.argsort(w)
        lines.append(f"  Weighted loss: YES (inverse-frequency, normalised)")
        lines.append(f"  Weight range:  {w.min():.4f} – {w.max():.4f}  "
                     f"(ratio {w.max()/max(w.min(), 1e-9):.1f}x)")
        lines.append(f"  Highest-weighted classes (rarest, most penalised):")
        for i in reversed(w_order[-5:]):
            lines.append(f"    [{i:2d}] {class_names[i]:<38} w={w[i]:.4f}  (n={support[i]:,})")
        lines.append(f"  Lowest-weighted classes (most common, least penalised):")
        for i in w_order[:5]:
            lines.append(f"    [{i:2d}] {class_names[i]:<38} w={w[i]:.4f}  (n={support[i]:,})")
    else:
        lines.append(f"  Weighted loss: NO (uniform weights)")
        lines.append(f"  NOTE: class imbalance ratio is "
                     f"{support[largest_idx]/max(support[smallest_idx],1):.0f}x — "
                     f"consider --weighted_loss on next run to improve minority-class recall")
    lines.append("")
    lines.append("-- Notes (add your own) " + "-" * 41)
    lines.append("")
    lines.append("")

    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return summary_path


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def train_model(model_name: str, model, train_loader, val_loader, test_loader,
                device, label_map: dict, face_in: int, facet_in: int, num_classes: int,
                run_dir: Path, class_weights=None, resumed_from: str = None,
                no_wandb: bool = False, data_path: Path = None,
                uv_cache_dir: Path = None):
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    # ── wandb run ─────────────────────────────────────────────────────────────
    _wrun = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        mode="disabled" if no_wandb else "online",
        name=f"{model_name} {'v1' if FEATURE_SET == 'v1_only' else 'v1+v2' if FEATURE_SET == 'hierarchical_v1_v2' else 'uvn'}",
        group=run_dir.name,
        tags=[
            f"arch:{model_name}",
            f"feature_set:{FEATURE_SET}",
            f"weighted_loss:{'yes' if class_weights is not None else 'no'}",
            "live",
        ],
        config={
            "hidden_dim":    HIDDEN_DIM,
            "num_layers":    NUM_LAYERS,
            "epochs_max":    EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "dropout":       DROPOUT,
            "patience":      PATIENCE,
            "face_in":       face_in,
            "facet_in":      facet_in,
            "num_classes":   num_classes,
            "arch":          model_name,
            "feature_set":   FEATURE_SET,
            "weighted_loss": class_weights is not None,
            "resumed_from":  resumed_from or "scratch",
        },
        reinit="finish_previous",
    )

    # Define epoch as x-axis for all time-series charts
    _wrun.define_metric("epoch")
    _wrun.define_metric("train/*",        step_metric="epoch")
    _wrun.define_metric("val/*",          step_metric="epoch")
    _wrun.define_metric("test/*",         step_metric="epoch")
    _wrun.define_metric("lr",             step_metric="epoch")
    _wrun.define_metric("best_val_f1",    step_metric="epoch")

    best_val_f1   = 0.0
    patience_ctr  = 0
    best_weights  = None
    epoch_times   = []
    train_start   = time.time()
    stopped_early = False
    final_epoch   = EPOCHS

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, device, class_weights)
        val_acc, val_f1, val_prec, val_rec, val_loss = evaluate(
            model, val_loader, device, class_weights=class_weights)
        scheduler.step(1 - val_f1)
        epoch_sec = time.time() - t0
        epoch_times.append(epoch_sec)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            improved = " ✓"
        else:
            patience_ctr += 1
            improved = ""

        # Per-epoch test metrics (current weights, not best — shows learning curve)
        test_acc_e, test_f1_e, test_prec_e, test_rec_e, test_loss_e = evaluate(
            model, test_loader, device, class_weights=class_weights)

        log_dict = {
            "epoch":          epoch,
            "train/loss":     loss,
            "val/loss":       val_loss,
            "val/acc":        val_acc,
            "val/f1":         val_f1,
            "val/precision":  val_prec,
            "val/recall":     val_rec,
            "test/loss":      test_loss_e,
            "test/acc":       test_acc_e,
            "test/f1":        test_f1_e,
            "test/precision": test_prec_e,
            "test/recall":    test_rec_e,
            "best_val_f1":    best_val_f1,
            "lr":             optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_sec,
        }
        if epoch % 10 == 0 or epoch == 1:
            train_acc_e, train_f1_e, train_prec_e, train_rec_e, _ = evaluate(
                model, train_loader, device, class_weights=class_weights)
            log_dict["train/acc"]       = train_acc_e
            log_dict["train/f1"]        = train_f1_e
            log_dict["train/precision"] = train_prec_e
            log_dict["train/recall"]    = train_rec_e
        _wrun.log(log_dict, step=epoch)

        if epoch % 10 == 0 or epoch == 1:
            avg_ep    = sum(epoch_times) / len(epoch_times)
            elapsed   = time.time() - train_start
            remaining = avg_ep * (EPOCHS - epoch)
            print(f"  Epoch {epoch:03d}/{EPOCHS} | {_fmt_time(epoch_sec)}/ep | "
                  f"elapsed {_fmt_time(elapsed)} | ETA {_fmt_time(remaining)} | "
                  f"loss {loss:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}{improved}")
            log_centroid_scatter(model, val_loader, device, label_map, _wrun, epoch)

        if patience_ctr >= PATIENCE:
            elapsed = time.time() - train_start
            print(f"  Early stopping at epoch {epoch}. "
                  f"Total time: {_fmt_time(elapsed)}")
            _wrun.log({"early_stopped": True, "stopped_epoch": epoch})
            stopped_early = True
            final_epoch   = epoch
            break

    total_time = time.time() - train_start
    print(f"  Total training time: {_fmt_time(total_time)}")

    # -- Final test evaluation (best weights) --
    model.load_state_dict(best_weights)
    test_acc, test_f1, test_prec, test_rec, _, all_preds, all_labels = evaluate(
        model, test_loader, device, return_preds=True, class_weights=class_weights)
    print(f"\n  {model_name.upper()} TEST → acc={test_acc:.4f}  macro-F1={test_f1:.4f}"
          f"  prec={test_prec:.4f}  rec={test_rec:.4f}")

    # -- Per-class metrics --
    class_names   = [label_map[i] for i in range(num_classes)]
    report        = classification_report(all_labels, all_preds,
                                          target_names=class_names, zero_division=0)
    from sklearn.metrics import f1_score as f1s
    per_class_f1  = f1s(all_labels, all_preds, average=None, zero_division=0)
    support       = np.bincount(all_labels, minlength=num_classes).tolist()
    print(f"\n  Per-class report:\n{report}")

    # ── wandb: per-class F1 table ─────────────────────────────────────────────
    pclass_table = wandb.Table(
        columns=["class_id", "class_name", "f1", "support"],
        data=[
            [i, label_map[i], float(per_class_f1[i]), int(support[i])]
            for i in range(num_classes)
        ],
    )
    bar_chart = wandb.plot.bar(pclass_table, "class_name", "f1",
                               title="Per-class F1 (test)")
    conf_matrix = wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels.tolist(),
        preds=all_preds.tolist(),
        class_names=class_names,
    )
    _wrun.log({
        "per_class_f1_table": pclass_table,
        "per_class_f1_bar":   bar_chart,
        "confusion_matrix":   conf_matrix,
    })
    _wrun.summary["test/acc"]       = float(test_acc)
    _wrun.summary["test/f1"]        = float(test_f1)
    _wrun.summary["test/precision"] = float(test_prec)
    _wrun.summary["test/recall"]    = float(test_rec)
    _wrun.summary["best_val_f1"]    = float(best_val_f1)
    _wrun.summary["epochs_run"]     = final_epoch
    _wrun.summary["stopped_early"]  = stopped_early

    # -- Save to run folder --
    (run_dir / model_name).mkdir(parents=True, exist_ok=True)
    model_run_dir = run_dir / model_name

    weighted_tag = "weighted_loss=yes" if class_weights is not None else "weighted_loss=no"
    report_path = model_run_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name.upper()} — acc={test_acc:.4f}  macro-F1={test_f1:.4f}  {weighted_tag}\n\n")
        f.write(report)

    config = {
        "hidden_dim":    HIDDEN_DIM,
        "num_layers":    NUM_LAYERS,
        "epochs_max":    EPOCHS,
        "epochs_run":    final_epoch,
        "batch_size":    BATCH_SIZE,
        "lr":            LR,
        "dropout":       DROPOUT,
        "patience":      PATIENCE,
        "face_in":       face_in,
        "facet_in":      facet_in,
        "num_classes":   num_classes,
        "arch":          "hierarchical",
        "weighted_loss": "yes" if class_weights is not None else "no",
        "resumed_from":  resumed_from or "scratch",
    }
    results = {
        "test_acc":           float(test_acc),
        "test_f1":            float(test_f1),
        "best_val_f1":        float(best_val_f1),
        "per_class_f1":       per_class_f1.tolist(),
        "per_class_support":  support,
    }
    summary_path = save_run_summary(
        model_run_dir, model_name, config, results, label_map,
        per_class_f1, total_time, stopped_early, final_epoch,
        class_weights=class_weights,
    )
    print(f"  Run summary saved: {summary_path}")

    # Also mirror report to models/ for backward compat
    mirror = OUT_DIR / f"{model_name}_classification_report.txt"
    with open(mirror, "w", encoding="utf-8") as f:
        f.write(f"{model_name.upper()} — acc={test_acc:.4f}  macro-F1={test_f1:.4f}  {weighted_tag}\n\n")
        f.write(report)

    # -- Save weights (both run folder and models/) --
    weights_path = model_run_dir / f"{model_name}_mfcad.pt"
    torch.save(best_weights, weights_path)
    out_path = OUT_DIR / f"{model_name}_mfcad.pt"
    torch.save(best_weights, out_path)
    print(f"  Saved: {out_path}")

    # ── wandb: mesh prediction visualizations ────────────────────────────────
    if data_path is not None:
        print("  Generating mesh prediction visualizations...")
        log_mesh_prediction_samples(
            model, test_loader.dataset, data_path, "test",
            label_map, _wrun, device,
            uv_cache_dir=uv_cache_dir, n_parts=4,
        )

    # ── wandb: save artifacts and close run ───────────────────────────────────
    wandb.save(str(report_path))
    wandb.save(str(summary_path))
    wandb.save(str(out_path))
    _wrun.finish()

    return model


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GCN / GraphSAGE on MFCAD++",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to MFCAD_dataset.zip or extracted folder")
    parser.add_argument("--model", default="both",
                        choices=["gcn", "graphsage", "both"],
                        help="Which model(s) to train (default: both)")
    parser.add_argument("--resume", default=None,
                        help=(
                            "Path to a .pt weights file (or a runs/ subfolder) to\n"
                            "warm-start from. The model architecture must match.\n"
                            "Examples:\n"
                            "  --resume models/graphsage_mfcad.pt\n"
                            "  --resume runs/run_20260410_both/graphsage/graphsage_mfcad.pt\n"
                            "  --resume runs/run_20260410_both   (loads gcn + graphsage automatically)"
                        ))
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (useful when resuming, e.g. --lr 3e-4)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (e.g. --epochs 50 for a fine-tune pass)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print HDF5 schema and exit")
    parser.add_argument("--uv_cache", default=None,
                        help="Path to uv_cache/ directory produced by extract_uv_features.py.\n"
                             "When provided, UV-net features (7D) are concatenated onto V_1 (5D)\n"
                             "giving 12D face features. Set FEATURE_SET='uv_net' in the script.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging (file-only mode)")
    parser.add_argument("--flat", action="store_true",
                        help="Use V_1 face features only (skip V_2 facet pass). "
                             "Reproduces the v1-only baseline architecture.")
    args = parser.parse_args()

    # Apply CLI overrides to module-level config
    global LR, EPOCHS
    if args.lr is not None:
        LR = args.lr
        print(f"LR overridden to {LR}")
    if args.epochs is not None:
        EPOCHS = args.epochs
        print(f"EPOCHS overridden to {EPOCHS}")

    data_dir = Path(args.data_dir)

    if args.inspect:
        inspect_h5(data_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Create timestamped run folder --
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag  = f"run_{run_ts}_{args.model}"
    if args.resume:
        run_tag += "_resumed"
    run_dir = RUNS_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {run_dir}")

    # -- Load label map --
    label_map = load_label_map(data_dir)
    num_classes = len(label_map)
    print(f"Classes: {num_classes}")
    for i, name in label_map.items():
        print(f"  {i:2d}: {name}")

    # -- Load graphs --
    uv_cache_dir = Path(args.uv_cache) if args.uv_cache else None
    global FEATURE_SET
    if uv_cache_dir:
        FEATURE_SET = "uv_net"
        print(f"UV cache: {uv_cache_dir}  (FEATURE_SET → uv_net)")
    elif args.flat:
        FEATURE_SET = "v1_only"
        print("Flat mode: V_1 features only  (FEATURE_SET → v1_only)")
    print("\nLoading dataset...")
    train_graphs = h5_to_pyg(data_dir, "train", label_map, uv_cache_dir)
    val_graphs   = h5_to_pyg(data_dir, "val",   label_map, uv_cache_dir)
    test_graphs  = h5_to_pyg(data_dir, "test",  label_map, uv_cache_dir)

    face_in  = train_graphs[0].x.shape[1]        # V_1: 5
    facet_in = train_graphs[0].x_facet.shape[1]  # V_2: 4
    print(f"Face feature dim:  {face_in}")
    print(f"Facet feature dim: {facet_in}")
    print(f"Train/val/test: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE)

    # Inverse-frequency class weights — upweight rare classes in cross-entropy loss
    all_train_labels = torch.cat([g.y for g in train_graphs])  # face-level labels
    counts = torch.bincount(all_train_labels, minlength=num_classes).float()
    class_weights = 1.0 / counts.clamp(min=1)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
    print(f"Class weights: min={class_weights.min():.4f}  max={class_weights.max():.4f}  "
          f"imbalance ratio={class_weights.max()/class_weights.min():.1f}x")

    # -- Save shared metadata (needed for AgentsCAD inference) --
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    model_config = {
        "face_in":     face_in,
        "facet_in":    facet_in,
        "hidden_dim":  HIDDEN_DIM,
        "num_classes": num_classes,
        "num_layers":  NUM_LAYERS,
        "arch":        "hierarchical",
    }
    with open(OUT_DIR / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nMetadata saved to {OUT_DIR}/")

    # -- Resolve resume checkpoints --
    # Builds a dict {model_name: Path} from --resume argument
    resume_weights = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file() and resume_path.suffix == ".pt":
            # Single .pt file — figure out which model it belongs to from filename
            for name in ["gcn", "graphsage"]:
                if name in resume_path.stem.lower():
                    resume_weights[name] = resume_path
                    break
            else:
                # Can't infer — apply to all models being trained
                for name in (["gcn", "graphsage"] if args.model == "both" else [args.model]):
                    resume_weights[name] = resume_path
        elif resume_path.is_dir():
            # Could be a run folder (contains gcn/ and graphsage/ subdirs)
            # or a model subfolder directly
            for name in ["gcn", "graphsage"]:
                candidates = [
                    resume_path / f"{name}_mfcad.pt",          # run subfolder
                    resume_path / name / f"{name}_mfcad.pt",   # run parent folder
                ]
                for c in candidates:
                    if c.exists():
                        resume_weights[name] = c
                        break
        if resume_weights:
            print("\nResuming from:")
            for name, p in resume_weights.items():
                print(f"  {name}: {p}")
        else:
            print(f"\nWARNING: --resume path '{args.resume}' did not resolve to any .pt files.")

    # -- Train --
    models_to_train = ["gcn", "graphsage"] if args.model == "both" else [args.model]

    use_v2 = not args.flat

    for name in models_to_train:
        if name == "gcn":
            net = HierarchicalGCN(face_in, facet_in, HIDDEN_DIM, num_classes,
                                  NUM_LAYERS, DROPOUT, use_v2=use_v2)
        else:
            net = HierarchicalGraphSAGE(face_in, facet_in, HIDDEN_DIM, num_classes,
                                        NUM_LAYERS, DROPOUT, use_v2=use_v2)

        # Warm-start from previous run if provided
        if name in resume_weights:
            ckpt = resume_weights[name]
            state = torch.load(ckpt, map_location=device)
            net.load_state_dict(state)
            print(f"  Loaded weights for {name} from {ckpt}")

        train_model(name, net, train_loader, val_loader, test_loader,
                    device, label_map, face_in, facet_in, num_classes, run_dir,
                    class_weights=class_weights,
                    resumed_from=str(resume_weights[name]) if name in resume_weights else None,
                    no_wandb=args.no_wandb,
                    data_path=data_dir,
                    uv_cache_dir=uv_cache_dir)

    print(f"\nAll done. Results saved to {run_dir}/")
    print("Copy ./models/ into agentsCAD/models/ to use in the pipeline.")


if __name__ == "__main__":
    main()
