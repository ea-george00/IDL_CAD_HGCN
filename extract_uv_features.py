"""
extract_uv_features.py
======================
Precompute UV-net face features from MFCAD++ STEP files.

For each B-Rep face, samples a 5×5 uniform grid over the parametric (UV)
domain, computes surface normals at each sample point, and classifies
whether the point is inside the trimmed face boundary.

Produces 7 UV features per face:
    [mean_nx, mean_ny, mean_nz,      -- mean surface normal over valid samples
     std_nx,  std_ny,  std_nz,       -- normal variance (captures curvature)
     coverage]                        -- fraction of UV grid inside trimmed boundary

Output: uv_cache/{train,val,test}.h5
    Same group structure as the original H5 files.
    Each group has a single dataset 'uv_features' of shape (N_faces, 7).
    Face ordering is preserved (row i in uv_features = row i in V_1).

Usage
-----
    python extract_uv_features.py --data_dir MFCAD_dataset.zip
    python extract_uv_features.py --data_dir MFCAD_dataset.zip --workers 4
    python extract_uv_features.py --data_dir MFCAD_dataset.zip --split val   # single split
"""

import argparse
import io
import os
import sys
import time
import traceback
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

warnings.filterwarnings("ignore")

# ── UV grid resolution (5×5 = 25 samples per face) ───────────────────────────
UV_GRID_N = 5
UV_FEATURES_DIM = 7  # [mean_nx,ny,nz, std_nx,ny,nz, coverage]

# ── Zip inner paths ───────────────────────────────────────────────────────────
_H5_INNER = {
    "train": "MFCAD++_dataset/hierarchical_graphs/training_MFCAD++.h5",
    "val":   "MFCAD++_dataset/hierarchical_graphs/val_MFCAD++.h5",
    "test":  "MFCAD++_dataset/hierarchical_graphs/test_MFCAD++.h5",
}
_STEP_INNER = "MFCAD++_dataset/step/{split}/{cad_id}.step"

# ── Per-face UV feature extraction ───────────────────────────────────────────

def _uv_features_for_face(cq_face) -> np.ndarray:
    """
    Extract 7 UV-net features for a single CadQuery Face object.

    Samples a UV_GRID_N × UV_GRID_N grid over the face's parametric domain,
    computes the surface normal at each sample, and classifies whether the
    sample is inside the trimmed face boundary using BRepTopAdaptor_FClass2d.

    Returns shape (7,) float32 array:
        [mean_nx, mean_ny, mean_nz, std_nx, std_ny, std_nz, coverage]
    Falls back to zeros on any OCC error.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCP.gp import gp_Pnt2d
    from OCP.TopAbs import TopAbs_IN

    out = np.zeros(UV_FEATURES_DIM, dtype=np.float32)
    try:
        # Access underlying OCC face
        occ_face = cq_face.wrapped

        surf = BRepAdaptor_Surface(occ_face)
        umin = surf.FirstUParameter()
        umax = surf.LastUParameter()
        vmin = surf.FirstVParameter()
        vmax = surf.LastVParameter()

        classifier = BRepTopAdaptor_FClass2d(occ_face, 1e-7)

        us = np.linspace(umin, umax, UV_GRID_N)
        vs = np.linspace(vmin, vmax, UV_GRID_N)

        normals_valid = []
        n_valid = 0

        for u in us:
            for v in vs:
                state = classifier.Perform(gp_Pnt2d(u, v))
                if state != TopAbs_IN:
                    continue
                n_valid += 1
                props = BRepLProp_SLProps(surf, u, v, 1, 1e-6)
                if props.IsNormalDefined():
                    n = props.Normal()
                    normals_valid.append([n.X(), n.Y(), n.Z()])

        total_samples = UV_GRID_N * UV_GRID_N
        coverage = n_valid / total_samples

        if normals_valid:
            normals = np.array(normals_valid, dtype=np.float32)
            mean_n = normals.mean(axis=0)
            std_n  = normals.std(axis=0)
            out[:3] = mean_n
            out[3:6] = std_n
        out[6] = coverage

    except Exception:
        pass  # returns zeros on failure

    return out


def _process_part(args) -> tuple[int, int, np.ndarray]:
    """
    Worker function: extract UV features for all faces of one CAD part.

    Args:
        args: (group_key, part_idx, face_start, face_end, step_bytes)
            step_bytes: raw bytes of the STEP file

    Returns:
        (group_key_int, part_idx, uv_array_shape=(n_faces, 7))
    """
    import tempfile
    import cadquery as cq
    import warnings
    warnings.filterwarnings("ignore")

    group_key, part_idx, face_start, face_end, step_bytes = args
    n_faces = face_end - face_start

    uv = np.zeros((n_faces, UV_FEATURES_DIM), dtype=np.float32)
    try:
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

        if len(faces) != n_faces:
            # Face count mismatch — return zeros rather than silently misalign
            return group_key, part_idx, uv

        for fi, face in enumerate(faces):
            uv[fi] = _uv_features_for_face(face)

    except Exception:
        pass  # return zeros

    return group_key, part_idx, uv


# ── Group-level processing ────────────────────────────────────────────────────

def _extract_group(group_key: str, h5_group, zf: zipfile.ZipFile,
                   split: str, workers: int) -> np.ndarray:
    """
    Extract UV features for all faces in one H5 group (a batch of CAD parts).

    Returns uv_features array of shape (N_faces_total, 7).
    Face ordering matches V_1 exactly (row i in output = row i in V_1).
    """
    cad_models = [b.decode() for b in np.array(h5_group["CAD_model"])]
    idx = np.array(h5_group["idx"])           # (n_parts, 2) — GLOBAL cumulative end indices
    n_faces_total = h5_group["V_1"].shape[0]

    # idx[:,0] are GLOBAL inclusive end indices for each part's faces.
    # Compute the global start index of this group's first face, then derive
    # LOCAL (within-group) face ranges for each part.
    group_global_start = int(idx[-1, 0]) - n_faces_total + 1

    def local_range(pi):
        """Return (local_start, local_end_excl) for part pi within this group."""
        g_end_incl = int(idx[pi, 0])                                     # global inclusive end
        g_start    = group_global_start if pi == 0 else int(idx[pi-1, 0]) + 1
        return g_start - group_global_start, g_end_incl - group_global_start + 1

    uv_out = np.zeros((n_faces_total, UV_FEATURES_DIM), dtype=np.float32)

    # Build list of (group_key, part_idx, face_start, face_end, step_bytes)
    tasks = []
    for pi, cad_id in enumerate(cad_models):
        face_start, face_end = local_range(pi)

        step_inner = _STEP_INNER.format(split=split, cad_id=cad_id)
        try:
            step_bytes = zf.read(step_inner)
        except KeyError:
            continue  # STEP file missing — leave zeros for this part
        tasks.append((group_key, pi, face_start, face_end, step_bytes))

    if workers == 1:
        results = [_process_part(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_process_part, tasks))

    for _, pi, uv_part in results:
        face_start, face_end = local_range(pi)
        uv_out[face_start:face_end] = uv_part

    return uv_out


# ── Split-level processing ────────────────────────────────────────────────────

def extract_split(data_path: Path, split: str, out_dir: Path,
                  workers: int, resume: bool):
    """
    Extract UV features for all groups in one split and write to HDF5.
    Supports resuming: skips groups whose key already exists in the output file.
    """
    out_path = out_dir / f"{split}.h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Split: {split}  ->  {out_path}")
    print(f"Workers: {workers}  |  Resume: {resume}")
    print(f"{'='*60}")

    # Open source ZIP
    zf = zipfile.ZipFile(str(data_path), "r")
    h5_bytes = zf.read(_H5_INNER[split])
    src_hf = h5py.File(io.BytesIO(h5_bytes), "r")
    group_keys = list(src_hf.keys())

    # Find already-completed groups if resuming
    done_keys: set[str] = set()
    if resume and out_path.exists():
        with h5py.File(out_path, "r") as existing:
            done_keys = set(existing.keys())
        print(f"Resuming: {len(done_keys)}/{len(group_keys)} groups already done.")

    todo = [k for k in group_keys if k not in done_keys]
    print(f"Groups to process: {len(todo)}")

    t0 = time.time()
    n_done = 0

    for gi, gk in enumerate(todo):
        t_group = time.time()
        try:
            uv_arr = _extract_group(gk, src_hf[gk], zf, split, workers=1)
        except Exception:
            traceback.print_exc()
            uv_arr = np.zeros(
                (src_hf[gk]["V_1"].shape[0], UV_FEATURES_DIM), dtype=np.float32
            )

        # Write group to output HDF5 (create or overwrite)
        mode = "a" if out_path.exists() else "w"
        with h5py.File(out_path, mode) as dst:
            if gk in dst:
                del dst[gk]   # remove stale entry before rewriting
            dst.create_dataset(gk, data=uv_arr, compression="gzip",
                               compression_opts=4)

        n_done += 1
        elapsed = time.time() - t0
        avg_per_group = elapsed / n_done
        remaining = avg_per_group * (len(todo) - n_done)
        n_faces = uv_arr.shape[0]
        print(f"  [{n_done:4d}/{len(todo)}] group={gk:>6s}  "
              f"faces={n_faces:5d}  "
              f"t={time.time()-t_group:.1f}s  "
              f"ETA={_fmt_time(remaining)}")

    src_hf.close()
    zf.close()

    total = time.time() - t0
    print(f"\nDone. {split} extraction: {_fmt_time(total)}")
    print(f"Output: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract UV-net features from MFCAD++ STEP files",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Path to MFCAD_dataset.zip")
    parser.add_argument("--out_dir", default="./uv_cache",
                        help="Output directory for uv_cache HDF5 files (default: ./uv_cache)")
    parser.add_argument("--split", default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split(s) to process (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers per group (default: 1 — use 4-8 for faster extraction)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Reprocess groups even if already in output (default: resume)")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    out_dir   = Path(args.out_dir)
    resume    = not args.no_resume

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        extract_split(data_path, split, out_dir, args.workers, resume)

    print("\nAll done.")


if __name__ == "__main__":
    main()
