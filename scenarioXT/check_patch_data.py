#!/usr/bin/env python3
"""Verify patch H5 vs train/val/test splits (same filters as PatchWellBagDataset).

Reports:
  - H5 shape / dtype
  - Per-split: well count, site count, rows skipped (missing H5, label, mol text)
  - Union of SAMPLE_KEYs used vs H5 size
  - Random-sample L2 norms (detect all-zero failed extractions)

Example:
  python scenarioXT/check_patch_data.py
  python scenarioXT/check_patch_data.py --patch_h5 /path/to/file.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd


def site_to_well_key(sample_key: str) -> str:
    clean = sample_key.replace(".npz", "")
    if "-" not in clean:
        return clean
    return clean.rsplit("-", 1)[0]


def load_label_map(labels_csv: str) -> dict:
    labels_df = pd.read_csv(labels_csv)
    label_cols = [c for c in labels_df.columns if c != "INCHIKEY"]
    return {
        str(row["INCHIKEY"]): row[label_cols].to_numpy(dtype=np.float32)
        for _, row in labels_df.iterrows()
    }


def build_h5_lookup(patch_h5: str) -> tuple[dict, int]:
    with h5py.File(patch_h5, "r") as f:
        wids = f["well_id"][:]
    h5_id_to_idx: dict[str, int] = {}
    for i, wid in enumerate(wids):
        s = wid.decode("utf-8") if isinstance(wid, bytes) else wid
        h5_id_to_idx[s] = i
        stripped = s.replace(".npz", "") if s.endswith(".npz") else s
        h5_id_to_idx[stripped] = i
    return h5_id_to_idx, len(wids)


def wells_for_split(
    split_csv: str,
    h5_id_to_idx: dict,
    label_map: dict,
    mol_text: dict,
) -> tuple[list, dict, dict]:
    split_df = pd.read_csv(split_csv)
    well_to_site_keys: dict[str, list] = {}
    skipped = {"no_h5": 0, "no_label": 0, "no_mol": 0}
    for _, row in split_df.iterrows():
        sk = str(row["SAMPLE_KEY"])
        ik = str(row["INCHIKEY"])
        if sk not in h5_id_to_idx:
            skipped["no_h5"] += 1
            continue
        if ik not in label_map:
            skipped["no_label"] += 1
            continue
        if sk not in mol_text:
            skipped["no_mol"] += 1
            continue
        wk = site_to_well_key(sk)
        well_to_site_keys.setdefault(wk, []).append(sk)
    wells = sorted(k for k, v in well_to_site_keys.items() if v)
    return wells, well_to_site_keys, skipped


def parse_args():
    p = argparse.ArgumentParser(description="Check patch H5 vs scenarioXT splits.")
    p.add_argument(
        "--patch_h5",
        default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_patch4x4.h5",
    )
    p.add_argument(
        "--molecule_csv",
        default="/data/huadi/cellpainting_data/bray2017/mol/cell_long_captions_all.csv",
    )
    p.add_argument(
        "--labels_csv",
        default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv",
    )
    p.add_argument("--train_split", default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv")
    p.add_argument("--val_split", default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv")
    p.add_argument("--test_split", default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv")
    p.add_argument("--norm_sample", type=int, default=5000, help="Sites to sample for L2 norm stats (0=skip).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.patch_h5):
        print("ERROR: patch H5 not found:", args.patch_h5, file=sys.stderr)
        sys.exit(1)

    print("=== H5 ===")
    with h5py.File(args.patch_h5, "r") as f:
        emb = f["embeddings"]
        print("embeddings shape:", emb.shape, "dtype:", emb.dtype)
        print("well_id count:", f["well_id"].shape[0])

    h5_id_to_idx, n_h5 = build_h5_lookup(args.patch_h5)
    label_map = load_label_map(args.labels_csv)
    mol_df = pd.read_csv(args.molecule_csv)
    mol_text = dict(zip(mol_df.iloc[:, 0].astype(str), mol_df.iloc[:, 1].astype(str)))

    split_map = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split,
    }

    print("\n=== Splits (filters: in H5, has label, has mol text) ===")
    used_sites: set[str] = set()
    for name, path in split_map.items():
        wells, w2s, skipped = wells_for_split(path, h5_id_to_idx, label_map, mol_text)
        n_sites = sum(len(w2s[w]) for w in wells)
        print(f"\n{name}: wells={len(wells)}  sites={n_sites}")
        print(f"  skipped rows: not_in_h5={skipped['no_h5']}  no_label={skipped['no_label']}  no_mol={skipped['no_mol']}")
        for w in wells:
            used_sites.update(w2s[w])

    print(f"\nUnique SAMPLE_KEY in training pipeline (union): {len(used_sites)}")
    print(f"H5 rows: {n_h5}")
    if len(used_sites) != n_h5:
        print(
            "NOTE: union size != H5 row count (some H5 sites unused by these splits, "
            "or some split keys missing from H5 — see skipped counts)."
        )

    if args.norm_sample <= 0:
        return

    print(f"\n=== L2 norms (random {args.norm_sample} sites from union) ===")
    rng = np.random.default_rng(args.seed)
    keys = list(used_sites)
    rng.shuffle(keys)
    keys = keys[: min(args.norm_sample, len(keys))]
    dead = 0
    norms = []
    with h5py.File(args.patch_h5, "r") as f:
        emb = f["embeddings"]
        for sk in keys:
            idx = h5_id_to_idx[sk]
            x = np.asarray(emb[idx], dtype=np.float32)
            n = float(np.linalg.norm(x))
            norms.append(n)
            if n < 1e-6:
                dead += 1
    norms = np.array(norms)
    print(f"L2: min={norms.min():.4f}  median={np.median(norms):.4f}  max={norms.max():.4f}")
    print(f"near-dead (||x|| < 1e-6): {dead}")


if __name__ == "__main__":
    main()
