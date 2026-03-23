"""
Aggregate well-level embeddings to compound level by INCHIKEY.

Reads an H5 file with well_ids (e.g. 26247-G13-6.npz) and a metadata CSV
with SAMPLE_KEY and INCHIKEY columns. Groups embeddings by INCHIKEY and
outputs mean embeddings per unique compound.
"""

import argparse
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate embeddings to compound level by INCHIKEY."
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="Path to input H5 file (well/replicate-level embeddings)",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        required=True,
        help="Path to output H5 file (compound-level embeddings)",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        required=True,
        help="CSV with SAMPLE_KEY and INCHIKEY columns (e.g. split file)",
    )
    parser.add_argument(
        "--sample_key_col",
        type=str,
        default="SAMPLE_KEY",
        help="Column name for well/sample ID",
    )
    parser.add_argument(
        "--inchikey_col",
        type=str,
        default="INCHIKEY",
        help="Column name for INCHIKEY",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load metadata: well_id -> INCHIKEY
    meta = pd.read_csv(args.metadata_csv)
    meta = meta[[args.sample_key_col, args.inchikey_col]].drop_duplicates()
    well_to_inchikey = dict(
        zip(
            meta[args.sample_key_col].astype(str),
            meta[args.inchikey_col].astype(str),
        )
    )

    with h5py.File(args.input_h5, "r") as f:
        embeddings = f["embeddings"][:]
        well_ids = [
            w.decode("utf-8") if isinstance(w, bytes) else str(w)
            for w in f["well_id"][:]
        ]

    # Normalize H5 ids (strip .npz) and group by INCHIKEY
    inchikey_to_indices = defaultdict(list)
    skipped = 0
    for i, wid in enumerate(well_ids):
        well_key = wid.replace(".npz", "")
        inchikey = well_to_inchikey.get(well_key)
        if inchikey is None:
            skipped += 1
            continue
        inchikey_to_indices[inchikey].append(i)

    inchikey_ids = sorted(inchikey_to_indices.keys())
    aggregated = []
    for ik in inchikey_ids:
        indices = inchikey_to_indices[ik]
        agg_emb = np.mean(embeddings[indices], axis=0).astype(np.float32)
        aggregated.append(agg_emb)

    aggregated = np.stack(aggregated)

    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("embeddings", data=aggregated)
        f.create_dataset(
            "well_id",
            data=np.array(inchikey_ids, dtype=h5py.string_dtype(encoding="utf-8")),
        )

    print(
        f"Aggregated {len(well_ids)} wells -> {len(inchikey_ids)} compounds (INCHIKEY). "
        f"Skipped {skipped} wells not in metadata. Saved to {args.output_h5}"
    )


if __name__ == "__main__":
    main()
