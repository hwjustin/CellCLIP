"""
Aggregate replicate-level embeddings to compound/perturbation level by averaging.

Reads an H5 file with well_id format PLATE-WELL-REPLICATE.npz (e.g. 26247-G13-6.npz)
and outputs a new H5 with compound-level IDs (e.g. 26247-G13) and mean embeddings.
"""

import argparse
from collections import defaultdict

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate replicate embeddings to compound level."
    )
    parser.add_argument(
        "--input_h5",
        type=str,
        required=True,
        help="Path to input H5 file (replicate-level embeddings)",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        required=True,
        help="Path to output H5 file (compound-level embeddings)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with h5py.File(args.input_h5, "r") as f:
        embeddings = f["embeddings"][:]
        well_ids = [
            w.decode("utf-8") if isinstance(w, bytes) else str(w)
            for w in f["well_id"][:]
        ]

    # Group indices by compound ID (strip .npz and last "-REPLICATE")
    compound_to_indices = defaultdict(list)
    for i, wid in enumerate(well_ids):
        clean = wid.replace(".npz", "")
        compound_id = clean.rsplit("-", 1)[0]  # e.g. "26247-G13-6" -> "26247-G13"
        compound_to_indices[compound_id].append(i)

    # Average embeddings per compound, preserve order
    compound_ids = list(compound_to_indices.keys())
    compound_ids.sort()

    aggregated = []
    for cid in compound_ids:
        indices = compound_to_indices[cid]
        agg_emb = np.mean(embeddings[indices], axis=0).astype(np.float32)
        aggregated.append(agg_emb)

    aggregated = np.stack(aggregated)

    with h5py.File(args.output_h5, "w") as f:
        f.create_dataset("embeddings", data=aggregated)
        f.create_dataset(
            "well_id",
            data=np.array(compound_ids, dtype=h5py.string_dtype(encoding="utf-8")),
        )

    print(
        f"Aggregated {len(well_ids)} replicates -> {len(compound_ids)} compounds. "
        f"Saved to {args.output_h5}"
    )


if __name__ == "__main__":
    main()
