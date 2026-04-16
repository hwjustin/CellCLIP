"""Extract spatially-pooled DINOv2-giant patch tokens for all cell painting sites.

For each NPZ image (520, 696, 5):
  1. Center-crop to 224x224.
  2. For each of 5 channels, replicate to RGB and normalise with the
     DINOv2 image processor (rescale + ImageNet mean/std).
  3. Forward through frozen DINOv2-giant  ->  (257, 1536) per channel
     where token 0 = CLS, tokens 1-256 = 16x16 patch grid.
  4. Reshape patches to (16, 16, 1536), apply adaptive_avg_pool2d
     to (pool_size, pool_size) -> pool_size^2 spatial tokens.
  5. Concatenate [CLS, spatial_tokens] -> (1 + pool_size^2, 1536).
  6. Store per-site as (5, 1+pool_size^2, 1536) in float16 HDF5.

Output H5 layout (default pool_size=4):
  embeddings : (N, 5, 17, 1536)  float16     [CLS @ index 0, 16 spatial @ 1:]
  well_id    : (N,)              UTF-8 string

Usage:
  CUDA_VISIBLE_DEVICES=0 python scenarioXT/extract_dinov2_patch_tokens.py \
      --input_dir /data/huadi/cellpainting_data/cpg0012/npzs \
      --output_file dinov2-giant_patch4x4.h5 \
      --pool_size 4 --batch_size 128
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellPaintingSiteDataset(Dataset):
    """Load individual site NPZ files for patch-token extraction."""

    def __init__(self, npz_dir: str, file_ids: list, processor, crop_size: int = 224):
        self.npz_dir = npz_dir
        self.file_ids = file_ids
        self.processor = processor
        self.crop_size = crop_size

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fname = self.file_ids[idx]
        fpath = os.path.join(self.npz_dir, fname)

        try:
            npz = np.load(fpath, allow_pickle=True)
            image = npz["sample"].astype(np.float32)  # (H, W, 5), 0-255
        except Exception as e:
            print(f"WARNING: skipping {fname}: {e}", file=sys.stderr)
            return fname, torch.zeros(5, 3, self.crop_size, self.crop_size)

        H, W, C = image.shape
        cs = self.crop_size

        # Center crop
        top = max((H - cs) // 2, 0)
        left = max((W - cs) // 2, 0)
        image = image[top : top + cs, left : left + cs, :]  # (cs, cs, 5)

        # Pad if image is smaller than crop_size (shouldn't happen for 520x696)
        if image.shape[0] < cs or image.shape[1] < cs:
            padded = np.zeros((cs, cs, C), dtype=np.float32)
            padded[: image.shape[0], : image.shape[1], :] = image
            image = padded

        # Per-channel -> RGB -> processor normalisation
        channels = []
        for c in range(C):
            ch = image[:, :, c]  # (cs, cs), float32, 0-255
            rgb = np.stack([ch, ch, ch], axis=2)  # (cs, cs, 3)
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)  # (3, cs, cs)
            processed = self.processor(rgb_tensor, return_tensors="pt", do_resize=False).pixel_values[0]
            channels.append(processed)

        # (5, 3, cs, cs)
        return fname, torch.stack(channels, dim=0)


def collate_sites(batch):
    fnames, tensors = zip(*batch)
    return list(fnames), torch.stack(tensors, dim=0)  # (B, 5, 3, H, W)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_and_pool(
    model: torch.nn.Module,
    batch_tensor: torch.Tensor,
    pool_size: int,
    grid_size: int = 16,
    device: str = "cuda",
    gpu_batch: int = 128,
):
    """Run DINOv2 on (B, 5, 3, H, W) and return pooled patch tokens.

    Returns: (B, 5, 1 + pool_size^2, 1536) float16
    """
    B, C_ch, C_rgb, H, W = batch_tensor.shape
    emb_dim = 1536
    n_spatial = pool_size * pool_size

    result = torch.zeros(B, C_ch, 1 + n_spatial, emb_dim, dtype=torch.float16)

    # Flatten to (B*5, 3, H, W) and process in GPU mini-batches
    flat = batch_tensor.reshape(B * C_ch, C_rgb, H, W)

    for start in range(0, flat.shape[0], gpu_batch):
        end = min(start + gpu_batch, flat.shape[0])
        mini = flat[start:end].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(pixel_values=mini).last_hidden_state  # (mb, 257, 1536)

        cls_tokens = out[:, 0, :]  # (mb, 1536)
        patch_tokens = out[:, 1:, :]  # (mb, 256, 1536)

        # Reshape patches to spatial grid and pool
        # (mb, 256, 1536) -> (mb, 1536, 16, 16) -> pool -> (mb, 1536, ps, ps)
        spatial = patch_tokens.reshape(-1, grid_size, grid_size, emb_dim)
        spatial = spatial.permute(0, 3, 1, 2)  # (mb, 1536, 16, 16)
        pooled = F.adaptive_avg_pool2d(spatial, (pool_size, pool_size))  # (mb, 1536, ps, ps)
        pooled = pooled.permute(0, 2, 3, 1).reshape(-1, n_spatial, emb_dim)  # (mb, ps^2, 1536)

        # Concat [CLS, spatial]
        combined = torch.cat([cls_tokens.unsqueeze(1), pooled], dim=1)  # (mb, 1+ps^2, 1536)

        result.view(B * C_ch, 1 + n_spatial, emb_dim)[start:end] = combined.cpu().half()

    return result  # (B, 5, 1+ps^2, 1536)


# ---------------------------------------------------------------------------
# H5 save
# ---------------------------------------------------------------------------

def save_batch(output_file: str, embeddings: np.ndarray, well_ids: list):
    """Append a batch to the H5 file."""
    with h5py.File(output_file, "a") as hf:
        batch_size = embeddings.shape[0]
        emb_shape = embeddings.shape[1:]

        if "embeddings" not in hf:
            hf.create_dataset(
                "embeddings",
                data=embeddings,
                maxshape=(None, *emb_shape),
                chunks=(min(64, batch_size), *emb_shape),
                dtype="float16",
            )
            hf.create_dataset(
                "well_id",
                data=np.array(well_ids, dtype=h5py.string_dtype(encoding="utf-8")),
                maxshape=(None,),
                chunks=True,
            )
        else:
            emb_ds = hf["embeddings"]
            id_ds = hf["well_id"]
            old = emb_ds.shape[0]
            emb_ds.resize(old + batch_size, axis=0)
            emb_ds[old : old + batch_size] = embeddings
            id_ds.resize(old + batch_size, axis=0)
            id_ds[old : old + batch_size] = np.array(
                well_ids, dtype=h5py.string_dtype(encoding="utf-8")
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract DINOv2-giant spatial patch tokens.")
    p.add_argument("--input_dir", type=str,
                   default="/data/huadi/cellpainting_data/cpg0012/npzs",
                   help="Directory containing site NPZ files.")
    p.add_argument("--output_dir", type=str,
                   default="/data/huadi/cellpainting_data/bray2017/img",
                   help="Directory to write the output H5 file.")
    p.add_argument("--output_file", type=str,
                   default="dinov2-giant_patch4x4.h5",
                   help="Output H5 filename.")
    p.add_argument("--pool_size", type=int, default=4,
                   help="Spatial pool target size (e.g. 4 -> 4x4=16 tokens per channel).")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Number of NPZ sites to load per DataLoader batch.")
    p.add_argument("--gpu_batch", type=int, default=64,
                   help="Max images per GPU mini-batch for DINOv2 forward pass.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto-detect).")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    print(f"[Info] Pool size: {args.pool_size} -> {args.pool_size**2} spatial tokens per channel")

    # Load DINOv2-giant
    print("[Info] Loading DINOv2-giant...")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    crop_size = processor.crop_size.get("height", 224) if hasattr(processor, "crop_size") else 224
    print(f"[Info] Crop size: {crop_size}")

    # Enumerate NPZ files
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".npz")])
    print(f"[Info] Found {len(all_files)} NPZ files in {args.input_dir}")

    # Resume: skip already-processed
    output_path = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(output_path):
        with h5py.File(output_path, "r") as hf:
            done_ids = set(wid.decode("utf-8") if isinstance(wid, bytes) else wid
                          for wid in hf["well_id"][:])
        remaining = [f for f in all_files if f not in done_ids]
        print(f"[Info] Resuming: {len(done_ids)} done, {len(remaining)} remaining")
    else:
        remaining = all_files
        print(f"[Info] Starting fresh: {len(remaining)} files to process")

    if len(remaining) == 0:
        print("[Info] All files already processed. Exiting.")
        return

    dataset = CellPaintingSiteDataset(
        npz_dir=args.input_dir,
        file_ids=remaining,
        processor=processor,
        crop_size=crop_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_sites,
        drop_last=False,
    )

    n_tokens = 1 + args.pool_size ** 2
    print(f"[Info] Output shape per site: (5, {n_tokens}, 1536)")
    print(f"[Info] Estimated total size: "
          f"{len(remaining) * 5 * n_tokens * 1536 * 2 / 1e9:.1f} GB (float16)")

    for fnames, batch_tensor in tqdm(loader, desc="Extracting"):
        embeddings = extract_and_pool(
            model=model,
            batch_tensor=batch_tensor,
            pool_size=args.pool_size,
            grid_size=crop_size // 14,
            device=device,
            gpu_batch=args.gpu_batch,
        )
        save_batch(output_path, embeddings.numpy(), fnames)

    # Final check
    with h5py.File(output_path, "r") as hf:
        print(f"\n[Done] Output: {output_path}")
        print(f"[Done] embeddings shape: {hf['embeddings'].shape}")
        print(f"[Done] well_id count: {hf['well_id'].shape[0]}")


if __name__ == "__main__":
    main()
