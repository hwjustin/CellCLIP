"""Re-extract patch tokens for sites missing from the patch H5.

Identifies sites present in the CLS H5 but absent from the patch H5,
writes them to a temp file list, and re-runs extraction with num_workers=0
to avoid silent DataLoader worker crashes.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def load_and_process_site(fpath, processor, crop_size=224):
    """Load a single NPZ and return (5, 3, crop_size, crop_size) tensor."""
    npz = np.load(fpath, allow_pickle=True)
    image = npz["sample"].astype(np.float32)  # (H, W, 5)
    H, W, C = image.shape
    cs = crop_size

    # Center crop
    top = max((H - cs) // 2, 0)
    left = max((W - cs) // 2, 0)
    image = image[top:top + cs, left:left + cs, :]

    if image.shape[0] < cs or image.shape[1] < cs:
        padded = np.zeros((cs, cs, C), dtype=np.float32)
        padded[:image.shape[0], :image.shape[1], :] = image
        image = padded

    channels = []
    for c in range(C):
        ch = image[:, :, c]
        rgb = np.stack([ch, ch, ch], axis=2)
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1)
        processed = processor(rgb_tensor, return_tensors="pt", do_resize=False).pixel_values[0]
        channels.append(processed)
    return torch.stack(channels, dim=0)  # (5, 3, cs, cs)


@torch.no_grad()
def extract_batch(model, batch_tensor, pool_size, grid_size, device, gpu_batch=64):
    B, C_ch, C_rgb, H, W = batch_tensor.shape
    emb_dim = 1536
    n_spatial = pool_size * pool_size
    result = torch.zeros(B, C_ch, 1 + n_spatial, emb_dim, dtype=torch.float16)

    flat = batch_tensor.reshape(B * C_ch, C_rgb, H, W)
    for start in range(0, flat.shape[0], gpu_batch):
        end = min(start + gpu_batch, flat.shape[0])
        mini = flat[start:end].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=mini).last_hidden_state
        cls_tokens = out[:, 0, :]
        patch_tokens = out[:, 1:, :]
        spatial = patch_tokens.reshape(-1, grid_size, grid_size, emb_dim)
        spatial = spatial.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(spatial, (pool_size, pool_size))
        pooled = pooled.permute(0, 2, 3, 1).reshape(-1, n_spatial, emb_dim)
        combined = torch.cat([cls_tokens.unsqueeze(1), pooled], dim=1)
        result.view(B * C_ch, 1 + n_spatial, emb_dim)[start:end] = combined.cpu().half()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", default="/data/huadi/cellpainting_data/cpg0012/npzs")
    parser.add_argument("--cls_h5", default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_ind.h5")
    parser.add_argument("--patch_h5", default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_patch4x4.h5")
    parser.add_argument("--pool_size", type=int, default=4)
    parser.add_argument("--gpu_batch", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Process this many files at a time (no DataLoader, sequential).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find missing sites
    with h5py.File(args.cls_h5, "r") as f:
        cls_ids = set(x.decode() if isinstance(x, bytes) else x for x in f["well_id"][:])
    with h5py.File(args.patch_h5, "r") as f:
        patch_ids = set(x.decode() if isinstance(x, bytes) else x for x in f["well_id"][:])

    missing = sorted(cls_ids - patch_ids)
    # Also check for files in npz_dir not in either H5
    npz_files = set(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    missing_from_npz = sorted(npz_files - patch_ids)
    # Merge: re-extract anything in cls_ids that's missing from patch
    missing = sorted(set(missing) & npz_files)  # only those that exist on disk

    print(f"[Info] CLS H5: {len(cls_ids)} | Patch H5: {len(patch_ids)} | Missing: {len(missing)}")
    if len(missing) == 0:
        print("[Info] Nothing to re-extract!")
        return

    # Load model
    print("[Info] Loading DINOv2-giant...")
    model = AutoModel.from_pretrained("facebook/dinov2-giant").to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
    crop_size = 224
    grid_size = crop_size // 14

    n_tokens = 1 + args.pool_size ** 2
    print(f"[Info] Re-extracting {len(missing)} sites (no DataLoader, sequential)")

    # Process in small batches, no DataLoader workers
    batch_names = []
    batch_tensors = []
    n_saved = 0
    n_failed = 0

    for i, fname in enumerate(tqdm(missing, desc="Re-extracting")):
        fpath = os.path.join(args.npz_dir, fname)
        try:
            tensor = load_and_process_site(fpath, processor, crop_size)
            batch_names.append(fname)
            batch_tensors.append(tensor)
        except Exception as e:
            print(f"  SKIP {fname}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        if len(batch_tensors) >= args.batch_size or i == len(missing) - 1:
            if len(batch_tensors) == 0:
                continue
            batch = torch.stack(batch_tensors, dim=0)
            embeddings = extract_batch(
                model, batch, args.pool_size, grid_size, device, args.gpu_batch
            )
            emb_np = embeddings.numpy()

            # Append to H5
            with h5py.File(args.patch_h5, "a") as hf:
                emb_ds = hf["embeddings"]
                id_ds = hf["well_id"]
                old = emb_ds.shape[0]
                bs = emb_np.shape[0]
                emb_ds.resize(old + bs, axis=0)
                emb_ds[old:old + bs] = emb_np
                id_ds.resize(old + bs, axis=0)
                id_ds[old:old + bs] = np.array(
                    batch_names, dtype=h5py.string_dtype(encoding="utf-8")
                )
            n_saved += len(batch_names)
            batch_names = []
            batch_tensors = []

    with h5py.File(args.patch_h5, "r") as f:
        print(f"\n[Done] Patch H5 now: {f['embeddings'].shape}")
    print(f"[Done] Saved: {n_saved}, Failed: {n_failed}")


if __name__ == "__main__":
    main()
