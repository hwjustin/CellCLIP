"""Train a 209-task bioactivity classifier on concatenated CellCLIP features.

Pipeline:
1) Extract frozen CellCLIP cell and perturbation embeddings for datasplit1.
2) Concatenate embeddings.
3) Train an FNN for multi-task binary classification with masked labels.
4) Report per-task and aggregate metrics on val/test.
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets import CellPainting
from src.helper import load


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concat Cell+Perturb FNN for 209-task bioactivity prediction."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="CellCLIP checkpoint path (.pt/.safetensors). If None, use HF default in src.helper.load callsite logic externally.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="cell_clip",
        help="CellCLIP model type for src.helper.load.",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1536,
        help="Input feature dim expected by CellCLIP visual tower.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cwcl",
        help="Loss type used for CellCLIP checkpoint config (affects model init in helper.load).",
    )
    parser.add_argument(
        "--image_h5",
        type=str,
        default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_ind.h5",
        help="H5 file with image embeddings used as CellCLIP image input.",
    )
    parser.add_argument(
        "--molecule_csv",
        type=str,
        default="/data/huadi/cellpainting_data/bray2017/mol/cell_long_captions_all.csv",
        help="Perturbation text CSV used by CellCLIP text encoder.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv",
        help="CSV with INCHIKEY + 209 assay labels.",
    )
    parser.add_argument("--extract_batch_size", type=int, default=256)
    parser.add_argument("--extract_num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--fc_units",
        type=int,
        default=2048,
        help="Hidden width for cellpaintssl head.",
    )
    parser.add_argument(
        "--cls_input_dropout",
        type=float,
        default=0.1,
        help="Input dropout before first FC layer in cellpaintssl head.",
    )
    parser.add_argument(
        "--cls_dropout",
        type=float,
        default=0.5,
        help="Dropout in intermediate FC blocks for cellpaintssl head.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--use_focal_loss", action="store_true", default=False)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/bioactivity_concat_fnn/cache",
        help="Cache split features as npz files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/bioactivity_concat_fnn",
        help="Directory to save model/metrics.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_samples_per_split",
        type=int,
        default=None,
        help="Optional debug cap per split after filtering.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LabelPack:
    label_cols: List[str]
    label_map: Dict[str, np.ndarray]


def load_labels(labels_csv: str) -> LabelPack:
    labels_df = pd.read_csv(labels_csv)
    if "INCHIKEY" not in labels_df.columns:
        raise ValueError(f"Expected INCHIKEY column in {labels_csv}")
    label_cols = [c for c in labels_df.columns if c != "INCHIKEY"]
    label_map = {
        row["INCHIKEY"]: row[label_cols].to_numpy(dtype=np.float32)
        for _, row in labels_df.iterrows()
    }
    return LabelPack(label_cols=label_cols, label_map=label_map)


def site_to_well_key(sample_key: str) -> str:
    """Convert PLATE-WELL-SITE(.npz) -> PLATE-WELL."""
    clean = sample_key.replace(".npz", "")
    if "-" not in clean:
        return clean
    return clean.rsplit("-", 1)[0]


class WellBagEmbeddingDataset(Dataset):
    """Build one sample per well, each as a bag of all site embeddings."""

    def __init__(
        self,
        split_csv: str,
        image_h5: str,
        molecule_csv: str,
        label_map: Dict[str, np.ndarray],
        max_samples: int = None,
    ):
        split_df = pd.read_csv(split_csv)
        if "SAMPLE_KEY" not in split_df.columns or "INCHIKEY" not in split_df.columns:
            raise ValueError(f"{split_csv} must include SAMPLE_KEY and INCHIKEY")

        base = CellPainting(
            sample_index_file=split_csv,
            mole_struc="text",
            context_length=512,
            image_directory_path=image_h5,
            molecule_path=molecule_csv,
            unique=False,
        )
        self.base = base
        key_to_idx = {k: i for i, k in enumerate(self.base.sample_keys)}

        well_to_site_keys: Dict[str, List[str]] = {}
        well_to_inchikey: Dict[str, str] = {}
        for _, row in split_df.iterrows():
            sample_key = str(row["SAMPLE_KEY"])
            inchikey = str(row["INCHIKEY"])
            if sample_key not in key_to_idx:
                continue
            if inchikey not in label_map:
                continue
            well_key = site_to_well_key(sample_key)
            well_to_site_keys.setdefault(well_key, []).append(sample_key)
            well_to_inchikey.setdefault(well_key, inchikey)

        ordered_wells = sorted(k for k, v in well_to_site_keys.items() if len(v) > 0)
        if max_samples is not None:
            ordered_wells = ordered_wells[:max_samples]

        if len(ordered_wells) == 0:
            raise ValueError(f"No valid samples for split: {split_csv}")
        self.key_to_idx = key_to_idx
        self.well_keys = ordered_wells
        self.well_to_site_keys = well_to_site_keys
        self.well_to_inchikey = well_to_inchikey
        self.label_map = label_map

    def __len__(self):
        return len(self.well_keys)

    def __getitem__(self, idx):
        well_key = self.well_keys[idx]
        site_keys = self.well_to_site_keys[well_key]
        inchikey = self.well_to_inchikey[well_key]

        imgs = []
        mol = None
        for sk in site_keys:
            base_idx = self.key_to_idx[sk]
            ((img, _extra_tokens), mol_i) = self.base[base_idx]
            imgs.append(torch.as_tensor(img, dtype=torch.float32))
            if mol is None:
                mol = mol_i

        bag_imgs = torch.stack(imgs, dim=0)  # (num_sites, C, D)
        y = torch.as_tensor(self.label_map[inchikey], dtype=torch.float32)
        return bag_imgs, mol, well_key, y


def collate_well_bags(batch):
    """Pad variable-site well bags to a dense tensor for MIL pooling."""
    bag_imgs, mols, keys, ys = zip(*batch)
    bsz = len(batch)
    max_sites = max(x.shape[0] for x in bag_imgs)
    channels = bag_imgs[0].shape[1]
    dim = bag_imgs[0].shape[2]

    padded = torch.zeros((bsz, max_sites, channels, dim), dtype=torch.float32)
    for i, x in enumerate(bag_imgs):
        padded[i, : x.shape[0]] = x

    mol_batch = {
        "input_ids": torch.stack([m["input_ids"] for m in mols], dim=0),
        "attention_mask": torch.stack([m["attention_mask"] for m in mols], dim=0),
    }
    y_batch = torch.stack(ys, dim=0)
    return padded, mol_batch, list(keys), y_batch


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class CellPaintSSLMLPHead(nn.Module):
    """CellPaintSSL-style deep MLP classifier on concatenated features."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        fc_units: int = 2048,
        cls_input_dropout: float = 0.1,
        cls_dropout: float = 0.5,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=cls_input_dropout),
            nn.Linear(input_dim, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(fc_units, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cls_dropout),
            nn.Linear(fc_units, out_dim),
        )
        self._init_parameters()

    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.classifier(x)


def masked_multitask_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
):
    mask = labels != -1
    targets = (labels == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if use_focal_loss:
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1.0, probs, 1.0 - probs)
        bce = ((1.0 - pt) ** focal_gamma) * bce
    masked = bce * mask.float()
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return masked.sum() / denom


def compute_metrics(logits: np.ndarray, labels: np.ndarray, label_cols: List[str]):
    probs = 1.0 / (1.0 + np.exp(-logits))
    n_tasks = labels.shape[1]
    per_task = []
    aucs, aps, f1s = [], [], []

    mask = labels != -1
    targets_global = (labels == 1).astype(np.int32)
    preds_global = (probs >= 0.5).astype(np.int32)
    global_acc = (
        float((preds_global[mask] == targets_global[mask]).mean()) if mask.any() else float("nan")
    )

    for i in range(n_tasks):
        valid = labels[:, i] != -1
        if valid.sum() == 0:
            continue
        y_true = (labels[valid, i] == 1).astype(np.int32)
        y_prob = probs[valid, i]
        y_pred = (y_prob >= 0.5).astype(np.int32)
        if np.unique(y_true).size < 2:
            continue

        auc = float(roc_auc_score(y_true, y_prob))
        ap = float(average_precision_score(y_true, y_prob))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        aucs.append(auc)
        aps.append(ap)
        f1s.append(f1)
        per_task.append(
            {
                "task": label_cols[i],
                "roc_auc": auc,
                "ap": ap,
                "f1": f1,
                "n_valid": int(valid.sum()),
                "n_pos": int(y_true.sum()),
                "n_neg": int((1 - y_true).sum()),
            }
        )

    def safe_mean_std(values: List[float]) -> Tuple[float, float]:
        if len(values) == 0:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    roc_mean, roc_std = safe_mean_std(aucs)
    ap_mean, ap_std = safe_mean_std(aps)
    f1_mean, f1_std = safe_mean_std(f1s)

    summary = {
        "n_tasks_total": n_tasks,
        "n_tasks_evaluated": len(aucs),
        "global_masked_accuracy": global_acc,
        "roc_auc_mean": roc_mean,
        "roc_auc_std": roc_std,
        "ap_mean": ap_mean,
        "ap_std": ap_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "roc_auc_gt_0.9": int(sum(v > 0.9 for v in aucs)),
        "roc_auc_gt_0.8": int(sum(v > 0.8 for v in aucs)),
        "roc_auc_gt_0.7": int(sum(v > 0.7 for v in aucs)),
    }
    return summary, per_task


def run_feature_extraction(
    split_name: str,
    split_csv: str,
    model,
    device: str,
    image_h5: str,
    molecule_csv: str,
    label_map: Dict[str, np.ndarray],
    cache_dir: str,
    batch_size: int,
    num_workers: int,
    max_samples: int,
):
    cache_path = os.path.join(cache_dir, f"{split_name}_well_features.npz")
    if os.path.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=True)
        return {
            "X_cell": arr["X_cell"],
            "X_pert": arr["X_pert"],
            "Y": arr["Y"],
            "keys": arr["keys"],
        }

    dataset = WellBagEmbeddingDataset(
        split_csv=split_csv,
        image_h5=image_h5,
        molecule_csv=molecule_csv,
        label_map=label_map,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_well_bags,
    )

    X_cell, X_pert, Y, keys = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, mol, batch_keys, batch_y in tqdm(loader, desc=f"Extract {split_name}"):
            imgs = imgs.to(device, non_blocking=True)
            mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
            bag_feats = model.encode_mil(imgs)
            cell_feats = model.encode_image(bag_feats)
            pert_feats = model.encode_text(mol)
            X_cell.append(cell_feats.detach().float().cpu().numpy())
            X_pert.append(pert_feats.detach().float().cpu().numpy())
            Y.append(batch_y.detach().cpu().numpy().astype(np.float32))
            keys.extend(batch_keys)

    out = {
        "X_cell": np.concatenate(X_cell, axis=0),
        "X_pert": np.concatenate(X_pert, axis=0),
        "Y": np.concatenate(Y, axis=0),
        "keys": np.asarray(keys),
    }
    np.savez(cache_path, **out)
    return out


def evaluate_model(model, loader, device, use_focal_loss, focal_gamma, label_cols):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(Xb)
            loss = masked_multitask_loss(
                logits, yb, use_focal_loss=use_focal_loss, focal_gamma=focal_gamma
            )
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    summary, per_task = compute_metrics(logits_np, labels_np, label_cols)
    summary["loss"] = float(total_loss / max(n_batches, 1))
    return summary, per_task, logits_np, labels_np


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    device = pick_device()
    print(f"[Info] Device: {device}")

    label_pack = load_labels(args.labels_csv)
    print(f"[Info] Loaded {len(label_pack.label_cols)} tasks from labels CSV.")

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = hf_hub_download("suinleelab/CellCLIP", "model.safetensors")

    model = load(
        model_path=ckpt_path,
        device=device,
        model_type=args.model_type,
        input_dim=args.input_dim,
        loss_type=args.loss_type,
    )

    train_feats = run_feature_extraction(
        split_name="train",
        split_csv=args.train_split,
        model=model,
        device=device,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        cache_dir=args.cache_dir,
        batch_size=args.extract_batch_size,
        num_workers=args.extract_num_workers,
        max_samples=args.max_samples_per_split,
    )
    val_feats = run_feature_extraction(
        split_name="val",
        split_csv=args.val_split,
        model=model,
        device=device,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        cache_dir=args.cache_dir,
        batch_size=args.extract_batch_size,
        num_workers=args.extract_num_workers,
        max_samples=args.max_samples_per_split,
    )
    test_feats = run_feature_extraction(
        split_name="test",
        split_csv=args.test_split,
        model=model,
        device=device,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        cache_dir=args.cache_dir,
        batch_size=args.extract_batch_size,
        num_workers=args.extract_num_workers,
        max_samples=args.max_samples_per_split,
    )

    X_train = np.concatenate([train_feats["X_cell"], train_feats["X_pert"]], axis=1)
    X_val = np.concatenate([val_feats["X_cell"], val_feats["X_pert"]], axis=1)
    X_test = np.concatenate([test_feats["X_cell"], test_feats["X_pert"]], axis=1)

    y_train = train_feats["Y"]
    y_val = val_feats["Y"]
    y_test = test_feats["Y"]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    train_loader = DataLoader(
        NumpyDataset(X_train, y_train),
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        NumpyDataset(X_val, y_val),
        batch_size=args.train_batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        NumpyDataset(X_test, y_test),
        batch_size=args.train_batch_size,
        shuffle=False,
        drop_last=False,
    )

    fnn = CellPaintSSLMLPHead(
        input_dim=int(X_train.shape[1]),
        out_dim=y_train.shape[1],
        fc_units=args.fc_units,
        cls_input_dropout=args.cls_input_dropout,
        cls_dropout=args.cls_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(fnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        fnn.train()
        running_loss = 0.0
        n_batches = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = fnn(Xb)
            loss = masked_multitask_loss(
                logits,
                yb,
                use_focal_loss=args.use_focal_loss,
                focal_gamma=args.focal_gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_summary, _, _, _ = evaluate_model(
            model=fnn,
            loader=val_loader,
            device=device,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            label_cols=label_pack.label_cols,
        )
        val_auc = val_summary["roc_auc_mean"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_summary["loss"],
            "val_roc_auc_mean": val_summary["roc_auc_mean"],
            "val_ap_mean": val_summary["ap_mean"],
            "val_f1_mean": val_summary["f1_mean"],
            "val_n_tasks_evaluated": val_summary["n_tasks_evaluated"],
        }
        history.append(record)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"val_roc_auc={val_summary['roc_auc_mean']:.4f} "
            f"val_ap={val_summary['ap_mean']:.4f}"
        )

        if np.isnan(val_auc):
            pass
        elif val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(fnn.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"[Info] Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        best_state = copy.deepcopy(fnn.state_dict())
        best_epoch = len(history)

    fnn.load_state_dict(best_state)
    val_summary, val_per_task, _, _ = evaluate_model(
        model=fnn,
        loader=val_loader,
        device=device,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_cols=label_pack.label_cols,
    )
    test_summary, test_per_task, _, _ = evaluate_model(
        model=fnn,
        loader=test_loader,
        device=device,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_cols=label_pack.label_cols,
    )

    ckpt_path = os.path.join(args.out_dir, "best_fnn.pt")
    torch.save(
        {
            "state_dict": fnn.state_dict(),
            "input_dim": int(X_train.shape[1]),
            "n_tasks": int(y_train.shape[1]),
            "fc_units": args.fc_units,
            "cls_input_dropout": args.cls_input_dropout,
            "cls_dropout": args.cls_dropout,
            "best_epoch": best_epoch,
            "feature_mean": mean.astype(np.float32),
            "feature_std": std.astype(np.float32),
            "label_cols": label_pack.label_cols,
            "args": vars(args),
        },
        ckpt_path,
    )

    metrics = {
        "best_epoch": best_epoch,
        "history": history,
        "val": val_summary,
        "test": test_summary,
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    val_df = pd.DataFrame(val_per_task)
    test_df = pd.DataFrame(test_per_task)
    val_tasks_path = os.path.join(args.out_dir, "val_per_task.csv")
    test_tasks_path = os.path.join(args.out_dir, "test_per_task.csv")
    val_df.to_csv(val_tasks_path, index=False)
    test_df.to_csv(test_tasks_path, index=False)

    print("\n[Result] Validation summary:")
    print(json.dumps(val_summary, indent=2))
    print("\n[Result] Test summary:")
    print(json.dumps(test_summary, indent=2))
    print(f"\n[Saved] ckpt={ckpt_path}")
    print(f"[Saved] metrics={metrics_path}")
    print(f"[Saved] val_per_task={val_tasks_path}")
    print(f"[Saved] test_per_task={test_tasks_path}")


if __name__ == "__main__":
    main()
