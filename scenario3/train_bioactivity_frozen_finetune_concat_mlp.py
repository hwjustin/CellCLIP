"""Train a 209-task bioactivity classifier by concatenating frozen + fine-tuned features.

Scenario 8 pipeline:
1) Load a frozen CellCLIP checkpoint branch and a trainable CellCLIP checkpoint branch.
2) Build per-well bags of site embeddings and perturbation text tokens.
3) Encode each sample with both branches and concatenate:
   [frozen_cell || frozen_text || finetune_cell || finetune_text].
4) Train only the fine-tuned branch + MLP head with masked multi-task loss.
5) Report per-task and aggregate metrics on val/test.
"""

import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm

from scenario2_5.train_bioactivity_preconcat_mlp_finetune import (
    ProjectionMLP,
    WellBagEmbeddingDataset,
    collate_well_bags,
    compute_metrics,
    load_labels,
    masked_multitask_loss,
    pick_device,
    set_seed,
)
from src.helper import load


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scenario 8: concatenate frozen CellCLIP features with fine-tuned CellCLIP "
            "features for 209-task bioactivity prediction."
        )
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Trainable CellCLIP checkpoint path (.pt/.safetensors). If None, use HF default.",
    )
    parser.add_argument(
        "--frozen_ckpt_path",
        type=str,
        default=None,
        help="Frozen branch checkpoint path. If None, reuse --ckpt_path.",
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
        help="Used only for CellCLIP init compatibility in helper.load.",
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
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument(
        "--text_lr",
        type=float,
        default=None,
        help="Optional override LR for fine-tuned text tower (`text.*`, `text_proj.*`).",
    )
    parser.add_argument(
        "--proj_lr",
        type=float,
        default=None,
        help="Optional LR override for learnable projection heads. Defaults to head_lr.",
    )
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Set fine-tuned branch backbone LR to 0 for first N epochs.",
    )
    parser.add_argument(
        "--enable_proj_mlp",
        action="store_true",
        default=False,
        help=(
            "Use learnable projection MLPs on fine-tuned cell/text features before "
            "classification and orthogonality loss. If disabled, uses identity projection."
        ),
    )
    parser.add_argument(
        "--proj_hidden_dim",
        type=int,
        default=None,
        help="Hidden width inside ProjectionMLP (defaults to input_dim).",
    )
    parser.add_argument(
        "--proj_num_layers",
        type=int,
        default=2,
        choices=[1, 2],
        help="ProjectionMLP linear layer count: 1 (linear) or 2 (linear-relu-linear).",
    )
    parser.add_argument(
        "--proj_dropout",
        type=float,
        default=0.0,
        help="Dropout applied to projection output when --enable_proj_mlp is set.",
    )
    parser.add_argument(
        "--residual_projection",
        action="store_true",
        default=False,
        help="Use residual projection (x + MLP(x)) when input_dim == output_dim.",
    )
    parser.add_argument(
        "--proj_identity_init",
        action="store_true",
        default=False,
        help="Initialize learnable projection as identity (requires residual form).",
    )
    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=0.0,
        help="Final weight for orthogonality loss between projected fine-tuned and frozen features.",
    )
    parser.add_argument(
        "--fc_units",
        type=int,
        default=2048,
        help="Hidden width for CellPaintSSL-style head.",
    )
    parser.add_argument(
        "--cls_num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden FC blocks in classifier head.",
    )
    parser.add_argument(
        "--cls_input_dropout",
        type=float,
        default=0.1,
        help="Input dropout before first FC layer in classifier head.",
    )
    parser.add_argument(
        "--cls_dropout",
        type=float,
        default=0.5,
        help="Dropout in intermediate FC blocks for classifier head.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--use_focal_loss", action="store_true", default=False)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/bioactivity_frozen_finetune_concat_mlp",
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class CellPaintSSLMLPHead(nn.Module):
    """Fixed classifier MLP on concatenated features."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int = 209,
    ):
        super().__init__()
        if out_dim != 209:
            raise ValueError(f"CellPaintSSLMLPHead expects out_dim=209, got {out_dim}")
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 209),
        )
        self._init_parameters()

    def _init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class FrozenFinetuneConcatModel(nn.Module):
    def __init__(
        self,
        frozen_backbone: nn.Module,
        finetune_backbone: nn.Module,
        cell_proj: nn.Module,
        text_proj: nn.Module,
        classifier: nn.Module,
        concat_dim: int,
    ):
        super().__init__()
        self.frozen_backbone = frozen_backbone
        self.finetune_backbone = finetune_backbone
        self.cell_proj = cell_proj
        self.text_proj = text_proj
        self.classifier = classifier
        self.frozen_backbone.eval()
        for p in self.frozen_backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen branch deterministic (no dropout/bn updates).
        self.frozen_backbone.eval()
        return self

    def forward(
        self, imgs: torch.Tensor, mol: Dict[str, torch.Tensor], return_aux: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            frozen_bag = self.frozen_backbone.encode_mil(imgs)
            frozen_cell = self.frozen_backbone.encode_image(frozen_bag)
            frozen_text = self.frozen_backbone.encode_text(mol)

        finetune_bag = self.finetune_backbone.encode_mil(imgs)
        finetune_cell_raw = self.finetune_backbone.encode_image(finetune_bag)
        finetune_text_raw = self.finetune_backbone.encode_text(mol)
        finetune_cell = self.cell_proj(finetune_cell_raw)
        finetune_text = self.text_proj(finetune_text_raw)

        feats = torch.cat([frozen_cell, frozen_text, finetune_cell, finetune_text], dim=1)
        logits = self.classifier(feats)
        if not return_aux:
            return logits

        ortho_cell = orthogonality_loss(finetune_cell, frozen_cell)
        ortho_text = orthogonality_loss(finetune_text, frozen_text)
        ortho = 0.5 * (ortho_cell + ortho_text)
        return logits, {
            "ortho_loss": ortho,
            "ortho_cell_loss": ortho_cell,
            "ortho_text_loss": ortho_text,
        }


def infer_branch_dims(backbone, loader, device: str) -> Tuple[int, int]:
    backbone.eval()
    imgs, mol, _keys, _y = next(iter(loader))
    imgs = imgs.to(device, non_blocking=True)
    mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
    with torch.no_grad():
        bag_feats = backbone.encode_mil(imgs)
        cell_feats = backbone.encode_image(bag_feats)
        text_feats = backbone.encode_text(mol)
    return int(cell_feats.shape[1]), int(text_feats.shape[1])


def orthogonality_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared cosine similarity penalty; 0 means orthogonal on average."""
    x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
    y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
    cosine = (x_norm * y_norm).sum(dim=1)
    return (cosine**2).mean()


def evaluate_model(model, loader, device, use_focal_loss, focal_gamma, label_cols):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    total_ortho_loss = 0.0
    total_ortho_cell_loss = 0.0
    total_ortho_text_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, mol, _keys, yb in loader:
            imgs = imgs.to(device, non_blocking=True)
            mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
            yb = yb.to(device, non_blocking=True)
            logits, aux = model(imgs, mol, return_aux=True)
            loss = masked_multitask_loss(
                logits, yb, use_focal_loss=use_focal_loss, focal_gamma=focal_gamma
            )
            total_loss += loss.item()
            total_ortho_loss += aux["ortho_loss"].item()
            total_ortho_cell_loss += aux["ortho_cell_loss"].item()
            total_ortho_text_loss += aux["ortho_text_loss"].item()
            n_batches += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    summary, per_task = compute_metrics(logits_np, labels_np, label_cols)
    summary["loss"] = float(total_loss / max(n_batches, 1))
    summary["ortho_loss"] = float(total_ortho_loss / max(n_batches, 1))
    summary["ortho_cell_loss"] = float(total_ortho_cell_loss / max(n_batches, 1))
    summary["ortho_text_loss"] = float(total_ortho_text_loss / max(n_batches, 1))
    return summary, per_task, logits_np, labels_np


def build_param_groups(
    model: FrozenFinetuneConcatModel,
    backbone_lr: float,
    text_lr: Optional[float],
    proj_lr: Optional[float],
    head_lr: float,
    weight_decay: float,
):
    vision_params, text_params, proj_params, head_params = [], [], [], []
    excluded = []
    group_kinds = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("finetune_backbone."):
            back_name = name[len("finetune_backbone.") :]
            if back_name in {"logit_scale", "bias"}:
                param.requires_grad = False
                excluded.append(back_name)
                continue
            if back_name.startswith("text.") or back_name.startswith("text_proj."):
                text_params.append(param)
            else:
                vision_params.append(param)
        elif name.startswith("cell_proj.") or name.startswith("text_proj."):
            proj_params.append(param)
        elif name.startswith("classifier."):
            head_params.append(param)

    groups = []
    if len(vision_params) > 0:
        groups.append({"params": vision_params, "lr": backbone_lr, "weight_decay": weight_decay})
        group_kinds.append("vision")
    if len(text_params) > 0:
        groups.append(
            {
                "params": text_params,
                "lr": text_lr if text_lr is not None else backbone_lr,
                "weight_decay": weight_decay,
            }
        )
        group_kinds.append("text")
    if len(proj_params) > 0:
        groups.append(
            {
                "params": proj_params,
                "lr": proj_lr if proj_lr is not None else head_lr,
                "weight_decay": weight_decay,
            }
        )
        group_kinds.append("proj")
    if len(head_params) > 0:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        group_kinds.append("head")
    return groups, excluded, group_kinds


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = pick_device()
    use_amp = args.amp and device == "cuda"
    print(f"[Info] Device: {device} | AMP: {use_amp}")

    label_pack = load_labels(args.labels_csv)
    print(f"[Info] Loaded {len(label_pack.label_cols)} tasks from labels CSV.")

    train_dataset = WellBagEmbeddingDataset(
        split_csv=args.train_split,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        max_samples=args.max_samples_per_split,
    )
    val_dataset = WellBagEmbeddingDataset(
        split_csv=args.val_split,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        max_samples=args.max_samples_per_split,
    )
    test_dataset = WellBagEmbeddingDataset(
        split_csv=args.test_split,
        image_h5=args.image_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        max_samples=args.max_samples_per_split,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_well_bags,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_well_bags,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_well_bags,
    )

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path = hf_hub_download("suinleelab/CellCLIP", "model.safetensors")
    frozen_ckpt_path = args.frozen_ckpt_path if args.frozen_ckpt_path is not None else ckpt_path

    finetune_backbone = load(
        model_path=ckpt_path,
        device=device,
        model_type=args.model_type,
        input_dim=args.input_dim,
        loss_type=args.loss_type,
    )
    frozen_backbone = load(
        model_path=frozen_ckpt_path,
        device=device,
        model_type=args.model_type,
        input_dim=args.input_dim,
        loss_type=args.loss_type,
    )

    cell_dim_ft, text_dim_ft = infer_branch_dims(
        backbone=finetune_backbone, loader=train_loader, device=device
    )
    cell_dim_frz, text_dim_frz = infer_branch_dims(
        backbone=frozen_backbone, loader=train_loader, device=device
    )
    if (cell_dim_ft, text_dim_ft) != (cell_dim_frz, text_dim_frz):
        raise ValueError(
            "Frozen and fine-tuned branch dims do not match. "
            f"finetune=({cell_dim_ft}, {text_dim_ft}), frozen=({cell_dim_frz}, {text_dim_frz})"
        )
    concat_dim = int(2 * (cell_dim_ft + text_dim_ft))
    print(
        f"[Info] Branch dims: cell={cell_dim_ft}, text={text_dim_ft} | "
        f"concat_dim={concat_dim}"
    )

    classifier = CellPaintSSLMLPHead(
        input_dim=concat_dim,
        out_dim=len(label_pack.label_cols),
    )
    if args.enable_proj_mlp:
        cell_proj = ProjectionMLP(
            input_dim=cell_dim_ft,
            output_dim=cell_dim_ft,
            hidden_dim=args.proj_hidden_dim,
            num_layers=args.proj_num_layers,
            proj_dropout=args.proj_dropout,
            residual=args.residual_projection,
            identity_init=args.proj_identity_init,
        )
        text_proj = ProjectionMLP(
            input_dim=text_dim_ft,
            output_dim=text_dim_ft,
            hidden_dim=args.proj_hidden_dim,
            num_layers=args.proj_num_layers,
            proj_dropout=args.proj_dropout,
            residual=args.residual_projection,
            identity_init=args.proj_identity_init,
        )
    else:
        cell_proj = nn.Identity()
        text_proj = nn.Identity()

    model = FrozenFinetuneConcatModel(
        frozen_backbone=frozen_backbone,
        finetune_backbone=finetune_backbone,
        cell_proj=cell_proj,
        text_proj=text_proj,
        classifier=classifier,
        concat_dim=concat_dim,
    ).to(device)

    param_groups, excluded_names, group_kinds = build_param_groups(
        model=model,
        backbone_lr=args.backbone_lr,
        text_lr=args.text_lr,
        proj_lr=args.proj_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if len(excluded_names) > 0:
        print(f"[Info] Excluded params from optimization: {excluded_names}")
    if args.freeze_backbone_epochs > 0:
        print(f"[Info] Freeze fine-tuned branch for first {args.freeze_backbone_epochs} epochs.")
    if args.enable_proj_mlp:
        proj_lr_print = args.proj_lr if args.proj_lr is not None else args.head_lr
        print(
            "[Info] Learnable projections enabled: "
            f"num_layers={args.proj_num_layers}, proj_lr={proj_lr_print:.2e}"
        )
    if args.ortho_weight > 0:
        print(f"[Info] Ortho loss enabled from epoch 1: weight={args.ortho_weight}")

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        backbone_lr_now = args.backbone_lr if epoch > args.freeze_backbone_epochs else 0.0
        text_lr_base = args.text_lr if args.text_lr is not None else args.backbone_lr
        text_lr_now = text_lr_base if epoch > args.freeze_backbone_epochs else 0.0
        proj_lr_now = args.proj_lr if args.proj_lr is not None else args.head_lr
        ortho_weight_now = float(args.ortho_weight)

        for group, kind in zip(optimizer.param_groups, group_kinds):
            if kind == "vision":
                group["lr"] = backbone_lr_now
            elif kind == "text":
                group["lr"] = text_lr_now
            elif kind == "proj":
                group["lr"] = proj_lr_now
            else:
                group["lr"] = args.head_lr

        model.train()
        running_loss = 0.0
        running_task_loss = 0.0
        running_ortho_loss = 0.0
        n_batches = 0

        for imgs, mol, _keys, yb in tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, aux = model(imgs, mol, return_aux=True)
                task_loss = masked_multitask_loss(
                    logits,
                    yb,
                    use_focal_loss=args.use_focal_loss,
                    focal_gamma=args.focal_gamma,
                )
                ortho_loss = aux["ortho_loss"]
                loss = task_loss + ortho_weight_now * ortho_loss

            if use_amp:
                scaler.scale(loss).backward()
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

            running_loss += loss.item()
            running_task_loss += task_loss.item()
            running_ortho_loss += ortho_loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_task_loss = running_task_loss / max(n_batches, 1)
        train_ortho_loss = running_ortho_loss / max(n_batches, 1)
        val_summary, _, _, _ = evaluate_model(
            model=model,
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
            "train_task_loss": train_task_loss,
            "train_ortho_loss": train_ortho_loss,
            "ortho_weight_now": ortho_weight_now,
            "backbone_lr_now": backbone_lr_now,
            "text_lr_now": text_lr_now,
            "proj_lr_now": proj_lr_now,
            "head_lr_now": args.head_lr,
            "val_loss": val_summary["loss"],
            "val_ortho_loss": val_summary["ortho_loss"],
            "val_roc_auc_mean": val_summary["roc_auc_mean"],
            "val_ap_mean": val_summary["ap_mean"],
            "val_f1_mean": val_summary["f1_mean"],
            "val_n_tasks_evaluated": val_summary["n_tasks_evaluated"],
        }
        history.append(record)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"task_loss={train_task_loss:.6f} "
            f"ortho={train_ortho_loss:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"val_roc_auc={val_summary['roc_auc_mean']:.4f} "
            f"val_ap={val_summary['ap_mean']:.4f}"
        )

        if np.isnan(val_auc):
            pass
        elif val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"[Info] Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)

    model.load_state_dict(best_state)
    val_summary, val_per_task, _, _ = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_cols=label_pack.label_cols,
    )
    test_summary, test_per_task, _, _ = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        label_cols=label_pack.label_cols,
    )

    ckpt_out_path = os.path.join(args.out_dir, "best_frozen_finetune_concat_joint_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "frozen_backbone_state_dict": model.frozen_backbone.state_dict(),
            "finetune_backbone_state_dict": model.finetune_backbone.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "cell_feature_dim": cell_dim_ft,
            "text_feature_dim": text_dim_ft,
            "concat_dim": concat_dim,
            "enable_proj_mlp": args.enable_proj_mlp,
            "proj_hidden_dim": args.proj_hidden_dim,
            "proj_num_layers": args.proj_num_layers,
            "proj_dropout": args.proj_dropout,
            "residual_projection": args.residual_projection,
            "proj_identity_init": args.proj_identity_init,
            "proj_lr": args.proj_lr,
            "ortho_weight": args.ortho_weight,
            "n_tasks": int(len(label_pack.label_cols)),
            "classifier_arch": "LayerNorm(input_dim) -> Dropout(0.1) -> Linear(input_dim,1024) -> ReLU -> Dropout(0.1) -> Linear(1024,512) -> ReLU -> Dropout(0.1) -> Linear(512,209)",
            "best_epoch": best_epoch,
            "label_cols": label_pack.label_cols,
            "excluded_optimizer_params": excluded_names,
            "args": vars(args),
        },
        ckpt_out_path,
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
    print(f"\n[Saved] ckpt={ckpt_out_path}")
    print(f"[Saved] metrics={metrics_path}")
    print(f"[Saved] val_per_task={val_tasks_path}")
    print(f"[Saved] test_per_task={test_tasks_path}")


if __name__ == "__main__":
    main()
