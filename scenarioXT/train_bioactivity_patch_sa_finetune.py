"""ScenarioXT: Native ViT patch tokens + self-attention for 209-task bioactivity.

Unlike ScenarioX (which used 6 summary tokens from CrossChannelFormer),
ScenarioXT feeds **native DINOv2 spatial patch tokens** directly into the
self-attention head.  CrossChannelFormer is bypassed entirely — the DINOv2
features are frozen/pre-extracted, and only the SA head + BERT text encoder
are trainable (MultiTab-Net style).

Pipeline:
1) Load pre-extracted DINOv2 patch tokens H5: (N, 5, 1+P, 1536)
   where P = pool_size^2 spatial tokens, index 0 = CLS.
2) MIL pool across sites per well (per spatial position).
3) Project image tokens (1536 → sa_hidden_dim).
4) Add channel embeddings + spatial position embeddings.
5) Encode text with pretrained BERT → full token sequence.
6) Self-attention over [K task tokens | N_img image tokens | L text tokens].
7) Per-task MLP towers → logits.
"""

import argparse
import copy
import json
import math
import os
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from scenario2.train_bioactivity_joint_finetune import (
    compute_metrics,
    load_labels,
    masked_multitask_loss,
    pick_device,
    set_seed,
    site_to_well_key,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="ScenarioXT: native DINOv2 patch tokens + self-attention."
    )

    # data
    p.add_argument("--patch_h5", type=str,
                   default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_patch4x4.h5",
                   help="H5 with patch token embeddings (N, 5, 1+P, 1536).")
    p.add_argument("--cls_h5", type=str,
                   default="/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_ind.h5",
                   help="Original CLS-only H5 used only for well_id → index mapping "
                        "and as fallback if patch H5 misses samples.")
    p.add_argument("--molecule_csv", type=str,
                   default="/data/huadi/cellpainting_data/bray2017/mol/cell_long_captions_all.csv")
    p.add_argument("--train_split", type=str,
                   default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-train.csv")
    p.add_argument("--val_split", type=str,
                   default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-val.csv")
    p.add_argument("--test_split", type=str,
                   default="/data/huadi/cellpainting_data/cpg0012/splits/datasplit1-test.csv")
    p.add_argument("--labels_csv", type=str,
                   default="/data/huadi/cellpainting_data/cpg0012/labels/compound_assay_activity.csv")

    # image token config
    p.add_argument("--use_cls_token", action="store_true", default=True,
                   help="Include DINOv2 CLS token (index 0) in image tokens.")
    p.add_argument("--no_cls_token", dest="use_cls_token", action="store_false")
    p.add_argument("--img_emb_dim", type=int, default=1536,
                   help="DINOv2 embedding dimension.")

    # text encoder
    p.add_argument("--bert_model", type=str, default="bert-base-cased")
    p.add_argument("--text_proj_dim", type=int, default=None,
                   help="Project BERT tokens to this dim before SA. None = use sa_hidden_dim.")
    p.add_argument("--max_text_len", type=int, default=128)

    # MIL pooling
    p.add_argument("--mil_pooling", type=str, default="attention",
                   choices=["attention", "mean"],
                   help="MIL pooling strategy across sites.")
    p.add_argument("--mil_hidden_dim", type=int, default=128)

    # self-attention head
    p.add_argument("--sa_hidden_dim", type=int, default=256,
                   help="Operating dimension for the self-attention head.")
    p.add_argument("--sa_num_layers", type=int, default=2)
    p.add_argument("--sa_num_heads", type=int, default=8)
    p.add_argument("--sa_ff_dim", type=int, default=1024)
    p.add_argument("--sa_dropout", type=float, default=0.1)
    p.add_argument("--task_token_dim", type=int, default=None,
                   help="Intrinsic task token dim (projected to sa_hidden_dim). None = sa_hidden_dim.")
    p.add_argument("--tower_hidden_dim", type=int, default=32)
    p.add_argument("--available_tasks_only", action="store_true", default=True,
                   help="Route B: only include K available task tokens.")
    p.add_argument("--all_tasks", dest="available_tasks_only", action="store_false")

    # training
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--text_lr", type=float, default=5e-6,
                   help="LR for BERT encoder (finetuned).")
    p.add_argument("--head_lr", type=float, default=5e-4,
                   help="LR for SA head + projections + MIL pooling.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--use_focal_loss", action="store_true", default=False)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # LR scheduler
    p.add_argument("--lr_schedule", type=str, default="none",
                   choices=["none", "cosine", "linear"],
                   help="LR schedule: none (constant), cosine (cosine anneal), linear (linear decay).")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Fraction of total steps for linear warmup (used with cosine/linear).")
    p.add_argument("--min_lr_ratio", type=float, default=0.01,
                   help="Minimum LR as fraction of initial LR (for cosine/linear end).")

    # misc
    p.add_argument("--out_dir", type=str, default="results/scenarioXT")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples_per_split", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset: loads patch tokens from H5
# ---------------------------------------------------------------------------

class PatchWellBagDataset(Dataset):
    """Build one sample per well from pre-extracted DINOv2 patch tokens."""

    def __init__(
        self,
        split_csv: str,
        patch_h5_path: str,
        molecule_csv: str,
        label_map: Dict[str, np.ndarray],
        max_text_len: int = 128,
        bert_model: str = "bert-base-cased",
        max_samples: Optional[int] = None,
    ):
        split_df = pd.read_csv(split_csv)
        if "SAMPLE_KEY" not in split_df.columns or "INCHIKEY" not in split_df.columns:
            raise ValueError(f"{split_csv} must include SAMPLE_KEY and INCHIKEY")

        # Load molecule text — CSV maps sample_key (ID column) → prompt text
        mol_df = pd.read_csv(molecule_csv)
        self.mol_text = dict(zip(mol_df.iloc[:, 0].astype(str), mol_df.iloc[:, 1].astype(str)))

        # Load patch H5
        self.h5_path = patch_h5_path
        self.h5_file = h5py.File(patch_h5_path, "r", swmr=True)
        h5_ids_raw = [
            wid.decode("utf-8") if isinstance(wid, bytes) else wid
            for wid in self.h5_file["well_id"][:]
        ]
        # Build lookup mapping both with and without .npz suffix
        self.h5_id_to_idx = {}
        for i, wid in enumerate(h5_ids_raw):
            self.h5_id_to_idx[wid] = i
            # Also map stripped version (without .npz)
            stripped = wid.replace(".npz", "") if wid.endswith(".npz") else wid
            self.h5_id_to_idx[stripped] = i

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.max_text_len = max_text_len

        # Build well → sites mapping
        well_to_site_keys: Dict[str, List[str]] = {}
        well_to_inchikey: Dict[str, str] = {}
        for _, row in split_df.iterrows():
            sample_key = str(row["SAMPLE_KEY"])
            inchikey = str(row["INCHIKEY"])
            if sample_key not in self.h5_id_to_idx:
                continue
            if inchikey not in label_map:
                continue
            if sample_key not in self.mol_text:
                continue
            well_key = site_to_well_key(sample_key)
            well_to_site_keys.setdefault(well_key, []).append(sample_key)
            well_to_inchikey.setdefault(well_key, inchikey)

        self.well_keys = sorted(k for k, v in well_to_site_keys.items() if len(v) > 0)
        if max_samples is not None:
            self.well_keys = self.well_keys[:max_samples]
        if len(self.well_keys) == 0:
            raise ValueError(f"No valid samples for split: {split_csv}")

        self.well_to_site_keys = well_to_site_keys
        self.well_to_inchikey = well_to_inchikey
        self.label_map = label_map
        print(f"  [{split_csv}] {len(self.well_keys)} wells")

    def __len__(self):
        return len(self.well_keys)

    def __getitem__(self, idx):
        well_key = self.well_keys[idx]
        site_keys = self.well_to_site_keys[well_key]
        inchikey = self.well_to_inchikey[well_key]

        # Load patch tokens for each site
        site_embs = []
        for sk in site_keys:
            h5_idx = self.h5_id_to_idx[sk]
            emb = self.h5_file["embeddings"][h5_idx]  # (5, 1+P, 1536)
            site_embs.append(torch.as_tensor(emb, dtype=torch.float32))
        bag_imgs = torch.stack(site_embs, dim=0)  # (num_sites, 5, 1+P, 1536)

        # Tokenize text (mol_text is keyed by sample_key, use first site key)
        text = str(self.mol_text[site_keys[0]])
        tok = self.tokenizer(
            text, max_length=self.max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        mol = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }

        y = torch.as_tensor(self.label_map[inchikey], dtype=torch.float32)
        return bag_imgs, mol, well_key, y


def collate_patch_well_bags(batch):
    """Pad variable-site well bags for patch token MIL pooling."""
    bag_imgs, mols, keys, ys = zip(*batch)
    bsz = len(batch)
    max_sites = max(x.shape[0] for x in bag_imgs)
    channels = bag_imgs[0].shape[1]
    n_tokens = bag_imgs[0].shape[2]
    dim = bag_imgs[0].shape[3]

    padded = torch.zeros((bsz, max_sites, channels, n_tokens, dim), dtype=torch.float32)
    for i, x in enumerate(bag_imgs):
        padded[i, : x.shape[0]] = x

    mol_batch = {
        "input_ids": torch.stack([m["input_ids"] for m in mols], dim=0),
        "attention_mask": torch.stack([m["attention_mask"] for m in mols], dim=0),
    }
    y_batch = torch.stack(ys, dim=0)
    return padded, mol_batch, list(keys), y_batch


# ---------------------------------------------------------------------------
# MIL Pooling for spatial tokens
# ---------------------------------------------------------------------------

class SpatialMILPooling(nn.Module):
    """MIL pooling across sites, independently per channel and spatial position.

    Input:  (B, M, C, T, D)  — M sites, C channels, T tokens per channel, D dim
    Output: (B, C, T, D)     — pooled across sites
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, pooling: str = "attention"):
        super().__init__()
        self.pooling = pooling
        if pooling == "attention":
            self.V = nn.Linear(input_dim, hidden_dim)
            self.U = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, C, T, D = x.shape

        if self.pooling == "mean":
            mask = (x.abs().sum(dim=-1, keepdim=True) > 0).float()  # (B, M, C, T, 1)
            denom = mask.sum(dim=1).clamp(min=1.0)  # (B, C, T, 1)
            return (x * mask).sum(dim=1) / denom

        # Attention pooling: flatten C*T, pool over M
        x_flat = x.reshape(B, M, C * T, D)  # (B, M, C*T, D)
        x_flat = x_flat.permute(0, 2, 1, 3).reshape(B * C * T, M, D)  # (B*C*T, M, D)

        h_V = torch.tanh(self.V(x_flat))
        h_U = torch.sigmoid(self.U(x_flat))
        h = h_V * h_U
        attn = self.attention(h)  # (B*C*T, M, 1)

        mask = (x_flat.abs().sum(dim=-1) > 0).float()  # (B*C*T, M)
        attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        attn = torch.softmax(attn, dim=1)

        pooled = (attn * x_flat).sum(dim=1)  # (B*C*T, D)
        return pooled.reshape(B, C, T, D)


# ---------------------------------------------------------------------------
# Text encoder (BERT, finetunable)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """BERT text encoder returning per-token representations."""

    def __init__(self, bert_model: str = "bert-base-cased", proj_dim: int = 256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.proj = nn.Linear(self.bert.config.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.proj(out.last_hidden_state)  # (B, L, proj_dim)
        return tokens, attention_mask


# ---------------------------------------------------------------------------
# Self-Attention Head (adapted from ScenarioX, takes variable image tokens)
# ---------------------------------------------------------------------------

class PatchSelfAttentionHead(nn.Module):
    """Self-attention over [task_tokens | image_patch_tokens | text_tokens].

    Uses available-tasks-only (Route B) by default.
    """

    def __init__(
        self,
        n_tasks: int,
        hidden_dim: int,
        n_channels: int = 5,
        n_spatial_tokens: int = 17,  # 1 CLS + 16 spatial
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        task_token_dim: Optional[int] = None,
        available_tasks_only: bool = True,
        img_input_dim: int = 1536,
        tower_hidden_dim: int = 32,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels
        self.n_spatial_tokens = n_spatial_tokens
        self.available_tasks_only = available_tasks_only

        # Image projection
        self.img_proj = nn.Linear(img_input_dim, hidden_dim, bias=False)

        # Channel embeddings (5 channels)
        self.channel_embed = nn.Parameter(torch.empty(n_channels, 1, hidden_dim))

        # Spatial position embeddings (for CLS + P spatial tokens)
        self.spatial_pos_embed = nn.Parameter(torch.empty(1, n_spatial_tokens, hidden_dim))

        # Text projection (if text encoder proj_dim != hidden_dim, handled externally)
        self.txt_proj = nn.Identity()  # text encoder already projects

        # Task tokens
        self.task_token_dim = task_token_dim or hidden_dim
        self.task_tokens = nn.Parameter(torch.empty(n_tasks, self.task_token_dim))
        self.task_proj = (
            nn.Linear(self.task_token_dim, hidden_dim, bias=False)
            if self.task_token_dim != hidden_dim
            else nn.Identity()
        )

        # Type embeddings
        self.type_embed_task = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.type_embed_image = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.type_embed_text = nn.Parameter(torch.empty(1, 1, hidden_dim))

        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, tower_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden_dim, 1),
            )
            for _ in range(n_tasks)
        ])

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.task_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.channel_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.spatial_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed_task, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed_image, mean=0.0, std=0.02)
        nn.init.normal_(self.type_embed_text, mean=0.0, std=0.02)
        if isinstance(self.img_proj, nn.Linear):
            nn.init.xavier_normal_(self.img_proj.weight)
        if isinstance(self.task_proj, nn.Linear):
            nn.init.xavier_normal_(self.task_proj.weight)
        for tower in self.task_towers:
            for m in tower:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def _prepare_image_tokens(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """Project, add channel + spatial embeddings, flatten.

        Args:
            img_tokens: (B, C, T, D_in)  e.g. (B, 5, 17, 1536)
        Returns:
            (B, C*T, hidden_dim)          e.g. (B, 85, 256)
        """
        B, C, T, D = img_tokens.shape
        x = self.img_proj(img_tokens)  # (B, C, T, hidden_dim)

        # Add channel embedding (broadcast over spatial positions)
        x = x + self.channel_embed.unsqueeze(0)  # (1, C, 1, hidden_dim) broadcast

        # Add spatial position embedding (broadcast over channels)
        x = x + self.spatial_pos_embed[:, :T, :]  # (1, 1, T, hidden_dim) broadcast

        return x.reshape(B, C * T, self.hidden_dim)

    def forward(
        self,
        img_tokens: torch.Tensor,   # (B, C*T, hidden_dim) - already prepared
        text_tokens: torch.Tensor,   # (B, L, hidden_dim)
        text_mask: torch.Tensor,     # (B, L)
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.available_tasks_only and labels is not None:
            return self._forward_available_only(img_tokens, text_tokens, text_mask, labels)
        return self._forward_all_tasks(img_tokens, text_tokens, text_mask, labels)

    def _forward_available_only(self, img_tokens, text_tokens, text_mask, labels):
        B = img_tokens.shape[0]
        N_img = img_tokens.shape[1]
        N_txt = text_tokens.shape[1]
        T = self.n_tasks
        D = self.hidden_dim
        device = img_tokens.device

        all_task_toks = self.task_proj(self.task_tokens)  # (T, D)

        available = labels != -1  # (B, T)
        n_avail = available.sum(dim=1)
        max_avail = max(n_avail.max().item(), 1)

        sorted_idx = available.float().argsort(dim=1, descending=True, stable=True)
        trim_idx = sorted_idx[:, :max_avail]

        all_expanded = all_task_toks.unsqueeze(0).expand(B, -1, -1)
        task_toks = all_expanded.gather(1, trim_idx.unsqueeze(-1).expand(-1, -1, D))

        positions = torch.arange(max_avail, device=device).unsqueeze(0)
        task_pad = positions >= n_avail.unsqueeze(1)

        task_toks = task_toks + self.type_embed_task
        img_tokens = img_tokens + self.type_embed_image
        text_tokens = text_tokens + self.type_embed_text

        seq = torch.cat([task_toks, img_tokens, text_tokens], dim=1)
        seq = self.input_norm(seq)

        img_pad = torch.zeros(B, N_img, dtype=torch.bool, device=device)
        txt_pad = ~text_mask.bool()
        padding_mask = torch.cat([task_pad, img_pad, txt_pad], dim=1)

        seq = self.transformer(seq, src_key_padding_mask=padding_mask)

        task_output = self.out_norm(seq[:, :max_avail, :])

        logits = torch.zeros(B, T, device=device, dtype=task_output.dtype)
        for b in range(B):
            for k in range(n_avail[b]):
                task_id = trim_idx[b, k].item()
                logits[b, task_id] = self.task_towers[task_id](
                    task_output[b, k]
                ).squeeze(-1)
        return logits

    def _forward_all_tasks(self, img_tokens, text_tokens, text_mask, labels):
        B = img_tokens.shape[0]
        N_img = img_tokens.shape[1]
        T = self.n_tasks
        device = img_tokens.device

        task_toks = self.task_proj(self.task_tokens).unsqueeze(0).expand(B, -1, -1)

        task_toks = task_toks + self.type_embed_task
        img_tokens = img_tokens + self.type_embed_image
        text_tokens = text_tokens + self.type_embed_text

        seq = torch.cat([task_toks, img_tokens, text_tokens], dim=1)
        seq = self.input_norm(seq)

        task_pad = torch.zeros(B, T, dtype=torch.bool, device=device)
        img_pad = torch.zeros(B, N_img, dtype=torch.bool, device=device)
        txt_pad = ~text_mask.bool()
        padding_mask = torch.cat([task_pad, img_pad, txt_pad], dim=1)

        seq = self.transformer(seq, src_key_padding_mask=padding_mask)

        task_output = self.out_norm(seq[:, :T, :])
        logits = torch.stack(
            [self.task_towers[t](task_output[:, t, :]).squeeze(-1) for t in range(T)],
            dim=1,
        )
        return logits


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class ScenarioXTModel(nn.Module):
    def __init__(
        self,
        mil_pooling: SpatialMILPooling,
        text_encoder: TextEncoder,
        sa_head: PatchSelfAttentionHead,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.mil_pooling = mil_pooling
        self.text_encoder = text_encoder
        self.sa_head = sa_head
        self.use_cls_token = use_cls_token

    def forward(
        self,
        imgs: torch.Tensor,
        mol: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # imgs: (B, M, C, T, D) — M sites, C=5 channels, T=1+P tokens, D=1536

        # MIL pool across sites → (B, C, T, D)
        bag_feats = self.mil_pooling(imgs)

        # Optionally drop CLS token (index 0)
        if not self.use_cls_token:
            bag_feats = bag_feats[:, :, 1:, :]  # (B, C, P, D)

        # Prepare image tokens: project + add channel/spatial embeddings + flatten
        img_tokens = self.sa_head._prepare_image_tokens(bag_feats)  # (B, C*T, hidden_dim)

        # Text encoding
        text_tokens, text_mask = self.text_encoder(
            mol["input_ids"], mol["attention_mask"]
        )

        # Trim text to batch max
        max_len = text_mask.sum(dim=1).max().int().item()
        text_tokens = text_tokens[:, :max_len, :]
        text_mask = text_mask[:, :max_len]

        return self.sa_head(img_tokens, text_tokens, text_mask, labels=labels)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_param_groups(model, text_lr, head_lr, weight_decay):
    text_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("text_encoder.bert."):
            text_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if text_params:
        groups.append({"params": text_params, "lr": text_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def evaluate_model(model, loader, device, use_focal_loss, focal_gamma, label_cols):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, mol, _keys, yb in loader:
            imgs = imgs.to(device, non_blocking=True)
            mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
            yb = yb.to(device, non_blocking=True)
            logits = model(imgs, mol, labels=yb)
            loss = masked_multitask_loss(logits, yb, use_focal_loss, focal_gamma)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    summary, per_task = compute_metrics(logits_np, labels_np, label_cols)
    summary["loss"] = float(total_loss / max(n_batches, 1))
    return summary, per_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device()
    use_amp = args.amp and device == "cuda"
    print(f"[Info] Device: {device} | AMP: {use_amp}")

    # Labels
    label_pack = load_labels(args.labels_csv)
    n_tasks = len(label_pack.label_cols)
    print(f"[Info] Loaded {n_tasks} tasks")

    # Datasets
    print("[Info] Loading datasets...")
    ds_kwargs = dict(
        patch_h5_path=args.patch_h5,
        molecule_csv=args.molecule_csv,
        label_map=label_pack.label_map,
        max_text_len=args.max_text_len,
        bert_model=args.bert_model,
        max_samples=args.max_samples_per_split,
    )
    train_dataset = PatchWellBagDataset(split_csv=args.train_split, **ds_kwargs)
    val_dataset = PatchWellBagDataset(split_csv=args.val_split, **ds_kwargs)
    test_dataset = PatchWellBagDataset(split_csv=args.test_split, **ds_kwargs)

    loader_kwargs = dict(
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_patch_well_bags,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)

    # Infer token counts from first sample
    sample_bag, _, _, _ = train_dataset[0]
    n_channels = sample_bag.shape[1]  # 5
    n_tokens_per_channel = sample_bag.shape[2]  # 1+P (e.g. 17)
    emb_dim = sample_bag.shape[3]  # 1536
    if not args.use_cls_token:
        n_tokens_per_channel -= 1
    n_img_tokens = n_channels * n_tokens_per_channel
    print(f"[Info] Image: {n_channels} channels x {n_tokens_per_channel} tokens = {n_img_tokens} image tokens")
    print(f"[Info] Embedding dim: {emb_dim}")

    # Build model
    sa_dim = args.sa_hidden_dim

    mil = SpatialMILPooling(
        input_dim=emb_dim,
        hidden_dim=args.mil_hidden_dim,
        pooling=args.mil_pooling,
    )
    text_enc = TextEncoder(bert_model=args.bert_model, proj_dim=sa_dim)
    sa_head = PatchSelfAttentionHead(
        n_tasks=n_tasks,
        hidden_dim=sa_dim,
        n_channels=n_channels,
        n_spatial_tokens=n_tokens_per_channel,
        num_layers=args.sa_num_layers,
        num_heads=args.sa_num_heads,
        ff_dim=args.sa_ff_dim,
        dropout=args.sa_dropout,
        task_token_dim=args.task_token_dim,
        available_tasks_only=args.available_tasks_only,
        img_input_dim=emb_dim,
        tower_hidden_dim=args.tower_hidden_dim,
    )
    model = ScenarioXTModel(
        mil_pooling=mil,
        text_encoder=text_enc,
        sa_head=sa_head,
        use_cls_token=args.use_cls_token,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    text_params = sum(p.numel() for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("text_encoder.bert."))
    head_params = total_params - text_params
    print(f"[Info] Trainable params: {total_params:,} (BERT: {text_params:,}, head+MIL+proj: {head_params:,})")

    mode_str = "Route B (available tasks only)" if args.available_tasks_only else "All tasks"
    print(f"[Info] Mode: {mode_str}")
    print(f"[Info] SA dim: {sa_dim}, layers: {args.sa_num_layers}, heads: {args.sa_num_heads}")

    param_groups = build_param_groups(model, args.text_lr, args.head_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # LR scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = None
    if args.lr_schedule != "none":
        from torch.optim.lr_scheduler import LambdaLR

        min_ratio = args.min_lr_ratio

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            if args.lr_schedule == "cosine":
                return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            else:  # linear
                return min_ratio + (1.0 - min_ratio) * (1.0 - progress)

        scheduler = LambdaLR(optimizer, lr_lambda)
        print(f"[Info] LR schedule: {args.lr_schedule}, warmup_steps: {warmup_steps}/{total_steps}, min_lr_ratio: {min_ratio}")
    else:
        print(f"[Info] LR schedule: none (constant LR)")

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for imgs, mol, _keys, yb in tqdm(train_loader, desc=f"Train {epoch:03d}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            mol = {k: v.to(device, non_blocking=True) for k, v in mol.items()}
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs, mol, labels=yb)
                loss = masked_multitask_loss(logits, yb, args.use_focal_loss, args.focal_gamma)

            if use_amp:
                scaler.scale(loss).backward()
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_summary, _ = evaluate_model(
            model, val_loader, device, args.use_focal_loss, args.focal_gamma, label_pack.label_cols
        )
        val_auc = val_summary["roc_auc_mean"]
        current_lrs = {f"lr_group_{i}": pg["lr"] for i, pg in enumerate(optimizer.param_groups)}
        if scheduler is not None:
            current_lrs = {f"lr_group_{i}": scheduler.get_last_lr()[i] for i in range(len(optimizer.param_groups))}
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_summary["loss"],
            "val_roc_auc_mean": val_summary["roc_auc_mean"],
            "val_ap_mean": val_summary["ap_mean"],
            "val_f1_mean": val_summary["f1_mean"],
            "val_n_tasks_evaluated": val_summary["n_tasks_evaluated"],
            **current_lrs,
        })
        lr_str = " ".join(f"{k}={v:.2e}" for k, v in current_lrs.items())
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
            f"val_loss={val_summary['loss']:.6f} "
            f"val_roc_auc={val_summary['roc_auc_mean']:.4f} "
            f"val_ap={val_summary['ap_mean']:.4f} "
            f"{lr_str}"
        )

        if not np.isnan(val_auc) and val_auc > best_val_auc:
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

    val_summary, val_per_task = evaluate_model(
        model, val_loader, device, args.use_focal_loss, args.focal_gamma, label_pack.label_cols
    )
    test_summary, test_per_task = evaluate_model(
        model, test_loader, device, args.use_focal_loss, args.focal_gamma, label_pack.label_cols
    )

    # Save
    ckpt_path = os.path.join(args.out_dir, "best_scenarioXT_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "n_tasks": n_tasks,
        "label_cols": label_pack.label_cols,
        "args": vars(args),
    }, ckpt_path)

    metrics = {
        "best_epoch": best_epoch,
        "history": history,
        "val": val_summary,
        "test": test_summary,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(val_per_task).to_csv(os.path.join(args.out_dir, "val_per_task.csv"), index=False)
    pd.DataFrame(test_per_task).to_csv(os.path.join(args.out_dir, "test_per_task.csv"), index=False)

    print(f"\n[Result] Val:  {json.dumps(val_summary, indent=2)}")
    print(f"\n[Result] Test: {json.dumps(test_summary, indent=2)}")
    print(f"\n[Saved] {ckpt_path}")


if __name__ == "__main__":
    main()
