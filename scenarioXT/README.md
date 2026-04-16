# Scenario XT: Native ViT Patch Tokens + Self-Attention

## Motivation

ScenarioX used 6 summary tokens (CLS + 5 channel embeddings) from
CrossChannelFormer — these are **not native ViT spatial tokens**, they're just
compressed summaries. The self-attention head had almost nothing rich to attend
over.

ScenarioXT feeds **real DINOv2 patch tokens** directly into the SA head,
bypassing CrossChannelFormer entirely (MultiTab-Net style: frozen encoder,
trainable head).

| Component | ScenarioX | ScenarioXT |
|-----------|-----------|------------|
| Image encoder | CrossChannelFormer (trainable) | DINOv2-giant (frozen, pre-extracted) |
| Image tokens | 6 (CLS + 5 channel summaries) | 85 (5 ch x 17 tokens: 1 CLS + 16 spatial) |
| Text encoder | CellCLIP BERT (trainable) | Fresh BERT (trainable) |
| Text tokens | ~35 (full BERT sequence) | ~35 (full BERT sequence) |
| CrossChannelFormer | Used | **Bypassed** |
| Spatial info | None | 4x4 pooled DINOv2 patches per channel |

## Architecture

```
DINOv2-giant patch tokens (frozen, pre-extracted H5)
    (num_sites, 5, 17, 1536)  [1 CLS + 16 spatial per channel]
    |
    v
Spatial MIL Pooling across sites (per position)
    --> (B, 5, 17, 1536)
    |
    v
Linear projection (1536 -> sa_hidden_dim)           TRAINABLE
    + channel embeddings (5 learned)                 TRAINABLE
    + spatial position embeddings (17 learned)       TRAINABLE
    --> flatten --> (B, 85, sa_hidden_dim)
    |
    |       BERT text encoder                        TRAINABLE (low LR)
    |       last_hidden_state -> proj -> (B, L, sa_hidden_dim)
    |       trimmed to batch max length
    |
    +--------+--------+
             |
             v
    K available task tokens
        --> task_proj --> (B, K, sa_hidden_dim)
    + type embeddings (task / image / text)
             |
             v
    [K task | 85 img | L text]   ~126 tokens total
             |
             v
    LayerNorm -> TransformerEncoder (self-attention)
             |
             v
    Extract task positions -> Per-task MLP towers -> logits (B, 209)
```

## Step 1: Extract Patch Tokens

```bash
CUDA_VISIBLE_DEVICES=0 python scenarioXT/extract_dinov2_patch_tokens.py \
    --input_dir /data/huadi/cellpainting_data/cpg0012/npzs \
    --output_dir /data/huadi/cellpainting_data/bray2017/img \
    --output_file dinov2-giant_patch4x4.h5 \
    --pool_size 4 --batch_size 32 --gpu_batch 64
```

This takes ~3 hours and produces a ~74 GB H5 file with shape (282k, 5, 17, 1536) in float16.

## Step 2: Train

```bash
CUDA_VISIBLE_DEVICES=2 python scenarioXT/train_bioactivity_patch_sa_finetune.py \
    --sa_hidden_dim 256 \
    --sa_num_layers 2 \
    --sa_num_heads 8 \
    --sa_ff_dim 1024 \
    --sa_dropout 0.1 \
    --tower_hidden_dim 32 \
    --text_lr 5e-6 \
    --head_lr 5e-4 \
    --epochs 30 \
    --patience 8 \
    --amp \
    --out_dir results/scenarioXT
```

### Smaller SA dim variant

```bash
CUDA_VISIBLE_DEVICES=2 python scenarioXT/train_bioactivity_patch_sa_finetune.py \
    --sa_hidden_dim 128 \
    --sa_num_layers 1 \
    --sa_num_heads 4 \
    --sa_ff_dim 512 \
    --tower_hidden_dim 16 \
    --text_lr 5e-6 \
    --head_lr 5e-4 \
    --epochs 30 \
    --amp \
    --out_dir results/scenarioXT_dim128
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch_h5` | `.../dinov2-giant_patch4x4.h5` | Pre-extracted patch token H5 |
| `--sa_hidden_dim` | 256 | SA operating dimension |
| `--sa_num_layers` | 2 | Transformer encoder layers |
| `--sa_num_heads` | 8 | Attention heads |
| `--tower_hidden_dim` | 32 | Per-task MLP hidden dim |
| `--text_lr` | 5e-6 | BERT learning rate |
| `--head_lr` | 5e-4 | SA head + projections + MIL LR |
| `--use_cls_token` / `--no_cls_token` | on | Include DINOv2 CLS in image tokens |
| `--available_tasks_only` / `--all_tasks` | on | Route B (recommended) |
| `--mil_pooling` | attention | MIL strategy: attention or mean |
