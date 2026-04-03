```bash
# Smoke test (small split cap)
CUDA_VISIBLE_DEVICES=1 python3 scenario3/train_bioactivity_frozen_finetune_concat_mlp.py \
  --epochs 2 \
  --patience 2 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_workers 2 \
  --backbone_lr 2e-5 \
  --head_lr 2e-4 \
  --max_samples_per_split 256 \
  --out_dir results/bioactivity_frozen_finetune_concat_mlp_smoke
```



```bash
# Disentanglement run (orthogonality loss: projected fine-tuned vs frozen features)
CUDA_VISIBLE_DEVICES=2 python3 scenario3/train_bioactivity_frozen_finetune_concat_mlp.py \
  --epochs 12 \
  --patience 4 \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --backbone_lr 5e-6 \
  --head_lr 5e-5 \
  --weight_decay 5e-4 \
  --amp \
  --ortho_weight 0.02 \
  --out_dir results/bioactivity_frozen_finetune_concat_mlp_ortho_no_proj
```

Outputs under `--out_dir`:
- `best_frozen_finetune_concat_joint_model.pt`
- `metrics.json`
- `val_per_task.csv`
- `test_per_task.csv`

Notes:
- Scenario 3 builds two parallel CellCLIP feature branches:
  - frozen branch: fixed reference features from a non-trainable checkpoint
  - fine-tuned branch: trainable features adapted to the bioactivity task
- Final representation is:
  - `[frozen_cell || frozen_perturbation || finetune_cell || finetune_perturbation]`
  - followed by LayerNorm + CellPaintSSL-style MLP classifier.
- By default, both branches use the same checkpoint (`--ckpt_path`); set `--frozen_ckpt_path` if you want a different frozen reference checkpoint.
- Scenario 3 style LR behavior is the default here too:
  - if `--text_lr` is not set, text encoder LR follows `--backbone_lr`.
  - use `--text_lr` only when you explicitly want a separate text-branch LR.
- Optional projection + orthogonality disentanglement:
  - `--enable_proj_mlp` enables fine-tuned feature projection heads (`cell_proj`, `text_proj`) before fusion.
  - Orthogonality loss is applied between projected fine-tuned and frozen features (cell/text), weighted by `--ortho_weight`.
  - Ortho loss is active from epoch 1 (no start/warmup scheduling).
- Only the fine-tuned branch and classifier are optimized; frozen branch stays in eval mode for stable feature targets.
