```bash
# Smoke test (small split cap)
CUDA_VISIBLE_DEVICES=1 python scenario2_5/train_bioactivity_preconcat_mlp_finetune.py \
  --epochs 2 \
  --patience 2 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_workers 2 \
  --backbone_lr 2e-5 \
  --head_lr 2e-4 \
  --embed_dim 512 \
  --max_samples_per_split 256 \
  --out_dir results/bioactivity_preconcat_mlp_finetune_smoke
```

```bash
# Full run (scenario2.5: pre-concat projection MLP ablation)
CUDA_VISIBLE_DEVICES=2 python scenario2_5/train_bioactivity_preconcat_mlp_finetune.py \
  --epochs 12 \
  --patience 4 \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --backbone_lr 5e-6 \
  --head_lr 5e-5 \
  --weight_decay 5e-4 \
  --amp \
  --embed_dim 512 \
  --proj_identity_init \
  --cls_input_dropout 0.1 \
  --cls_dropout 0.5 \
  --fc_units 2048 \
  --out_dir results/bioactivity_preconcat_mlp_finetune_lr5e6_head5e5
```

Outputs under `--out_dir`:
- `best_preconcat_mlp_joint_model.pt`
- `metrics.json`
- `val_per_task.csv`
- `test_per_task.csv`

Notes:
- Scenario2.5 fine-tunes both CellCLIP encoders and keeps task-loss-only training (`masked_multitask_loss`, optional focal variant via `--use_focal_loss`).
- Compared with scenario2, it adds two per-modality projection MLPs before fusion:
  - `cell encoder feature -> ProjectionMLP -> cell feature`
  - `perturb encoder feature -> ProjectionMLP -> perturb feature`
- Classifier input is `concat(cell feature, perturb feature)` after `LayerNorm`.
- No CLIP loss, no CLUB regularization, and no disentanglement branches are used in this scenario.
- Optional stabilization knobs:
  - `--proj_lr` for separate projection-MLP LR
  - `--freeze_backbone_epochs` for warmup before unfreezing backbone
  - `--proj_identity_init` to start projection branches as exact identity (for equal in/out dims)
  - `--residual_projection` and `--proj_dropout` for more stable projection branches
  - `--proj_hidden_dim` and `--proj_num_layers` to reduce projection capacity
  - `--fc_units` and `--cls_num_hidden_layers` to reduce classifier capacity
