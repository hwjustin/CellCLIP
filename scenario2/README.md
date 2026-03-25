```bash
# Smoke test (small split cap)
CUDA_VISIBLE_DEVICES=1 python train_bioactivity_joint_finetune.py \
  --epochs 2 \
  --patience 2 \
  --train_batch_size 8 \
  --eval_batch_size 8 \
  --num_workers 2 \
  --backbone_lr 2e-5 \
  --head_lr 2e-4 \
  --max_samples_per_split 256 \
  --out_dir results/bioactivity_joint_finetune_smoke
```

```bash
# Full run (task loss only, end-to-end fine-tuning)
CUDA_VISIBLE_DEVICES=1 python scenario2/train_bioactivity_joint_finetune.py \
  --epochs 12 \
  --patience 4 \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --backbone_lr 5e-6 \
  --head_lr 5e-5 \
  --weight_decay 5e-4 \
  --amp \
  --cls_input_dropout 0.1 \
  --cls_dropout 0.5 \
  --fc_units 2048 \
  --out_dir results/bioactivity_joint_finetune_lr5e6_head5e5
```

```bash
CUDA_VISIBLE_DEVICES=1 python scenario2/train_bioactivity_joint_finetune.py \
  --epochs 12 \
  --patience 4 \
  --train_batch_size 64 \
  --eval_batch_size 64 \
  --num_workers 4 \
  --backbone_lr 5e-6 \
  --head_lr 5e-5 \
  --weight_decay 5e-4 \
  --amp \
  --seed 123 \
  --cls_input_dropout 0.1 \
  --cls_dropout 0.5 \
  --fc_units 2048 \
  --out_dir results/bioactivity_joint_finetune_lr5e6_head5e5_seed123
```

Outputs under `--out_dir`:
- `best_joint_model.pt`
- `metrics.json`
- `val_per_task.csv`
- `test_per_task.csv`

Notes:
- This scenario fine-tunes both CellCLIP encoders (`encode_mil`+`encode_image`, `encode_text`) and classifier jointly.
- Objective is task loss only (`masked_multitask_loss`, optional focal variant via `--use_focal_loss`).
- Optimizer groups use `--backbone_lr` for visual/MIL params, `--text_lr` (or fallback to `--backbone_lr`) for text params, and `--head_lr` for classifier params.
