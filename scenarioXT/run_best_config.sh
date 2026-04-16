#!/bin/bash
# ScenarioXT — BEST config (after hyperparameter tuning)
#
# Changes vs original:
#   sa_num_layers: 2 → 4     (Phase 4: depth > width)
#   sa_dropout:    0.1 → 0.3 (Phase 2: regularization)
#   weight_decay:  1e-4 → 1e-3 (Phase 2)
#   head_lr:       5e-4 → 2e-4 (Phase 3: slower head)
#   lr_schedule:   none → cosine (Phase 1: warmup + cosine decay)
#
# Multi-seed results (4 seeds: 42, 123, 456, 789):
#   Val AUC:  0.8077 ± 0.0040
#   Test AUC: 0.7874 ± 0.0108
#
# Usage: bash scenarioXT/run_best_config.sh [GPU_ID] [SEED]
#   GPU_ID: CUDA device index (default: 2)
#   SEED:   random seed (default: 42)

GPU=${1:-2}
SEED=${2:-42}

source /home/huadi/miniconda3/etc/profile.d/conda.sh
conda activate cellclip

cd /data/huadi/CellCLIP
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=/data/huadi/CellCLIP:$PYTHONPATH

python scenarioXT/train_bioactivity_patch_sa_finetune.py \
    --sa_hidden_dim 256 \
    --sa_num_layers 4 \
    --sa_num_heads 8 \
    --sa_ff_dim 1024 \
    --sa_dropout 0.3 \
    --tower_hidden_dim 32 \
    --text_lr 5e-6 \
    --head_lr 2e-4 \
    --weight_decay 1e-3 \
    --epochs 30 \
    --patience 8 \
    --amp \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --lr_schedule cosine \
    --warmup_ratio 0.05 \
    --min_lr_ratio 0.01 \
    --seed $SEED \
    --out_dir results/scenarioXT_best_s${SEED}
