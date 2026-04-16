#!/bin/bash
# ScenarioXT — ORIGINAL config (baseline)
# Val AUC: 0.8021, Test AUC: 0.7894 (seed 42)
#
# Usage: bash scenarioXT/run_full_train.sh [GPU_ID] [SEED]
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
    --sa_num_layers 2 \
    --sa_num_heads 8 \
    --sa_ff_dim 1024 \
    --sa_dropout 0.1 \
    --tower_hidden_dim 32 \
    --text_lr 5e-6 \
    --head_lr 5e-4 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --patience 8 \
    --amp \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --lr_schedule none \
    --seed $SEED \
    --out_dir results/scenarioXT_original_s${SEED}
