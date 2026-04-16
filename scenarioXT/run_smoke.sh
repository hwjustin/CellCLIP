#!/bin/bash
# ScenarioXT — Quick smoke test (5 samples, 1 epoch)
# Usage: bash scenarioXT/run_smoke.sh [GPU_ID]

GPU=${1:-2}

source /home/huadi/miniconda3/etc/profile.d/conda.sh
conda activate cellclip

cd /data/huadi/CellCLIP
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=/data/huadi/CellCLIP:$PYTHONPATH

python scenarioXT/train_bioactivity_patch_sa_finetune.py \
    --max_samples_per_split 5 \
    --epochs 1 \
    --sa_hidden_dim 64 \
    --sa_num_layers 1 \
    --sa_num_heads 4 \
    --out_dir /tmp/xt_smoke
