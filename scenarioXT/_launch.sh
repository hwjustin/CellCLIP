#!/bin/bash
export PYTHONPATH=/data/huadi/CellCLIP
export CUDA_VISIBLE_DEVICES=2
exec /home/huadi/miniconda3/envs/cellclip/bin/python /data/huadi/CellCLIP/scenarioXT/train_bioactivity_patch_sa_finetune.py --sa_hidden_dim 256 --sa_num_layers 2 --sa_num_heads 8 --sa_ff_dim 1024 --sa_dropout 0.1 --tower_hidden_dim 32 --text_lr 5e-6 --head_lr 5e-4 --epochs 30 --patience 8 --amp --train_batch_size 64 --eval_batch_size 64 --out_dir /data/huadi/CellCLIP/results/scenarioXT_full
