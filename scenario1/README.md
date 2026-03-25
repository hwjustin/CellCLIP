```
CUDA_VISIBLE_DEVICES=1 python scenario1/train_bioactivity_concat_fnn.py \
  --epochs 15 \
  --patience 3 \
  --extract_batch_size 64 \
  --train_batch_size 64 \
  --lr 5e-5 \
  --weight_decay 1e-4 \
  --fc_units 2048 \
  --cls_input_dropout 0.1 \
  --cls_dropout 0.5 \
  --out_dir results/bioactivity_concat_fnn_lr5e5 \
  --cache_dir results/bioactivity_concat_fnn_lr5e5/cache
```