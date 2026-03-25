```
CUDA_VISIBLE_DEVICES=1 python train_bioactivity_concat_fnn.py   --epochs 8   --patience 3   --extract_batch_size 64   --train_batch_size 64   --fc_units 2048   --cls_input_dropout 0.1   --cls_dropout 0.5   --out_dir results/bioactivity_concat_fnn_perwell_encoderpool_bs64   --cache_dir results/bioactivity_concat_fnn_perwell_encoderpool/cache
```