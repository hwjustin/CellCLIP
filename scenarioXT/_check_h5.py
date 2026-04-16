import h5py
f = h5py.File("/data/huadi/cellpainting_data/bray2017/img/dinov2-giant_patch4x4.h5", "r")
print("shape:", f["embeddings"].shape)
print("dtype:", f["embeddings"].dtype)
print("well_id count:", f["well_id"].shape[0])
f.close()
