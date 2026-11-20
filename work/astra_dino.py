"""Run from repository root: PYTHONPATH=work/astra_packages .venv/bin/python -B work/astra_dino.py [crop.png ...]"""
import os
from pathlib import Path
W = Path(__file__).resolve().parent
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "4"
os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(W/"astra_hf"), TORCH_HOME=str(W/"astra_torch"))
import csv
import json
import pickle
import sys
import numpy as np
from PIL import Image
import torch
import timm
from timm.data import create_transform, resolve_model_data_config

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
name = "vit_base_patch16_224.dino"


def encoder():
    m = timm.create_model(name, pretrained=True, num_classes=0,
                          pretrained_cfg_overlay={"file": str(W/"astra_dino.safetensors")}).cpu().eval()
    return m, create_transform(**resolve_model_data_config(m), is_training=False)


def rows(file):
    with (W/file).open() as f:
        return list(csv.DictReader(f))


def key(r):
    return f"{r['image_id']}_{r['ann_id']}"


def crop_paths():
    from material_split import DATA, load_frames
    from material_features import bounds, polygon_mask
    import cv2
    cv2.setNumThreads(1)
    primary = rows("features.csv")
    primary_ids = {key(r) for r in primary}
    extra = [r for r in rows("features_all_v2.csv") if key(r) not in primary_ids]
    paths = {p.stem:p for p in (W/"crops").glob("*/*/*.png")}
    assert primary_ids == paths.keys() and len(primary_ids) == 868
    for _, iid, im, anns in load_frames():
        for a in anns:
            k = f"{iid}_{a['id']}"
            if k in paths:
                continue
            color = cv2.imread(str(DATA/im["file_name"]))
            h,w = color.shape[:2]
            x,y,end,bottom = bounds(a["bbox"],w,h,.1)
            mask = polygon_mask(a["segmentation"],end-x,bottom-y,x,y)
            crop = cv2.resize(np.where(mask[...,None],color[y:bottom,x:end],np.uint8(0)),(128,128),
                              interpolation=cv2.INTER_AREA if max(mask.shape)>128 else cv2.INTER_LINEAR)
            paths[k] = W/f"astra_tmp/astra_{k}.png"
            assert cv2.imwrite(str(paths[k]),crop)
    return primary+extra, paths


@torch.inference_mode()
def embed(paths):
    m,t = encoder()
    out = []
    for start in range(0,len(paths),8):
        batch = []
        for p in paths[start:start+8]:
            with Image.open(p) as im:
                batch.append(t(im.convert("RGB")))
        out.append(m(torch.stack(batch)).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    if len(sys.argv)>1:
        with (W/"astra_probe.pkl").open("rb") as f:
            print(*pickle.load(f).predict(embed(sys.argv[1:])),sep="\n")
    else:
        data, paths = crop_paths()
        ids = [key(r) for r in data]
        X = embed([paths[k] for k in ids])
        assert X.shape == (len(set(ids)),768) and np.isfinite(X).all()
        for sl,prefix in [(slice(0,868),"astra_"),(slice(868,None),"astra_extra_")]:
            np.save(W/(prefix+"embeddings.npy"),X[sl])
            (W/(prefix+"ids.json")).write_text(json.dumps(ids[sl],indent=1)+"\n")
