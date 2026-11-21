"""Offline material inference for masked BGR crops. Dependencies: torch, timm, PIL, cv2, numpy, safetensors."""
from pathlib import Path
import os
os.environ.setdefault("HF_HUB_OFFLINE","1")
import cv2
import torch
import timm
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from timm.data import create_transform, resolve_model_data_config

def norm(x):
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def masked_crop(color, mask):
    bgr=np.where(mask[...,None],color,np.uint8(0))
    return cv2.resize(bgr,(128,128),interpolation=cv2.INTER_AREA if max(mask.shape)>128 else cv2.INTER_LINEAR)

class Material:
    def __init__(self,directory,compact=False):
        torch.set_num_threads(4)
        root=Path(directory)
        name="vit_small_patch16_224.dino" if compact else "vit_base_patch16_224.dino"
        self.m=timm.create_model(name,pretrained=False,num_classes=0).cpu().eval()
        headfile="astra_distilled.npz" if compact else "astra_dino_head.npz"
        with np.load(root/headfile,allow_pickle=False) as z:
            if compact:
                state={}
                for k,v in self.m.state_dict().items():
                    d=z[k]
                    if k+".scale" in z.files:
                        q=np.stack((d&15,d>>4),axis=-1).reshape(-1,32).astype("float32")-8
                        d=(q*z[k+".scale"][:,None]).reshape(v.shape)
                    state[k]=torch.from_numpy(d.astype("float32"))
                self.m.load_state_dict(state)
            else:
                self.m.load_state_dict(load_file(str(root/"astra_dino.safetensors")))
            self.head={k:z[k] for k in ("W","b","labels","map","offset","mix") if k in z.files}
        self.t=create_transform(**resolve_model_data_config(self.m),is_training=False)

    @torch.inference_mode()
    def encode(self,crops):
        return np.concatenate([self.m(torch.stack([
            self.t(Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB))) for x in crops[i:i+8]
        ])).numpy() for i in range(0,len(crops),8)])

    def predict(self,crops):
        if not len(crops):
            return np.array([],dtype=str)
        x=norm(self.encode(crops))
        h=self.head
        if "map" in h:
            x=norm(np.column_stack([h["mix"]*x,norm(x@h["map"]+h["offset"])]))
        return h["labels"][(x@h["W"].T+h["b"]).argmax(axis=1)]
