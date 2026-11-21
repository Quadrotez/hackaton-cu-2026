import os
from pathlib import Path
import json
import sys
import pickle
W=Path(__file__).resolve().parent
sys.path.insert(0,str(W/"astra_clip_packages"))
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[var]="4"
os.environ.update(HF_HUB_OFFLINE="1", HF_HOME=str(W/"astra_hf"), TORCH_HOME=str(W/"astra_torch"))
import torch
import numpy as np
from PIL import Image
import open_clip


@torch.inference_mode()
def embed(paths):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model,_,t=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu")
    model.eval()
    if len(sys.argv)==1:
        # Fixed English descriptions, independent of held-out objects.
        from astra_train import labels
        prompts=["a photo of an object made of {}.","a close-up photo of {} material.","a piece of {}."]
        tokens=open_clip.get_tokenizer("ViT-B-32")([p.format(k) for k in labels for p in prompts])
        text=model.encode_text(tokens,normalize=True).reshape(6,len(prompts),-1).mean(1)
        np.save(W/"astra_clip_text.npy",torch.nn.functional.normalize(text).numpy())
    out=[]
    for i in range(0,len(paths),8):
        batch=[]
        for p in paths[i:i+8]:
            with Image.open(p) as im:
                batch.append(t(im.convert("RGB")))
        out.append(model.encode_image(torch.stack(batch)).numpy())
    return np.concatenate(out)


if __name__=="__main__":
    if len(sys.argv)>1:
        with (W/"astra_probe.pkl").open("rb") as f:
            print(*pickle.load(f).predict(embed(sys.argv[1:])),sep="\n")
    else:
        paths={p.stem:p for p in (W/"crops").glob("*/*/*.png")}
        paths.update({p.stem.removeprefix("astra_"):p for p in (W/"astra_tmp").glob("astra_*.png")})
        ids=sum([json.loads((W/(p+"ids.json")).read_text()) for p in ("astra_","astra_extra_")],[])
        X=embed([paths[k] for k in ids])
        assert X.shape==(892,512) and np.isfinite(X).all()
        for sl,prefix in [(slice(0,868),"astra_clip_"),(slice(868,None),"astra_clip_extra_")]:
            np.save(W/(prefix+"embeddings.npy"),X[sl])
            (W/(prefix+"ids.json")).write_text(json.dumps(ids[sl],indent=1)+"\n")
