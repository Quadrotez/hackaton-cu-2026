"""Replay frozen Joint04 encoders on verified predicted-mask donors."""
import argparse
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize

from astra_joint04_views import encoders, masked_crop, bounds
from fusion import sprint5_crop_adapter as adapter


def encode_donors(base, donors, models, device):
    """Return per-slot donor identities, original/flip vectors and mask areas."""
    full, clip, transform = models
    full.m.to(device)
    clip.to(device)
    keys, indices, areas, crops = [], [], [], []
    seen = {}
    with ExitStack() as stack:
        archives = {}
        for image, detections in base.items():
            color = None
            for j in range(len(detections)):
                archive, _, key, slot, h, w, x, y, r, b, rgb, _ = donors[image, j]
                donor = archive, key, slot
                keys.append((image, j, adapter.digest([image, j, detections[j]])))
                if donor not in seen:
                    if archive not in archives:
                        archives[archive] = stack.enter_context(np.load(adapter.ROOT / archive, allow_pickle=False))
                    if color is None:
                        color = cv2.imread(str(adapter.ROOT / rgb))
                        assert color is not None and color.shape[:2] == (h, w)
                    mask = np.unpackbits(archives[archive][key][slot], axis=-1, count=w).astype(bool)
                    l, t, rr, bb = bounds((x, y, r-x, b-y), w, h, .1)
                    local = mask[t:bb, l:rr]
                    assert local.any()
                    seen[donor] = len(crops)
                    crops.append(masked_crop(color[t:bb, l:rr], local))
                    areas.append(int(mask.sum()))
                indices.append(seen[donor])

    def views():
        with torch.inference_mode():
            for flip in (False, True):
                dino, visual = [], []
                for start in range(0, len(crops), 8):
                    batch = crops[start:start+8]
                    if flip:
                        batch = [cv2.flip(c, 1) for c in batch]
                    rgb = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in batch]
                    dino.append(full.m(torch.stack([full.t(c) for c in rgb]).to(device)).cpu().numpy())
                    visual.append(clip.encode_image(torch.stack([transform(c) for c in rgb]).to(device)).cpu().numpy())
                yield normalize(np.concatenate(dino))
                yield normalize(np.concatenate(visual))

    print(f"Encoding {len(crops)} unique predicted crops for {len(keys)} slots on {device}", flush=True)
    d, c, df, cf = views()
    return dict(keys=np.asarray(keys), donor_index=np.asarray(indices), area=np.asarray(areas), d=d, c=c, df=df, cf=cf)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    folds = adapter.training_contract()[0]
    _, base = adapter.load_oof(adapter.OOF_BASE, folds)
    donors = adapter.load_donors(base, folds)
    bundle = encode_donors(base, donors, encoders(), args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as file:
        np.savez_compressed(file, **bundle)


if __name__ == "__main__":
    main()
