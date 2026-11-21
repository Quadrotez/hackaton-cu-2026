from astra_detector_stack01 import *

def encoders():
    full=Material(W)
    cm,_,t=open_clip.create_model_and_transforms("ViT-B-32",pretrained=str(W/"astra_clip.bin"),device="cpu");cm.eval()
    return full,cm,t

@torch.inference_mode()
def encode(crops,enc):
    full,cm,t=enc; c=[]
    for j in range(0,len(crops),8):
        batch=torch.stack([t(Image.fromarray(q[:,:,::-1])) for q in crops[j:j+8]])
        c.append(cm.encode_image(batch).numpy())
    return normalize(full.encode(crops)),normalize(np.concatenate(c))

if __name__=="__main__":
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ids=keys(read_data()[0])
    paths={p.stem:p for p in (W/"crops").glob("*/*/*.png")}
    paths.update({p.stem[6:]:p for p in (W/"astra_tmp").glob("astra_*.png")})
    with threadpool_limits(4):
        d,c=encode([cv2.flip(cv2.imread(str(paths[k])),1) for k in ids],encoders())
    with (W/"astra_joint04_train_flip.npz").open("xb") as f:np.savez(f,ids=ids,d=d,c=c)
