"""Replay an actual crop with network access forbidden; compare to saved features."""
import os
import sys
from pathlib import Path
W=Path(__file__).resolve().parent
for var in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[var]="4"
def offline(event,args):
    if event in {"socket.connect","socket.getaddrinfo"}:
        raise RuntimeError("Offline check: "+event)
sys.addaudithook(offline)
import pickle
import json
import numpy as np
with (W/"astra_probe.pkl").open("rb") as f:
    probe=pickle.load(f)
if probe.n_features_in_==768:
    from astra_dino import embed
    prefix="astra_"
else:
    from astra_clip import embed
    prefix="astra_clip_"
    # Indicate inference mode (do not generate training text prototypes).
    sys.argv.append("infer")
ids=json.loads((W/(prefix+"ids.json")).read_text())
crop=next((W/"crops").glob("*/*/"+ids[0]+".png"))
expected=np.load(W/(prefix+"embeddings.npy"))[:1]
got=embed([crop])
np.testing.assert_allclose(expected,got,atol=2e-4,rtol=1e-4)
assert list(probe.predict(expected))==list(probe.predict(got))
print(f"Офлайн-проверка: {crop.stem} → {probe.predict(got)[0]}; эмбеддинг воспроизведён при заблокированных сетевых соединениях.")
