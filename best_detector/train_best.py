"""Astramis: улучшенный детектор YOLO11s-seg.
Запуск: pip install -q -U ultralytics kagglehub && python train_best.py
Локально нужен датасет в /kaggle/input/astramis-slim или variables KAGGLE_DATASET.
"""

# --- Настройки ---
import csv
import json
import random
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
MODEL = 'yolo11s-seg.pt'
EPOCHS = 10
IMGSZ = 640
SEED = 0
CONF = 0.025
FORMS = ['bulk', 'solid']
VAL_OBJECTS = ['b02', 'b12', 'b21', 's02', 's07', 's08', 's18', 's36', 's42']
EMPTY_SAMPLE_SEED = 42
N_EMPTY = 25
KAGGLE_DATASET = 'b5wvix/astramis-slim'

SRC = Path('/kaggle/input/astramis-slim')
if not (SRC / 'real').exists():
    try:
        import kagglehub
        SRC = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as e:
        print('kagglehub недоступен, пробуем kaggle CLI:', e)
        dl = WORK / 'dl'
        dl.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'kaggle', 'datasets', 'download',
                        '-d', KAGGLE_DATASET, '-p', str(dl), '--unzip'], check=True)
        SRC = dl
if (SRC / 'real').exists():
    DATA = SRC
else:
    DATA = WORK / 'data'
    DATA.mkdir(parents=True, exist_ok=True)
    for z in SRC.glob('*.zip'):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA / z.stem)
    for f in SRC.iterdir():
        if f.is_file() and f.suffix != '.zip':
            shutil.copy(f, DATA / f.name)
        elif f.is_dir() and not (DATA / f.name).exists():
            shutil.copytree(f, DATA / f.name)
sys.path.insert(0, str(DATA))
import score as metric
print('DATA =', DATA)


# --- Сплит по object_id (как просил Виталя) ---
def color_path(frame_dir):
    p = frame_dir / 'color.png'
    return p if p.exists() else frame_dir / 'color.jpg'


def load(labels_file):
    info = json.loads((DATA / 'labels' / labels_file).read_text(encoding='utf-8'))
    anns = defaultdict(list)
    for a in info['annotations']:
        anns[a['image_id']].append(a)
    out = []
    for im in info['images']:
        d = DATA / Path(im['file_name']).parent
        if not color_path(d).exists():
            continue
        out.append({'dir': d, 'id': d.name, 'w': im['width'], 'h': im['height'],
                    'anns': anns[im['id']]})
    return out


real = load('real_train.json')
val_objects = set(VAL_OBJECTS)
has_val_object = lambda f: any(a['attributes']['object_id'] in val_objects for a in f['anns'])
empty_real_ids = sorted(f['id'] for f in real if not f['anns'])
extra_empty = random.Random(EMPTY_SAMPLE_SEED).sample(empty_real_ids, N_EMPTY)

val_ids = {f['id'] for f in real if has_val_object(f)} | set(extra_empty)
val = [f for f in real if f['id'] in val_ids]
train_real = [f for f in real if f['id'] not in val_ids and not has_val_object(f)]

synth = load('synth_train.json')
synth_dropped = len(synth)
train_synth = [f for f in synth if f['id'] not in val_ids and not has_val_object(f)]
synth_dropped -= len(train_synth)

train_all = train_real + train_synth
train_ids = sorted(f['id'] for f in train_all)
val_ids_sorted = sorted(val_ids)
print(f'real_train={len(train_real)} train_synth={len(train_synth)} synth_dropped={synth_dropped} '
      f'TRAIN={len(train_ids)} VAL={len(val_ids_sorted)} empty_extra={len(extra_empty)}')

(WORK / 'train_ids.json').write_text(json.dumps(train_ids, indent=1), encoding='utf-8')
(WORK / 'val_ids.json').write_text(json.dumps(val_ids_sorted, indent=1), encoding='utf-8')


# --- YOLO-разметка ---
yolo_dir = WORK / 'yolo'
if yolo_dir.exists():
    shutil.rmtree(yolo_dir)
for split, frames in (('train', train_all), ('val', val)):
    (yolo_dir / 'images' / split).mkdir(parents=True)
    (yolo_dir / 'labels' / split).mkdir(parents=True)
    for f in frames:
        src = color_path(f['dir'])
        shutil.copy(src, yolo_dir / 'images' / split / f"{f['id']}{src.suffix}")
        lines = []
        for a in f['anns']:
            form = a['attributes']['form']
            if form not in FORMS:
                continue
            seg = max(a['segmentation'], key=len)
            pts = np.array(seg, dtype=float).reshape(-1, 2) / [f['w'], f['h']]
            lines.append(f'{FORMS.index(form)} ' + ' '.join(f'{v:.5f}' for v in pts.clip(0, 1).reshape(-1)))
        (yolo_dir / 'labels' / split / f"{f['id']}.txt").write_text('\n'.join(lines))
(yolo_dir / 'data.yaml').write_text(
    f'path: {yolo_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n'
    + ''.join(f'  {i}: {n}\n' for i, n in enumerate(FORMS)), encoding='utf-8')


# --- Обучение ---
model = YOLO(MODEL)
model.train(data=str(yolo_dir / 'data.yaml'), epochs=EPOCHS, imgsz=IMGSZ,
            project=str(WORK / 'seg_run'), name='seg', exist_ok=True,
            plots=False, verbose=False, seed=SEED, device=0)
best_src = WORK / 'seg_run' / 'seg' / 'weights' / 'best.pt'
BEST_PT = WORK / 'best_yolo11s_seg_valfixed.pt'
shutil.copy(best_src, BEST_PT)
print('BEST_PT =', BEST_PT, '| bytes =', BEST_PT.stat().st_size)


# --- D через score.py, только детекция ---
detector = YOLO(str(BEST_PT))


def predict(frame_dir):
    r = detector(str(color_path(frame_dir)), conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    out = []
    if r.boxes is None:
        return out
    for box, score, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({'box': (x1, y1, x2, y2), 'score': float(score), 'form': FORMS[int(cls)]})
    return out


gt, pred = {}, {}
for f in val:
    gt[f['id']] = [{'box': (a['bbox'][0], a['bbox'][1],
                            a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]),
                    'form': a['attributes']['form']}
                   for a in f['anns']]
    pred[f['id']] = predict(f['dir'])

aps = [ap for ap in (metric._average_precision(gt, pred, form) for form in FORMS) if ap is not None]
D = float(np.mean(aps)) if aps else 0.0
print('FINAL_D =', f'{D:.6f}')

(WORK / 'metrics.json').write_text(json.dumps({
    'D': D, 'AP_by_form': {f: ap for f, ap in zip(FORMS, aps)},
    'conf': CONF, 'n_train': len(train_ids), 'n_val': len(val_ids_sorted),
}, indent=2), encoding='utf-8')

with open(WORK / 'val_predictions.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow(['image_id', 'n', 'x1', 'y1', 'x2', 'y2', 'score', 'form'])
    for fid in val_ids_sorted:
        objs = pred.get(fid, [])
        if not objs:
            w.writerow([fid, 0, '', '', '', '', '', ''])
        for o in objs:
            w.writerow([fid, 1, *[f'{v:.2f}' for v in o['box']], f"{o['score']:.4f}", o['form']])

config = {
    'model': MODEL, 'epochs': EPOCHS, 'imgsz': IMGSZ, 'seed': SEED, 'conf': CONF,
    'classes': FORMS, 'val_objects': VAL_OBJECTS,
    'empty_frames': {'count': N_EMPTY, 'random_seed': EMPTY_SAMPLE_SEED,
                     'sampled_ids': extra_empty},
    'n_train': len(train_ids), 'n_val': len(val_ids_sorted),
    'split_rule': 'val = all real frames with one of val_objects + 25 sampled empty real frames; '
                  'train = real+synth frames with no val object, excluding all val frame ids',
    'best_pt': str(BEST_PT), 'D': D,
}
(WORK / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
print('ГОТОВО: best.pt, train_ids.json, val_ids.json, config.json, metrics.json, val_predictions.csv')
