# Улучшенный детектор Астрамис (YOLO11s-seg)

Исходники улучшенного пайплайна. Не трогает `baseline.ipynb`, `score.py`, `ЖУРНАЛ.md`,
не отправляет сабмиты на платформу. Веса и данные в Git не лежат.

## Что делает

- `yolo11s-seg.pt`, 10 эпох, `imgsz=640`, `seed=0`, классы только `bulk`/`solid`.
- Сплит (уточнение Витали, 20.11 22:53):
  - val = все реальные кадры с объектами `b02 b12 b21 s02 s07 s08 s18 s36 s42`
    + `random.Random(42).sample(sorted(empty_real_ids), 25)`;
  - train = реальные + синтетические кадры без val-объектов, val-кадры исключены;
  - результат: 635 train / 78 val.
- D считается через `score._average_precision` из `score.py` при `conf=0.025`.
  Материал, объём и `take` не используются и не пересчитываются.

## Запуск на Kaggle

1. Загрузить компактный датасет как приватный Kaggle dataset `b5wvix/astramis-slim`
   (только `color.jpg` + `labels/*.json` + `score.py`).
2. `kaggle kernels push -p best_detector` — push создаёт версию и запускает ноутбук
   на T4 GPU (`enable_gpu=true`, `enable_internet=true`).
3. `kaggle kernels status b5wvix/astramis-best-detector-yolo11s-seg`
4. `kaggle kernels output b5wvix/astramis-best-detector-yolo11s-seg -p <папка>`

Если датасет не смонтировался в `/kaggle/input`, ноутбук скачивает его через
`kagglehub.dataset_download('b5wvix/astramis-slim')` сfallback на `kaggle datasets download`.

Локальный запуск: `pip install -q -U ultralytics kagglehub && python train_best.py`
(нужен доступ к тому же датасету или папка `/kaggle/input/astramis-slim`).

## Выходные файлы

- `best_yolo11s_seg_valfixed.pt` — веса под отдельным именем, существующие веса не перезаписываются;
- `train_ids.json`, `val_ids.json` — ID кадров сплита;
- `config.json` — все гиперпараметры и правило сплита;
- `metrics.json` — D и AP по классам;
- `val_predictions.csv` — `image_id,n,x1,y1,x2,y2,score,form` (только детекция).
