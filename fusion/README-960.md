# Astramis: fusion YOLO960 A/B

`submission_fusion.py` — неизменённый скрипт из `origin/submission-fusion` (`17f3705`).
Новый запуск — `yolo960_ab.py`, настройки — `yolo960_ab.json`. Скрипт не обучает модели,
не отправляет сабмиты и не меняет старые CSV. Для каждого нового детектора он использует
его сегментационные маски и те же замороженные головы DINO+CLIP / RGB-D из решения 78.41
(`work/astra_joint04_heads.pkl`).

| Вариант | Детекторы |
|---|---|
| A | 5 × YOLO11m-seg (960) + 5 × YOLO11l-seg (960) + старый YOLO11s-seg (640) |
| B | A + 5 × YOLO11n-seg (640), вес координат каждого `0.3 × confidence` |

В обоих вариантах эталон атрибутов — проверенный CSV одиночной YOLO11s (`work/submission_grisha_single_final.csv`):
его `material/volume/mass/take` сохраняются при совпадении. Если YOLO11s не участвовала,
атрибуты берутся у самого уверенного основного детектора в кластере. Как в исходном fusion,
сравниваются рамки одной формы с IoU ≥ 0.5, максимум один голос от модели;
итоговая уверенность — `max` по основным моделям. В B nano уточняет координаты только у уже
найденных объектов, не создаёт новых боксов, не меняет уверенность или атрибуты.
Старый CSV WBF6, который уже включает nano, не добавляется в A.

## Подключение весов

После завершения запусков скачайте артефакты в соответствующие папки:

```bash
kaggle datasets download -d v1tush/astramis-yolo11m-960-5folds \
  -p work/kaggle/astramis-yolo11m-960-5folds/download
kaggle kernels output v1tush/astramis-yolo11l-960-5folds \
  -p work/kaggle/astramis-yolo11l-960-5folds/output --page-size 200 \
  --file-pattern '(^|/)(last\.pt|run\.json|detector_folds_v1\.json|cv_folds\.json)$'
```

Из архива m разложите `yolo11m_960_fX.pt` в
`work/kaggle/astramis-yolo11m-960-5folds/output/astramis-cv/fold_0X/weights/last.pt`.
Выход l уже имеет структуру `output/astramis-cv/fold_0X/weights/last.pt`.
Пути в конфиге считаются от корня репозитория, `{fold:02d}` разворачивается в `01`–`05`.

Из корня репозитория:

```bash
python3 fusion/yolo960_ab.py --check
work/venv_yolo/bin/python fusion/yolo960_ab.py --variant both --device mps --batch 2
```

`--check` проверяет наличие, ZIP CRC и уникальность SHA-256 чекпойнтов, в том числе
сверяет SHA-256 s640 с известным. Инференс не запускается. Для CPU используйте
`--device cpu`, для CUDA — `--device 0`. `--variant A --check` проверяет 11 чекпойнтов,
без `--variant A` проверяются 16.
Результаты: `work/submission_yolo960_ab/submission_A.csv`, `submission_B.csv`,
промежуточные `per_model/` и SHA-256 входов в `inputs.json`. Для повторного запуска укажите
новый каталог через `--out`.

### Строгий OOF для новых весов

```bash
work/venv_yolo/bin/python fusion/evaluate_yolo960_oof.py --device mps --batch 2
```

Для каждого валидационного кадра голосуют только модели его фолда, а головы и
`rescue-small` используют только train этого фолда. `evaluate_yolo960_oof.py`
включает группу только при сверке экспортированных манифестов и `run.json`.
Поэтому это **не** замер полного тестового ансамбля: текущий архив m960 не содержит
манифеста обучения, а s640 не имеет совместимого пятифолдового разбиения. Отчёт
и OOF-предсказания сохраняются в `work/yolo960_strict_oof/`.

## A/B постобработки `volume / mass / take`

Чтобы получить рядом базовый `submission_A.csv` и `submission_A_post.csv`, не меняя
детекции, confidence, форму и материал:

```bash
python3 fusion/yolo960_ab.py --variant A --postprocess best --check
work/venv_yolo/bin/python fusion/yolo960_ab.py --variant A --device mps --batch 2 \
  --postprocess best --out work/submission_yolo960_A_attr
```

`best` читает рекомендацию из `work/astramis_attr_v2/policy.json`; путь можно заменить
через `--postprocess-model`. Без флага поведение прежнее. `inputs.json` фиксирует
выбранный режим и SHA-256 политики. К готовому fusion CSV можно применить ту же политику
без повторного инференса:

```bash
python3 fusion/attribute_postprocess.py --mode best \
  --input work/submission_yolo960_ab/submission_A.csv \
  --out work/submission_yolo960_ab/submission_A_post.csv
```

Режимы для ручного A/B: `density` заменяет `take` вероятностью массы ≤500 г,
`density-small` добавляет к этому правилу допуск по малому объёму,
`rescue-small` меняет только отказы исходной головы при малом объёме **и** высокой
вероятности лёгкого объекта. Плотность берётся по одному наблюдению на физический
предмет, с большим весом совпадающих формы и материала. `bulk` отдельно включает
сглаженную монотонную калибровку объёма, пропорционально меняющую массу;
`density+bulk` и `density-small+bulk` оценивают оба изменения вместе. В комбинированных
режимах `take` считается от исходного объёма: так он калибровался.

Воспроизвести вложенный OOF и переобучить только постобработку (новый `--out` обязателен
для повторного запуска; YOLO не запускается):

```bash
work/venv_yolo/bin/python fusion/evaluate_attributes.py \
  --out work/astramis_attr_v2_recheck
```

Внешний тест — сохранённые детекции на 263 real-кадрах из исходных пяти фолдов.
Во внутренних фолдах материал и объём заново предсказываются отдельными головами на
замороженных признаках **размеченных** масок, а плотности, пороги и кривая калибровки
считаются без внешней валидации. Размеченные маски — приближение на внутреннем этапе:
внешний YOLO уже видел внутренние объекты и потому там не используется.
`report.json` содержит результаты по фолдам, проверку с удалением каждого тяжёлого
предмета из оценки и отдельную проверку с его удалением из обучения плотности и выбора
порогов (внутренние признаки последней проверки остаются фиксированными).
