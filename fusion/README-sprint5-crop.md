# Sprint 5: A/B классификатора предсказанных кропов

Адаптер на CPU, не обучает модель, не вызывает Kaggle. База — полный объёмный
OOF Sprint 4 (`Score=74.86069298423111`) и замороженный
`work/sprint4_volume_gate/submission_best_verified.csv`. Это оценка **базы**,
не результат классификатора кропов. Реальных таблиц вероятностей пока нет:
новый OOF, решение о деплое и новый сабмит не получены.

## Prepare / контракт для Григория

Из корня репозитория, Python 3.11+ с `numpy` и `pandas`:

```sh
P=.venv/bin/python
"$P" -B fusion/sprint5_crop_adapter.py prepare --split oof \
  --out /path/to/oof_inventory.csv --training-contract-out /path/to/training.json
"$P" -B fusion/sprint5_crop_adapter.py prepare --split test --out /path/to/test_inventory.csv
```

OOF по умолчанию: `work/sprint5_cpu/base_7486069_oof.json` (реконструкция CPU-агента).
`--oof-base` принимает копию полного JSON `image_id -> ordered list of detection dicts`,
включая пустые списки. Проверяются компоненты официальной метрики и канонический
SHA-256 JSON (отступы и порядок ключей несущественны; порядок детекций существенен).
Простой `selected_oof.json` из Sprint 4 form rank не подходит.

В inventory одна строка на детекцию. `hyp_idx` — неизменный **0-based** слот в
полном базовом списке или в порядке через `;` test CSV, **не** ранг по score.
`detection_sha256` привязывает слот ко всем атрибутам данной базы; копировать из
prepare, а не рассчитывать в классификаторе. Для OOF порядок кадров официальный
(`file_name` в real labels), для test — порядок frozen CSV. Пустой кадр не даёт
строк вероятностей, но остаётся в базе/метрике/финальном CSV.

По умолчанию prepare добавляет проверенные указатели предмасок:

- `mask_archive` (путь относительно корня), `mask_archive_sha256`, `mask_key` (ключ NPY
  в NPZ, равен `image_id`), `mask_slot` (**0-based** в массиве архива), `mask_height`,
  `mask_width`, `donor_x1`, `donor_y1`, `donor_x2`, `donor_y2`, `rgb_path`, `depth_path`.
- OOF: raw `n640` **своего внешнего фолда**. Test: raw WBF. Для базовых слотов
  проверяется соответствие исходным якорям; для дописанной альтернативной формы —
  тот же донор Sprint 4 (якорь противоположной формы с лучшим IoU ≥ .5,
  при равенстве первый; на test якорь предварительно привязан к raw WBF).
  `mask_slot` **не обязан** совпадать с `hyp_idx`. Альтернативы могут
  разделять один донор: вычислить posterior один раз можно, но выдать строку
  на каждый слот, с тем же `p_bulk` независимо от `base_form`. `x1...y2` относятся
  к базовой детекции, не к маске донора.
- Рекомендуемый единый RGB-D crop: используйте bbox донора + 10% по каждой стороне,
  одна область для RGB, глубины и маски. Ни GT box, ни GT segmentation/mask, ни
  GT match, ни object ID/GT материал не используются при OOF/test inference,
  в том числе на ложных детекциях. `--boxes-only` явно отключает доноры; для bbox-only
  модели используются только `x1...y2`. Не смешивайте crop-режимы между OOF и test.

Пример чтения локально после сверки checksum архива (OpenCV и NumPy; `row` из
`csv.DictReader`, запуск из корня). Изображения не экспортируются адаптером:

```python
import math
import cv2
import numpy as np

h, w = int(row["mask_height"]), int(row["mask_width"])
with np.load(row["mask_archive"], allow_pickle=False) as z:
    packed = z[row["mask_key"]][int(row["mask_slot"])]
    mask = np.unpackbits(packed, axis=-1, count=w, bitorder="big").astype(bool)
bgr = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
depth = cv2.imread(row["depth_path"], cv2.IMREAD_UNCHANGED)  # uint16, мм, aligned to RGB
assert bgr is not None and depth is not None
assert bgr.shape[:2] == depth.shape == mask.shape == (h, w) and depth.dtype == np.uint16
x, y, x2, y2 = (float(row[f"donor_{k}"]) for k in ("x1", "y1", "x2", "y2"))
bw, bh = x2 - x, y2 - y
l, t = max(0, math.floor(x - .1 * bw)), max(0, math.floor(y - .1 * bh))
r, b = min(w, math.ceil(x + 1.1 * bw)), min(h, math.ceil(y + 1.1 * bh))
m = mask[t:b, l:r]
assert m.any()
rgb = np.where(m[..., None], bgr[t:b, l:r, ::-1], 0)
depth_mm = np.where(m, depth[t:b, l:r], 0)
```

### Входные CSV вероятностей (строго)

OOF, пример с **вымышленными** вероятностями и SHA (в работе скопировать реальный
идентификатор и SHA из inventory):

```csv
image_id,hyp_idx,fold,detection_sha256,p_bulk,p_material_metal,p_material_wood,p_material_plastic,p_material_glass,p_material_paper,p_material_mineral
example_frame,0,1,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,0.18,0.02,0.90,0.03,0.02,0.02,0.01
```

На test удалить столбец `fold`, остальные одинаковы. `p_bulk=P(bulk | crop)`,
`P(solid)=1-p_bulk`; шесть material-колонок — вероятности взаимно исключающих
классов, не логиты, не порядок индексов из произвольного даталоадера. Все значения
конечны и в `[0,1]`; сумма шести равна 1 с допуском `1e-6`, без автонормализации.
Автоматически проверяются заголовок, порядок prepare, уникальность, точное
покрытие, индекс, fold и digest. Пропуск, дубль или лишняя строка, неверный fold,
перестановка или NaN/Inf — ошибка, не fallback на baseline.

### Обучение / происхождение OOF

- `training.json` — **allow-list**, не список фактически использованных сэмплов.
  Для fold `k` брать real train только из `limits[k].real` (detector `train_image_ids`
  ∩ real labels). Можно real-only; если добавлять synth, использовать лишь
  `limits[k].allowed_synth` из `work/sprint3_synth_backgrounds/allowed_synth.json`
  (учитывает фон из held-out real empty-серий и исключённые mixed synth).
  Не брать held-out кадры, даже пустые, или изображения с held-out object ID.
  Тот же outer mask действует на препроцессинг, калибровку и fine-tuning.
  Если нужно выбирать гиперпараметры, использовать отдельный split внутри outer train
  с группировкой object ID и соответствующими ограничениями; не подбирать по pooled OOF.
- Каждая OOF строка — inference модели, обученной для **её** внешнего фолда.
  Для test отдельная модель на полном real train; опциональный synth —
  объединение fold allow-lists (`limits.test.allowed_synth`), без test кадров.
  GT training crop внутри разрешённого train — не разрешение на GT inference crop.
- Опциональный JSON `--oof-attestation`/`--test-attestation`:
  `version: 1`, `split: "oof"` или `"test"`, `dataset` — дословно из `training.json`,
  `base_sha256` — SHA-256 **файла** нужной базы, `probabilities_sha256` — SHA-256
  CSV. `models` содержит ключи `"1"`…`"5"` для OOF или только `"test"` для test.
  На модель: `train_image_ids` (фактически использованные ID, включая данные,
  использованные при выборе/калибровке; без повторов), `inference_crop_source`
  (`"predicted_mask"` или `"predicted_box"`), `checkpoint_sha256` (уникальный SHA
  байтов чекпоинта; для нескольких голов можно хэшировать один bundle); при наличии
  локального артефакта `checkpoint_path` (абсолютный или относительно аттестации).
  Адаптер сверяет метки по SHA, заявленные train IDs и их object IDs по маскам,
  артефакт по SHA при доступном пути, а также привязку входов. Это **не доказывает**
  фактическую историю обучения или запуска inference; CSV и даже аттестация не
  заменяют проверяемую историю пайплайна. Без аттестации отчёт явно `unattested`.

## Score / решение о deploy

```sh
"$P" -B fusion/sprint5_crop_adapter.py score --oof-probs /path/to/oof_probs.csv \
  --oof-attestation /path/to/oof_attestation.json --out /path/to/crop_score.json
```

Фиксировано до получения вероятностей: **form** только меняет score:
`logit(score) += alpha * (2*P(base_form) - 1)`, `alpha=.35`. Общий множитель
кадра откатывает изменение до 99% ближайшей границы пересечения; не допускаются
изменения ранга или score ties внутри кадра. **material** меняет только label:
другой top-1 ≥ `.85`, отрыв от top-2 ≥ `.35`, вероятность базового label ≤ `.15`.
**both** применяет оба независимых изменения. Никакой смены `form`, удаления,
перестановок, изменений bbox/volume/mass/take/take_p. В OOF проверяется совпадение
слотов official `_match`, в test — round-trip CSV и стабильность порядка score.

Отчёт сравнивает baseline/form/material/both: `Score, D, M, V, S` и AP50 каждой
формы, pooled и per-fold, включая отрицательные кадры; `D` — официальный pooled
macro AP50 по формам. Отдельно изменения на unmatched предсказаниях, списки
distinct matched object IDs с изменениями/исправлениями/ухудшениями, а также
leave-one-object-out (переоценка после удаления всех кадров объекта, без
переобучения, multi-object кадр удаляется целиком). Для положительного gate:
pooled `ΔScore > 1e-9`, ≥3 положительных внешних фолда и худший LOO `ΔScore > 1e-9`.
Если активен form, также `ΔD > 1e-9` и изменения matched score минимум на двух
объектах; если material, `ΔM > 1e-9` и исправления минимум на двух объектах.
Число объектов с изменениями score — поддержка, не причинная атрибуция выигрыша AP.

Предварительно зафиксируйте **один** primary вариант (например, `both`) и
конфигурацию; `export` не выбирает победителя среди A/B. Для deploy обязательны
ручное указание варианта и отдельный локальный JSON spec. `plan` генерирует
его из файлов без ручного ввода хэшей, не перезаписывает существующий файл,
не вычисляет метрики и не выдаёт разрешение на export. Поля spec:
`version: 1`, `variant`, `config` (точно `CONFIG` из адаптера),
`oof_base_sha256`, `test_base_sha256`, `oof_probabilities_sha256`,
`test_probabilities_sha256`, `oof_attestation_sha256`, `test_attestation_sha256`
(байтовые SHA-256 файлов; `null` для отсутствующих аттестаций).
Файл test base должен иметь SHA-256
`e9c38fe9349c95fac7a6191c55fb8b936f30c7302d356d709ac666b36a65c49f`.

```sh
"$P" -B fusion/sprint5_crop_adapter.py plan --variant both \
  --oof-probs /path/to/oof_probs.csv --oof-attestation /path/to/oof_attestation.json \
  --test-probs /path/to/test_probs.csv --test-attestation /path/to/test_attestation.json \
  --out /path/to/deploy.json
"$P" -B fusion/sprint5_crop_adapter.py export --variant both --deploy-spec /path/to/deploy.json \
  --oof-probs /path/to/oof_probs.csv --oof-attestation /path/to/oof_attestation.json \
  --test-probs /path/to/test_probs.csv --test-attestation /path/to/test_attestation.json \
  --out /path/to/crop_candidate.csv
```

Export пересчитывает OOF, не доверяет сохранённому отчёту и не пишет candidate
без положительного gate. Также проверяются spec и test CSV; отсутствующие
аттестации требуют явного `--allow-unattested`, который **не** отменяет gate
или привязку SHA. Нет автоэкспорта, копирования базы как «нового» решения или
перезаписи результатов. Все `--out` создают новый файл; Kaggle submit отдельно.
Быстрый путь после заполнения таблиц: `score → проверка заранее выбранной ветки →
plan → export`. Время вычисления без `prepare`, обучения, checkpoint I/O и
ручной проверки — см. smoke ниже.

## Локальные артефакты и тесты

- Для работы нужны `score.py`, `sample_submission.csv` (test CSV порядок),
  `dataset/labels/{real_train,synth_train}.json`, `work/detector_folds_v1.json`,
  `work/sprint3_synth_backgrounds/allowed_synth.json`, соответствующая база.
  Если доступен `work/sprint4_form_rank/selected_oof.json`, выполняется ещё
  сверка не-объёмных полей; точный канонический хэш OOF обязателен независимо.
- Для prepare с указателями OOF дополнительно
  `work/sprint1_agnostic_oof_v2/{mask_index.json,predictions.json}` и
  `work/detector_oof_v1/fold_*/evaluation/{base_detections.jsonl,base_masks.npz}`.
  Для test: `work/sprint1_grisha/submission_fused_max_7897_rescue_small.csv`,
  `work/submission_wbf6_final.csv`, `work/test_wbf_v1/{base_detections.jsonl,base_masks.npz}`
  и пути к `dataset/real/test/*/color.*`. Для inference нужны RGB/Depth файлы
  по inventory и, для режима `predicted_mask`, указанные NPZ. Ни сырой датасет,
  ни сабмиты, инвентари, OOF/test матрицы вероятностей, ни веса не добавлять в Git.
  `score` и `export` не требуют архивов масок или RGB-D; им нужны уже заполненные
  вероятности.
- `.venv/bin/python -B -m unittest -v fusion.test_sprint5_crop_adapter`:
  синтетические проверки отказов, сортировки и инвариантов, без записи project
  метрик. Smoke `prepare` на замороженных базах: 377 OOF и 342 test слота с
  проверенными указателями предмасок. Для smoke отдельно сгенерированы
  искусственные таблицы без GT: Form — `.90` для базового bulk (`.10` для solid)
  и равномерный материал; Material — `p_bulk=.50` и `.90` metal (`.02` каждый
  другой). Это проверки входов, не качество новых моделей. Обе таблицы прошли
  `score`, обе получили отказ `export` по OOF gate без файла/fallback. Проверка
  сериализации на frozen test в памяти: Form меняет только токен 4 (score),
  Material — только токен 6 (material), индексы с нуля.

| Smoke, CPU wall time | `score` | `plan` | `export` (отказ) | Сумма |
|---|---:|---:|---:|---:|
| Form | 0.302 с | 0.180 с | 0.297 с | 0.779 с |
| Material | 0.297 с | 0.175 с | 0.305 с | 0.777 с |

`prepare` отдельно: OOF 0.253 с, test 0.219 с. Временá включают старт Python,
не включают человеческий выбор; цель вычислительного пути — ≤10 с при локальных
файлах. Успешный production deploy не измерялся. Синтетические CSV/отчёты не
сохранялись в репозитории и не являются результатами A/B.

Положительный end-to-end контроль (по умолчанию пропущен):

```sh
SPRINT5_ORACLE_SMOKE=1 .venv/bin/python -B -m unittest -v \
  fusion.test_sprint5_crop_adapter.FrozenOracleSmoke
```

Это **GT-oracle fixture, не ML**: на frozen OOF только для этого теста
`P(GT material)=.99` при официальном match, иначе `.99` у базового material.
В test fixture принудительно меняются два label. `plan` и `export
--allow-unattested` запускаются через настоящий CLI, затем
`work/submission_tools.py` проверяет 137 кадров / 342 детекции / 10 `none`.
Изменены ровно два токена material, SHA исходного test CSV не изменился.
Замер: plan 0.182 с, export 0.303 с, validator 0.175 с. Oracle-метрики не
выводятся и не сохраняются; все файлы находятся во временной папке и удаляются.
