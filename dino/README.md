# Как получен 82.0385 (DINO +fusion, ветка `dino-82-0385`)

Публичный балл **82.0385** (было 78.97, +3.07). Отправлено 21.11.2026 ~21:02 с аккаунта Григория.

## Если коротко
База команды — `submission_best_verified.csv` (публичный 81.7147, OOF 74.86069).
К ней подмешаны вероятности формы и материала от отдельной 4-канальной
кроп-модели (RGB + замаскированная глубина) на DINO ViT-B/16.
Адаптер `fusion/sprint5_crop_adapter.py` из ветки `submit-81-71` проверил
OOF-гейт и собрал финальный CSV.

## Хронология

### 1. Контракт и инвентарь
Из `submit-sprint4` / `submit-81-71`: `work/sprint3_synth_backgrounds/allowed_synth.json`
(чистая синтетика по фолдам), `work/detector_folds_v1.json`, замороженные
артефакты из Telegram-архива `astramis_crop_handoff.zip` (SHA сверены с PINNED):
- `work/sprint5_cpu/base_7486069_oof.json` — 377 OOF-гипотез, 263 real-кадра;
- `work/sprint4_volume_gate/submission_best_verified.csv` — 342 тестовые гипотезы;
- маска-архивы `work/sprint1_agnostic_oof_v2/*`, `work/test_wbf_v1/*`.

Инвентарь собран командой (boxes-only, без GT- геометрии на инференсе):
```
.venv/bin/python -B fusion/sprint5_crop_adapter.py prepare --split oof --boxes-only \
  --out work/crop_inventory_oof.csv --training-contract-out work/crop_training_contract.json
.venv/bin/python -B fusion/sprint5_crop_adapter.py prepare --split test --boxes-only \
  --out work/crop_inventory_test.csv
```
Итог: 377 + 342 строк с `image_id, hyp_idx, fold, detection_sha256` — ключ совпадает с базой.

### 2. Кропы
`work/crop_model/prepare_inventory_crops.py`: 4-канальные кропы 128×128
(RGB + depth/2000, обнулено вне бокса с 10% паддингом) по box из инвентаря,
никаких GT-полигонов и object_id на инференсе. Датасет `b5wvix/astramis-inv-crops`.

### 3. Модель
Colab T4, ноут `one_cell.ipynb` (одна ячейка, чтобы не было NameError из-за
незапущенной ячейки):
- `timm.create_model('vit_base_patch16_224.dino', pretrained=True, num_classes=0)`;
- эмбеддинги 377 train + 342 test кропов: `emb (377, 768) (342, 768)`;
- 5 фолдов по `object_id` из `train_labels.csv`, на каждом LogisticRegression
  для form (bulk/solid) и material (6 классов), `class_weight='balanced'`;
- OOF-предсказания для своего фолда, test — усреднение 5 фолдов;
- выход: `dino_oof_probs.csv`, `dino_test_probs.csv` в точном формате адаптера
  (`p_bulk`, `p_material_{metal,wood,plastic,glass,paper,mineral}`, сумма = 1).

### 4. Гейт
```
.venv/bin/python -B fusion/sprint5_crop_adapter.py score \
  --oof-probs work/dino_oof_probs.csv --out work/dino_score.json
```
| вариант | OOF Score | дельта |
|---|---:|---:|
| baseline | 74.8607 | — |
| form | 75.0647 | +0.20 |
| material | 75.6502 | +0.79 |
| **both** | **75.8542** | **+0.99** ✅ |

`both` прошёл все проверки: pooled_score, ≥3 положительных фолда,
leave-one-object-out, form/material component. ResNet34-вариант ранее не проходил.

### 5. Export и отправка
```
fusion/sprint5_crop_adapter.py plan ... --variant both --out work/dino_deploy_spec.json
fusion/sprint5_crop_adapter.py export ... --variant both \
  --deploy-spec work/dino_deploy_spec.json --allow-unattested \
  --out work/dino_final_submission.csv
```
137 кадров, 342 детекции, 10 полей на объект. Загружено через веб-форму
StackMoreLayers — публичный балл **82.0385**.

## Файлы в ветке
- `dino_final_submission.csv` — отправленный сабмит (SHA `7328784ada0bd534f0bb74b108f6eeb3e061d301bf391360beaac0042d8fbe43`).
- `dino_oof_probs.csv`, `dino_test_probs.csv` — вероятности для адаптера.
- `dino_score.json` — полный отчёт гейта по фолдам.
- `one_cell.ipynb` — ноут Colab для воспроизведения.
- `STATUS.md` — журнал итераций.

## Уроки
- Две ошибки в первом ноуте: `mdl` вместо `m` и перебор всех ключей вместо среза батча
  (получилось 4524/3762 вместо 377/342). Исправлено в `one_cell.ipynb`.
- ResNet34 (с нуля и с ImageNet) давал material accuracy 0.53 против 0.66 у базы —
  в смесь не годится. DINO-эмбеддинги с линейным пробе дают нужный прирост.
- Слота не тратил, пока OOF не показал +0.5; отправка была только после гейта.
