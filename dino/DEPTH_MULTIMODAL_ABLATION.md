# RGB, depth и multimodal: честная OOF-абляция

Дата: 21.11.2026. Цель эксперимента — улучшить только классификационные
вероятности для уже замороженного инвентаря Sprint 5, не меняя ни одной
детекции, бокса, объёма, массы или решения `take`.

## Данные и изоляция

Использован версионированный `origin/crop-model:colab/inventory_crops.zip`:

- 377 OOF-слотов и 342 test-слота, ключ —
  `image_id, hyp_idx, fold, detection_sha256`;
- каждый кроп — `uint8 (4, 128, 128)`: RGB и нормированная замаскированная
  глубина;
- форма и материал обучаются отдельно в пяти внешних фолдах `object_id`;
- никакие test-метки, test-геометрия или generated-изображения не используются.

## Результаты на тех же OOF-слотах

Это диагностические классификационные метрики, а не официальный Score: для
официального Score нужен отсутствующий в Git frozen
`work/sprint5_cpu/base_7486069_oof.json`.

| Модель | Form balanced accuracy | Form Brier | Material macro recall |
| --- | ---: | ---: | ---: |
| Опубликованный DINO 82.0385 | 0.7982 | 0.1457 | 0.7493 |
| Только depth-descriptor | 0.6018 | — | 0.3911 |
| DINO RGB + depth-descriptor | 0.8300 | 0.1257 | 0.7131 |
| DINO + OpenCLIP + depth/color | **0.8433** | **0.1237** | **0.7854** |

Depth сам по себе слаб и не используется как замена RGB. Однако multimodal
вариант даёт устойчивый выигрыш формы в каждом диагностическом сравнении и
лучший pooled material recall.

## Nested DINO + OpenCLIP: контроль переобучения подбора

Отдельно проверен более строгий вариант: для каждого внешнего фолда параметры
линейных голов и power-calibration выбирались только на `GroupKFold` разбиении
его train-части по `object_id`. На пяти внешних фолдах он получил 0.8380 form
balanced accuracy и 0.7759 material macro recall. Результат подтверждает, что
сочетание эмбеддингов полезно, но он не превосходит основной multimodal-вариант
по форме (0.8433), поэтому не добавлен в кандидат для official gate. Скрипт:
`work/crop_model/nested_dino_clip_probe.py`.

## Консервативный кандидат

Для первого official gate подготовлены probability CSV
`mm_form_dino_material_{oof,test}_probs.csv` на сервере в
`/Users/conexl/Code/active/crop-model-20261121/work/candidates/`.

В них `p_bulk` берётся от DINO+OpenCLIP+depth/color, а все шесть
`p_material_*` байт-в-байт повторяют опубликованный DINO. Такой вариант
изолирует проверку улучшения D: он не меняет материал относительно решения
82.0385. Запрещено экспортировать или отправлять CSV, пока
`fusion/sprint5_crop_adapter.py score` не подтвердит положительный OOF gate.

## Воспроизведение depth-абляции

На сервере с PyTorch/MPS:

```sh
ENV=/Users/conexl/Code/active/astramis-ml-venv
"$ENV/bin/python" work/crop_model/dino_depth_probe.py \
  --archive colab/inventory_crops.zip --out work/depth_probe \
  --mode rgb_depth --device mps
```

Скрипт выпускает два строго-форматных CSV и JSON-отчёт. Предобученные веса
DINO и OpenCLIP должны быть зафиксированы в описании финального решения,
если multimodal-вариант пройдёт gate.
