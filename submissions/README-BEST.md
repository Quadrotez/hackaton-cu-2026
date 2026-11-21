# Лучший результат: 84.6908527185

Публичный балл StackMoreLayers, получен 22.11.2026 ~01:08 (аккаунты «Григорий Бычков»).
Базовый чемпион: 83.9070. Прирост: **+0.7838**.

## Сабмит

`submissions/submission_v27_second_0.2.csv` — 137 кадров, 226 гипотез, SHA-256
`c9917ed0a43e86f11ea53de964a45bd7e7f3f1119982798d622ffecfbc0a912d`.

## Воспроизведение

```bash
python3 fusion/prune_hypotheses.py --mode top2 \
  --out submissions/submission_v20_top2.csv
python3 fusion/second_score.py --value 0.2 \
  --in submissions/submission_v20_top2.csv \
  --out submissions/submission_v27_second_0.2.csv
```

Вход: `submissions/submission_83_91_champion.csv` из ветки `submit-83-91-champion`.

## Что и почему

1. **top-2 гипотезы на кадр** вместо трёх: чемпион выдавал 326 гипотез на 114 непустых реальных кадров, а реальный кадр содержит один объект. Лишние варианты формы были ложными срабатываниями в mAP50. Балл 83.9070 → 84.0582.
2. **Скор второй (страховочной) гипотезы = 0.2**. Primary сохраняет исходный скор (~0.99), поэтому class-agnostic match в метрике (а значит M/V/S) не меняется. В D важно только ordering внутри формы: все primary выше всех hedge. 0.2 и 0.3 дали синхронно 84.6909; 0.5 → 84.23, 0.7 → 83.76.

Модуляция скоров через DINO/CLIP p(form) (ветки v18/v19/v25) стабильно ухудшала балл — не использовать.

## Второй финалист

`submissions/submission_v20_top2.csv` — 84.0582. Оба сабмита отмечены для финального зачёта.
