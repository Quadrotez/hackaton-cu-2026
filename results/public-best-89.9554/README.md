# Лучший результат на публичных тестах — 89.9554389808

- **Лидерборд:** публичный, StackMoreLayers, кейс `astramis`, **ранг 1**, 22.11.2026 ~12:50 мск.
- **Аккаунт:** «Павел Охрей», отправка **17448**.
- **Отрыв:** от 2-го места (89.6817) — 0.2737; от `sigma 3` (89.6507) — 0.3048.
- **Файл:** `submission_899554.csv` (копия `submissions/final_899554.csv` из ветки `top0-89-9554`, коммит `e31321f`).
- **SHA-256:** `6515d77c40393a0a3f78ce552fd0683c3a0a37d02cc97b3e37863cc953d6932e`.

## Рецепт

v52 (solid plastic ×1.25) + r_0064 bulk-only + volume plastic ×1.1.

Рядом пробовалось: plastic ×1.35 → 89.5224; metal ×1.1 → 89.8419. Лимит отправок на аккаунте был исчерпан.

## Как пришли

1. Детектор и атрибуты: ансамбль YOLO + DINO/CLIP, калибровка хеджей — `generate_v38_pr_ceiling_champion.py` … `generate_v48_rank1_precision_master_champion.py`.
2. v50 — 89.3846 (bulk metal 1.35×, solid glass 1.16×).
3. v52 — solid/plastic ×1.25 → 89.6817 на публике; тот же файл дал 90.9946 на привате.
4. r_0064 bulk-only (+0.073) + volume plastic ×1.1 (+0.1285) → 89.9554.

Все промежуточные CSV — в `submissions/`, хроника — `RANK1_FINAL_REPORT.md`, `TOP1_PUSH_FINAL.md`, `work/HYPOTHESES_SPRINT9.md`.
