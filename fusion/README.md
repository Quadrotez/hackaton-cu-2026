# Слияние сабмитов детекторов

`submission_fusion.py` берёт несколько готовых CSV-сабмитов (каждый — вывод своей модели/ансамбля)
и собирает из них один: рамки объединяются по IoU ≥ 0.5 внутри одной формы, вес рамки —
уверенность модели, итоговая уверенность — максимум по участникам (режим `--score-mode max`,
лучше на публичной части). Атрибуты материала/объёма/массы/`take` берутся от **первого**
(primary) сабмита — у него головы считаются эталонными.

```bash
python submission_fusion.py primary.csv model1.csv model2.csv ... out.csv
python submission_fusion.py --score-mode avg primary.csv model1.csv out.csv
```

НаLocal A/B: `submission_wbf6_final.csv` (78.41) + `submission_grisha_single_final.csv` (77.61)
→ 78.97 на публичной части. Простое отбрасывание рамок с confidence < 0.5 дало 72.44 — не делать.
