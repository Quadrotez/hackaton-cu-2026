#!/usr/bin/env python3
"""Только для чтения: быстрая проверка работы Соланы.

Запуск: python3 work/verify.py
Ничего не меняет, не коммитит, не отправляет.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'} {name}: {detail}")


def run_git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {exc}"


# 1. Состояние git
head = run_git("rev-parse", "HEAD")
remote = run_git("rev-parse", "origin/main")
status = run_git("status", "--short")
report("git", head == remote and not status,
       f"HEAD={head[:8]} origin/main={remote[:8]} status={status or 'чисто'}")

# 2. Данные и артефакты не должны попасть в git
tracked = run_git("ls-files", "dataset", "work", "*.pt", "*.pth", "submission.csv")
report("gitignore", not tracked, f"отслеживаемых данных/весов: {tracked or 'нет'}")

# 3. Сплит по object_id
split_files = list(WORK.glob("**/split.json"))
if not split_files:
    report("split", False, "work/**/split.json не найден")
else:
    sp = json.loads(split_files[0].read_text(encoding="utf-8"))
    def ids(key: str) -> set[str]:
        v = sp.get(key, [])
        if isinstance(v, dict):
            v = v.get("object_ids", v.get("objects", []))
        return {str(x) for x in v}
    tr, va = ids("train"), ids("val")
    overlap = tr & va
    report("split", bool(tr and va and not overlap),
           f"train={len(tr)} val={len(va)} пересечений={len(overlap)} "
           f"{list(overlap)[:5]}")

# 4. Признаки
feat_files = list(WORK.glob("**/features.csv"))
if not feat_files:
    report("features", False, "work/**/features.csv не найден")
else:
    with feat_files[0].open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    splits = Counter(r.get("split", "?") for r in rows)
    mats = Counter(r.get("material", "?") for r in rows)
    keys = [(r.get("image_id"), r.get("ann_id")) for r in rows]
    dups = len(keys) - len(set(keys))
    need = {"image_id", "material", "split", "object_id", "form"}
    cols = set(rows[0].keys()) if rows else set()
    report("features", bool(rows) and need <= cols and dups == 0,
           f"строк={len(rows)} сплиты={dict(splits)} "
           f"материалы={len(mats)} дублей={dups}")

# 5. Crops и модель
crops_dir = WORK / "crops"
crop_count = len(list(crops_dir.glob("**/*.png"))) if crops_dir.exists() else 0
report("crops", crop_count > 0, f"png-вырезок: {crop_count}")
models = [p for p in WORK.glob("**/*.pkl") if p.stat().st_size > 100]
report("model", bool(models),
       ", ".join(f"{p.name} ({p.stat().st_size // 1024} КБ)" for p in models) or "нет .pkl")

# 6. Пересчёт метрики по её предсказаниям (если уже есть)
preds = sorted(WORK.glob("**/val_*.csv")) + sorted(WORK.glob("**/*oracle*.csv"))
if not preds:
    report("metric", False, "work/val_*.csv нет — метрику пересчитать нечем")
else:
    labels = ROOT / "dataset" / "labels" / "real_train.json"
    for p in preds[:3]:
        cmd = [PY, str(ROOT / "score.py"), "check",
               "--labels", str(labels), "--submission", str(p)]
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
            text = (out.stdout + out.stderr).strip().replace("\n", " | ")
        except Exception as exc:  # noqa: BLE001
            text = f"ERROR {exc}"
        report(f"metric:{p.name}", "Score" in text, text[:400])

# 7. Журнал
journal = ROOT / "ЖУРНАЛ.md"
if journal.exists():
    txt = journal.read_text(encoding="utf-8")
    report("journal", len(txt) > 200, f"длина {len(txt)} симв., пометок D/M/V/S: {txt.count('M ')}")

failed = [n for n, ok, _ in results if not ok]
print("\nИТОГ:", "ВСЁ ОК" if not failed else f"ПРОВАЛИЛИ: {', '.join(failed)}")
sys.exit(1 if failed else 0)
