"""
Метрика кейса Astramis. Один файл для трёх задач:

1. Метрика на Kaggle: функция score(solution, submission, row_id_column_name).
2. Проверка своего решения на обучающих кадрах:
       python score.py check --labels dataset/labels/real_train.json --submission my_submission.csv
3. Подготовка файлов для Kaggle (только организаторам):
       python score.py kaggle --labels dataset/labels_hidden/real_test.json --out kaggle

Формат сабмита: одна строка на каждый тестовый кадр.
    image_id,prediction
    r_0141,212 305 298 371 0.91 solid plastic 500 25 1;402 280 520 344 0.77 bulk wood 350 60 1
    r_0152,none
Объекты кадра разделяются точкой с запятой. У каждого объекта 10 значений через пробел:
    x1 y1 x2 y2 score form material volume_ml mass_g take
    form: bulk | solid;  material: metal | wood | plastic | glass | paper | mineral;  take: 1 забрать, 0 объехать
Если на кадре ничего не найдено, пишется слово none.

Итог: Score = 100 * (0.40*D + 0.20*M + 0.25*V + 0.15*S)
    D  mAP50 по двум классам формы
    M  доля верно определённых материалов, усреднённая по материалам
    V  точность объёма: 1 при точном попадании, 0 при ошибке в 3 раза и больше
    S  решение «забрать / объехать»: среднее по тяжёлым (> 500 г) и лёгким объектам
M, V и S считаются по всем размеченным объектам: пропущенный объект даёт в них 0.
"""
import math

import numpy as np
import pandas as pd

FORMS = ("bulk", "solid")
MATERIALS = ("metal", "wood", "plastic", "glass", "paper", "mineral")
WEIGHTS = {"D": 0.40, "M": 0.20, "V": 0.25, "S": 0.15}
IOU_THR = 0.5
HEAVY_G = 500.0
MAX_PER_IMAGE = 30


class ParticipantVisibleError(Exception):
    """Ошибка, текст которой Kaggle показывает участнику."""


def _parse(cell, with_score, image_id):
    """Строка с объектами кадра -> список словарей."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    text = str(cell).strip()
    if text == "" or text.lower() == "none":
        return []
    out = []
    n = 10 if with_score else 8
    for part in text.split(";"):
        t = part.split()
        if not t:
            continue
        if len(t) != n:
            raise ParticipantVisibleError(f"{image_id}: у объекта должно быть {n} значений, получено {len(t)}: '{part.strip()}'")
        try:
            x1, y1, x2, y2 = (float(v) for v in t[:4])
            k = 4
            sc = 1.0
            if with_score:
                sc = float(t[4])
                k = 5
            form, mat = t[k].lower(), t[k + 1].lower()
            vol = float(t[k + 2])
            if with_score:
                float(t[k + 3])                      # масса: проверяем формат, в метрике не участвует
                take = int(float(t[k + 4]))
            else:
                take = int(float(t[k + 3]) <= HEAVY_G)
        except ValueError:
            raise ParticipantVisibleError(f"{image_id}: не удалось прочитать числа в '{part.strip()}'")
        if form not in FORMS:
            raise ParticipantVisibleError(f"{image_id}: неизвестная форма '{form}'")
        if mat not in MATERIALS:
            raise ParticipantVisibleError(f"{image_id}: неизвестный материал '{mat}'")
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, sc, vol)):
            raise ParticipantVisibleError(f"{image_id}: в предсказании есть бесконечность или NaN")
        if x2 <= x1 or y2 <= y1:
            raise ParticipantVisibleError(f"{image_id}: у рамки должно быть x2 > x1 и y2 > y1")
        if with_score and take not in (0, 1):
            raise ParticipantVisibleError(f"{image_id}: take должен быть 0 или 1")
        out.append({"box": (x1, y1, x2, y2), "score": sc, "form": form, "material": mat, "volume": vol, "take": take})
    if with_score and len(out) > MAX_PER_IMAGE:
        raise ParticipantVisibleError(f"{image_id}: больше {MAX_PER_IMAGE} объектов на кадре")
    return out


def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _average_precision(gt, pred, form):
    """AP при IoU 0.5 для одного класса формы (площадь под кривой точность-полнота)."""
    n_gt = sum(1 for objs in gt.values() for o in objs if o["form"] == form)
    if n_gt == 0:
        return None
    dets = sorted(((p["score"], img, p) for img, objs in pred.items() for p in objs if p["form"] == form),
                  key=lambda x: -x[0])
    used = {img: [False] * len(objs) for img, objs in gt.items()}
    tp = []
    for _, img, p in dets:
        best, best_j = 0.0, -1
        for j, g in enumerate(gt.get(img, [])):
            if g["form"] != form or used[img][j]:
                continue
            v = _iou(p["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= IOU_THR:
            used[img][best_j] = True
            tp.append(1)
        else:
            tp.append(0)
    if not tp:
        return 0.0
    tp = np.array(tp)
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / (np.arange(len(tp)) + 1)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def _match(gt_objs, pred_objs):
    """Сопоставление без учёта класса: предсказания по убыванию score, каждому лучший свободный объект."""
    match = [None] * len(gt_objs)
    for p in sorted(pred_objs, key=lambda x: -x["score"]):
        best, best_j = 0.0, -1
        for j, g in enumerate(gt_objs):
            if match[j] is not None:
                continue
            v = _iou(p["box"], g["box"])
            if v > best:
                best, best_j = v, j
        if best >= IOU_THR:
            match[best_j] = p
    return match


def components(gt, pred):
    """gt, pred: {image_id: [объекты]}. Возвращает словарь с D, M, V, S и итоговым Score."""
    aps = [ap for ap in (_average_precision(gt, pred, f) for f in FORMS) if ap is not None]
    D = float(np.mean(aps)) if aps else 0.0

    mat_hit = {m: [] for m in MATERIALS}
    vol, heavy, light = [], [], []
    for img, objs in gt.items():
        for g, p in zip(objs, _match(objs, pred.get(img, []))):
            mat_hit[g["material"]].append(1.0 if p is not None and p["material"] == g["material"] else 0.0)
            if p is None or p["volume"] <= 0 or g["volume"] <= 0:
                vol.append(0.0)
            else:
                vol.append(max(0.0, 1.0 - abs(math.log(p["volume"] / g["volume"])) / math.log(3.0)))
            (heavy if g["take"] == 0 else light).append(1.0 if p is not None and p["take"] == g["take"] else 0.0)
    per_mat = [float(np.mean(v)) for v in mat_hit.values() if v]
    M = float(np.mean(per_mat)) if per_mat else 0.0
    V = float(np.mean(vol)) if vol else 0.0
    parts = [float(np.mean(x)) for x in (heavy, light) if x]
    S = float(np.mean(parts)) if parts else 0.0
    total = 100.0 * (WEIGHTS["D"] * D + WEIGHTS["M"] * M + WEIGHTS["V"] * V + WEIGHTS["S"] * S)
    return {"D": D, "M": M, "V": V, "S": S, "Score": total}


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Метрика для Kaggle. solution: image_id, objects (и Usage); submission: image_id, prediction."""
    if "prediction" not in submission.columns:
        raise ParticipantVisibleError("В сабмите должна быть колонка prediction")
    if submission[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError("В сабмите повторяются image_id")
    sub = dict(zip(submission[row_id_column_name].astype(str), submission["prediction"]))
    gt, pred = {}, {}
    for img, cell in zip(solution[row_id_column_name].astype(str), solution["objects"]):
        gt[img] = _parse(cell, False, img)
        if img not in sub:
            raise ParticipantVisibleError(f"В сабмите нет кадра {img}")
        pred[img] = _parse(sub[img], True, img)
    return components(gt, pred)["Score"]


# ---------- всё, что ниже, нужно только для запуска на своём компьютере ----------

def labels_to_solution(labels_path):
    """Разметка в формате real_train.json -> таблица image_id, objects (и Usage, если есть)."""
    import json
    from pathlib import Path
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    by_img = {}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        at = a["attributes"]
        by_img.setdefault(a["image_id"], []).append(
            f"{x:.1f} {y:.1f} {x + w:.1f} {y + h:.1f} {at['form']} {at['material']} {at['volume_ml']:g} {at['mass_g']:g}")
    rows = []
    for im in sorted(data["images"], key=lambda i: i["file_name"]):
        row = {"image_id": Path(im["file_name"]).parent.name, "objects": ";".join(by_img.get(im["id"], [])) or "none"}
        if "leaderboard" in im:
            row["Usage"] = "Public" if im["leaderboard"] == "public" else "Private"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "kaggle"])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--submission")
    ap.add_argument("--out", default="kaggle")
    args = ap.parse_args()
    sol = labels_to_solution(args.labels)

    if args.command == "kaggle":
        out = Path(args.out)
        out.mkdir(exist_ok=True)
        sol.to_csv(out / "solution.csv", index=False)
        pd.DataFrame({"image_id": sol["image_id"], "prediction": "none"}).to_csv(out / "sample_submission.csv", index=False)
        print(f"Записано: {out / 'solution.csv'} ({len(sol)} кадров) и {out / 'sample_submission.csv'}")
        if "Usage" in sol:
            print("Кадров по частям:", sol["Usage"].value_counts().to_dict())
    else:
        if not args.submission:
            raise SystemExit("Укажите --submission файл.csv")
        subm = pd.read_csv(args.submission)
        try:
            subsets = [("все кадры", sol)]
            if "Usage" in sol:
                subsets += [(u, sol[sol["Usage"] == u]) for u in ("Public", "Private")]
            for name, part in subsets:
                sub_d = dict(zip(subm["image_id"].astype(str), subm["prediction"]))
                gt = {i: _parse(c, False, i) for i, c in zip(part["image_id"], part["objects"])}
                missing = [i for i in gt if i not in sub_d]
                if missing:
                    raise ParticipantVisibleError(f"В сабмите нет кадров: {missing[:5]} (всего {len(missing)})")
                pred = {i: _parse(sub_d[i], True, i) for i in gt}
                c = components(gt, pred)
                score(part, subm, "image_id")          # та же проверка, что на Kaggle
                print(f"{name:10s} Score {c['Score']:6.2f} | D {c['D']:.3f}  M {c['M']:.3f}  V {c['V']:.3f}  S {c['S']:.3f}")
        except ParticipantVisibleError as e:
            raise SystemExit(f"Ошибка в сабмите: {e}")
