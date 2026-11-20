"""Compare available probes, refit artifact already produced by astra_train."""
import hashlib
import json
import shutil
from pathlib import Path

W = Path(__file__).resolve().parent
models = {"DINO": ("astra_", "astra_dino.py", "astra_dino.safetensors")}
if (W/"astra_clip_metrics.json").exists():
    models["CLIP"]=("astra_clip_","astra_clip.py","astra_clip.bin")
if (W/"astra_clip_prior_metrics.json").exists():
    models["CLIP+text"]=("astra_clip_prior_","astra_clip.py","astra_clip.bin")
res={k:json.loads((W/(v[0]+"metrics.json")).read_text()) for k,v in models.items()}
def rank(k):
    v=res[k]; metal,n=v["v2"]["real"][1]["metal"]
    return (metal/n,v["v2"]["real"][0],v["primary"]["real"][0])
best=max(res,key=rank)
prefix,script,weight=models[best]
shutil.copyfile(W/(prefix+"candidate.pkl"),W/"astra_probe.pkl")
lines=[
    "Материалы: замороженный визуальный энкодер + линейная логистическая регрессия.",
    "DINO: Caron et al., Emerging Properties in Self-Supervised Vision Transformers, Facebook AI (2021).",
    "timm/vit_base_patch16_224.dino; источник: https://huggingface.co/timm/vit_base_patch16_224.dino",
    "868×768 float32: astra_embeddings.npy; ключ строки image_id_ann_id: astra_ids.json.",
    "24 кропа, исключённые primary, восстановлены по исходным маскам; astra_extra_embeddings.npy + astra_extra_ids.json.",
    "Маски сплитов: пересечение image_ids и object_ids по каждой части, с проверкой числа аннотаций.",
    "Train/val (real+synth): primary 685/183, v2 676/186. Для сравнения с baseline — только real val (49 в каждом).",
    "L2-нормализация; веса классов balanced. C по 4-fold StratifiedGroupKFold на объектах train, цель — pooled real macro recall.",
    "M — среднее recall по материалам с ненулевым support; — = нет примеров. Запись: верно/всего.",
    "CLIP: open_clip_torch 4.2.2, laion/CLIP-ViT-B-32-laion2B-s34B-b79K, 512 признаков. CLIP+text смешивает коэффициенты линейной головы с фиксированными текстовыми прототипами материалов; alpha по тому же внутреннему CV.",
]
for model,r in res.items():
    for split,base in (("primary",.549),("v2",.6713)):
        for scope in ("real","all"):
            m,counts=r[split][scope]
            delta=f" (baseline {base:.4f}, Δ {m-base:+.4f})" if scope=="real" else ""
            lines+=["",f"{model} {split}/{scope}: M={m:.6f}{delta}, C={r[split]['C']}, alpha={r[split].get('alpha',0)}"]
            for k,(a,b) in counts.items():
                val=f"{a/b:.2%}" if b else "—"
                lines.append(f"  {k:8} {val:>7}  {a}/{b}")
            if scope=="real" and "metal_predictions" in r[split]:
                lines.append("  Истинный metal → "+str(r[split]["metal_predictions"]))
lines+=["",f"Выбран {best}: приоритет metal recall v2, затем M v2 и primary. astra_probe.pkl: refit по всем 892 размеченным кропам, C={res[best]['refit_C']}.",
        "Метрики выше относятся к моделям, обученным только на train соответствующего сплита. Все семь реальных примеров metal на v2 — один объект b02.",
        "", "Инференс в среде соревнования без интернета:",
        f"Заранее положить в work/ {weight}, astra_probe.pkl, {script} и совместимые зависимости для целевой ОС/CPU.",
        "Версии локального DINO-стека: torch 2.15.0, torchvision 0.29.2, timm 1.0.31, scikit-learn 1.9.1.",
        "Источник тестового кропа: bbox с 10% полями, фон вне сегментации чёрный, resize 128×128 (OpenCV AREA при уменьшении максимальной стороны, иначе LINEAR).",
        "Скрипт читает RGB, применяет штатный eval transform энкодера и передаёт в probe. astra_embeddings.npy на тесте не нужен.",
        f"Из корня: PYTHONPATH=work/astra_packages .venv/bin/python -B work/{script} /path/to/crop.png",
        "DINO загружается из astra_dino.safetensors через pretrained_cfg_overlay.file при HF_HUB_OFFLINE=1.",
        "Размер DINO: %.1f MiB; SHA256: %s"%((W/"astra_dino.safetensors").stat().st_size/2**20,hashlib.sha256((W/"astra_dino.safetensors").read_bytes()).hexdigest()),
        "", "Если веса не помещаются: дистиллировать учителя в MobileNetV3-Small с линейной проекцией в размерность эмбеддинга.",
        "Обучать на train-кропах по cosine loss к L2-нормированному выходу учителя + cross-entropy по материалу; для аугментаций считать teacher target на той же версии кропа.",
        "Проверить оба object-disjoint сплита и metal recall до/после INT8, затем refit и экспорт только студента с головой. Это предложение, качество студента не измерялось."]
(W/"astra_report.txt").write_text("\n".join(lines)+"\n")
