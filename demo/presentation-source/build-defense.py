"""Build the offline HTML and editable PPTX from one layout. Requires python-pptx.

Pass --chrome /path/to/chrome to also export the five-page PDF.
"""
import argparse
from base64 import b64encode
from html import escape
from pathlib import Path
import subprocess
import tempfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt


DEMO = Path(__file__).resolve().parents[1]
BACKGROUNDS = DEMO / "assets" / "defense-backgrounds"
INK, PLUM, MUTED = "211828", "403048", "4E4053"
PAPER, LIGHT, HAIR = "F4F2E8", "FFFDF7", "D8D4C4"
GOLD, OLIVE, MAUVE = "F1D374", "465237", "68516E"
slides = []


def box(x, y, w, h, color=PAPER):
    slides[-1].append(("box", x, y, w, h, color))


def text(x, y, w, value, size=26, color=INK, bold=False):
    slides[-1].append(("text", x, y, w, size, color, value, bold))


def page(number, label, title, source):
    slides.append([])
    slides[-1].append(("image", 0, 0, 1280, 720, f"slide-{number}.jpg"))
    if label:
        text(50, 36, 350, label, 18, INK, True)
    box(1156, 28, 74, 32, PLUM)
    text(1167, 35, 58, f"0{number} / 05", 16, LIGHT, True)
    if title:
        text(50, 73, 1180, title, 46, INK, True)
    box(50, 666, 1180, 28, PAPER)
    text(61, 672, 1158, source, 14, MUTED)


page(1, "", "",
     "BRIEF.md · gen_submission/README.md · origin/submit-82-1097-gen-material · f02779a")
text(102, 226, 685, "Что под щёткой?", 46, INK, True)
text(104, 290, 650, "Робот решает: убрать или объехать.", 26, INK, True)
box(102, 356, 622, 103, PLUM)
text(124, 375, 582, "Из кадра RGB-D — рамка, свойства", 26, LIGHT)
text(124, 413, 582, "предмета и действие робота.", 26, LIGHT)
box(795, 208, 435, 306)
box(795, 208, 435, 7, GOLD)
text(820, 243, 388, "ПУБЛИЧНЫЙ ЛИДЕРБОРД", 18, PLUM, True)
text(817, 282, 390, "82,1097", 72, INK, True)
text(820, 382, 380, "балла из 100", 26, PLUM, True)
text(820, 444, 380, "Лучший сабмит ветки", 26, MUTED)
box(305, 551, 925, 57, PAPER)
text(323, 564, 891, "До 500 г включительно — убрать. Тяжелее — объехать.", 26, INK, True)
text(415, 623, 710, "Команда Astramis · Научная орбита 2026", 18, INK, True)

page(2, "РЕШЕНИЕ И ПРОВЕРКА", "От кадра к решению",
     "TODO.md · ИНСТРУКЦИЯ.md · dino/README.md · gen_submission/gen_model.ipynb и gen_score.json")
for i, (cap, label, note) in enumerate((
    ("01  КАМЕРА", "Кадр RGB-D", "Готовые рамки"),
    ("02  ПРИЗНАКИ", "DINO ViT-B/16", "Только RGB-кроп"),
    ("03  АТРИБУТЫ", "Форма и материал", "Две линейные модели"),
    ("04  САБМИТ", "Сборка CSV", "Объём, масса, take из базы"),
)):
    x = 50+i*298
    box(x, 175, 286, 154)
    box(x, 175, 286, 5, MAUVE)
    text(x+14, 195, 257, cap, 18, MUTED, True)
    text(x+14, 230, 257, label, 26, INK, True)
    text(x+14, 277, 257, note, 18, MUTED)
box(50, 351, 1180, 131, PLUM)
text(71, 376, 1138, "5 фолдов по предметам, а не по кадрам", 26, LIGHT, True)
text(71, 420, 1138, "180 PNG — только для материала; происхождение не подтверждено.", 26, LIGHT)
text(50, 502, 1138, "ЭТАПЫ 0–8", 18, INK, True)
box(50, 540, 1180, 79)
for i, (label, status) in enumerate((
    ("0 Бейзлайн", "ориентир"), ("1 Среда", "Kaggle"), ("2 Сплит", "по ID"),
    ("3 Кропы", "готовые"), ("4 Материал", "DINO"),
    ("5 Объём", "из базы"), ("6 Решение", "из базы"),
    ("7 Замеры", "локально"), ("8 Сборка", "CSV"),
)):
    x = 50+i*132
    if i:
        box(x, 552, 1, 55, HAIR)
    text(x+10, 552, 110, label, 18, INK, True)
    text(x+10, 579, 110, status, 18, MUTED)

page(3, "МЕТРИКА И БЕЙЗЛАЙН", "Материал — слабое место бейзлайна",
     "BRIEF.md:21–26 · score.py:19–24, 135–172 · значения из описания кейса")
box(50, 158, 1180, 78, PLUM)
text(70, 181, 1135, "Score = 100 × (0,40D + 0,20M + 0,25V + 0,15S)", 30, LIGHT, True)
text(64, 262, 670, "КОМПОНЕНТА И ВЕС", 18, INK, True)
text(760, 262, 427, "БЕЙЗЛАЙН · ШКАЛА ОТ 0 ДО 1", 18, INK, True)
for i, (symbol, weight, label, v, value, c) in enumerate((
    ("D", "40%", "Рамка и форма · mAP50", .86, "0,86", OLIVE),
    ("M", "20%", "Материал · среднее по классам", .17, "0,17", PLUM),
    ("V", "25%", "Объём в мл · учитываем ошибку в разах", .62, "0,62", OLIVE),
    ("S", "15%", "Действие · лёгкие и тяжёлые поровну", .42, "0,42", MAUVE),
)):
    y = 295+i*70
    box(50, y, 1180, 60, "F5E1A0" if symbol == "M" else PAPER)
    text(65, y+14, 45, symbol, 26, c, True)
    text(115, y+14, 70, weight, 26, INK, True)
    text(207, y+14, 523, label, 26, INK)
    box(760, y+24, 330, 12, HAIR)
    box(760, y+24, round(330*v), 12, c)
    text(1112, y+14, 101, value, 26, c, True)
text(50, 599, 1178, "≈59 баллов по описанию · 59,6 из округлённых компонент", 26, INK, True)

page(4, "РЕЗУЛЬТАТ", "На тесте прирост меньше",
     "gen_submission/README.md и gen_score.json · dino/README.md и dino_score.json · сверка двух CSV")
box(50, 193, 1180, 339)
text(80, 219, 402, "ГДЕ ПРОВЕРЯЛИ", 18, MUTED, True)
text(480, 219, 409, "БАЛЛ: БЫЛО → СТАЛО", 18, MUTED, True)
text(907, 219, 292, "ПРИРОСТ, БАЛЛЫ", 18, MUTED, True)
box(72, 256, 1136, 105, "F5E1A0")
text(80, 289, 389, "Публичный лидерборд", 26, INK, True)
text(480, 289, 408, "82,0385 → 82,1097", 26, INK, True)
text(907, 278, 292, "+0,0712", 46, INK, True)
text(80, 398, 389, "Локальная проверка (OOF)", 26, INK, True)
text(480, 398, 408, "75,8542 → 81,1336", 26, INK, True)
text(907, 387, 292, "+5,28", 46, INK, True)
box(80, 463, 1118, 1, HAIR)
text(80, 485, 1118, "В публичном CSV поменялся только материал у 17 из 342 детекций.", 26, INK)
text(50, 573, 1180, "≈59 — ориентир из описания кейса, не замер на том же тесте.", 26, INK, True)

page(5, "ОШИБКИ И ВЫВОД", "Проверяем на новых предметах",
     "ЖУРНАЛ.md:8–17,24–28 · gen_submission/README.md и gen_score.json")
for i, (cap, line) in enumerate((
    ("НИЗКИЙ СЫПУЧИЙ МЕТАЛЛ", "0 из 7 кадров одного предмета"),
    ("БУМАГА → ДЕРЕВО", "18 из 19 кадров; с текстурой — 17"),
    ("КАЛИБРОВКА ОБЪЁМА", "V вырос: 0,4053 → 0,5800"),
    ("НОВЫЙ КЛАССИФИКАТОР TAKE", "S снизился: 0,7389 → 0,7056"),
)):
    x, y = 50+(i%2)*535, 163+(i//2)*128
    box(x, y, 517, 110)
    text(x+18, y+15, 480, cap, 18, PLUM, True)
    text(x+18, y+56, 480, line, 26, INK)
text(50, 434, 1040, "Ранние локальные замеры, не скрытый тест.", 18, INK, True)
box(50, 483, 1052, 143, PLUM)
text(68, 501, 1015, "СЛЕДУЮЩАЯ ПРОВЕРКА", 18, GOLD, True)
text(68, 537, 1015, "Тяжёлые предметы и пропуски глубины.", 26, LIGHT, True)
text(68, 581, 1015, "Спасибо! Готовы к вопросам.", 26, LIGHT)


def html(slides):
    sections = []
    for n, slide in enumerate(slides, 1):
        parts = []
        for kind, x, y, w, size, c, *extra in slide:
            coords = f"left:{x}px;top:{y}px;width:{w}px"
            if kind == "image":
                data = b64encode((BACKGROUNDS / c).read_bytes()).decode("ascii")
                parts.append(f'<img alt="" style="{coords};height:{size}px" src="data:image/jpeg;base64,{data}">')
            elif kind == "box":
                parts.append(f'<div aria-hidden="true" style="{coords};height:{size}px;background:#{c}"></div>')
            else:
                value, bold = extra
                tag = "h1" if y in (73, 226) else "p"
                parts.append(f'<{tag} style="{coords};font-size:{size}px;font-weight:{700 if bold else 400};color:#{c}">{escape(value)}</{tag}>')
        sections.append(f'<section aria-label="Слайд {n}">{"".join(parts)}</section>')
    return """<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astramis — финальная защита</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#211828;color:white;font-family:Arial,sans-serif}
main{width:1280px;transform-origin:top left}
section{position:relative;width:1280px;height:720px;overflow:hidden;break-inside:avoid}
section>*{position:absolute;margin:0;white-space:nowrap;line-height:1.16}
nav{position:fixed;bottom:3px;right:10px;font:13px Arial;color:#aabdd0;z-index:1}
button{background:#17283a;color:white;border:1px solid #52677d;cursor:pointer}
@media screen{section[hidden]{display:none}}
@page{size:1280px 720px;margin:0}
@media print{*{print-color-adjust:exact;-webkit-print-color-adjust:exact}main{transform:none!important}
section[hidden]{display:block}section:not(:last-child){break-after:page}nav{display:none}}
</style><nav aria-label="Управление"><button id="prev" aria-label="Назад">←</button> <button id="next" aria-label="Вперёд">→</button> <button id="pdf">Печать</button></nav><main>""" + "".join(sections) + """</main>
<script>
const slides=[...document.querySelectorAll('section')], m=document.querySelector('main');let i=0;
function fit(){let s=Math.min(innerWidth/1280,innerHeight/720);m.style.transform=`translate(${(innerWidth-1280*s)/2}px,${(innerHeight-720*s)/2}px) scale(${s})`;document.body.style.overflow='hidden'}
function go(n){i=Math.min(slides.length-1,Math.max(0,n));slides.forEach((s,k)=>s.hidden=k!==i);prev.disabled=i===0;next.disabled=i===slides.length-1}
const prev=document.querySelector('#prev'),next=document.querySelector('#next');prev.onclick=()=>go(i-1);next.onclick=()=>go(i+1);document.querySelector('#pdf').onclick=()=>print();
addEventListener('keydown',e=>{if(e.target.tagName==='BUTTON'&&e.key===' ')return;const to={ArrowLeft:i-1,PageUp:i-1,ArrowRight:i+1,PageDown:i+1,' ':i+1,Home:0,End:slides.length-1};if(e.key in to){e.preventDefault();go(to[e.key])}if(e.key.toLowerCase()==='f'&&!document.fullscreenElement)document.documentElement.requestFullscreen?.()});
addEventListener('resize',fit);go(0);fit();
</script></html>
"""


def rgb(c):
    return RGBColor.from_string(c)


def px(n):
    return Emu(round(n * 9525))


def pptx(slides, path):
    deck = Presentation()
    deck.slide_width, deck.slide_height = px(1280), px(720)
    deck.core_properties.title = "Astramis — финальная защита"
    deck.core_properties.subject = "submit-82-1097-gen-material · f02779a"
    for elements in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        for kind, x, y, w, size, c, *extra in elements:
            if kind == "image":
                slide.shapes.add_picture(str(BACKGROUNDS / c), px(x), px(y), px(w), px(size))
            elif kind == "box":
                shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px(x), px(y), px(w), px(size))
                shape.fill.solid()
                shape.fill.fore_color.rgb = rgb(c)
                shape.line.fill.background()
            else:
                value, bold = extra
                shape = slide.shapes.add_textbox(px(x), px(y), px(w), px(size * 1.22))
                tf = shape.text_frame
                tf.word_wrap = False
                tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
                tf.vertical_anchor = MSO_ANCHOR.TOP
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT
                p.space_after = p.space_before = Pt(0)
                run = p.add_run()
                run.text = value
                run.font.name = "Arial"
                run.font.size = Pt(size * .75)
                run.font.color.rgb = rgb(c)
                run.font.bold = bold
    deck.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", help="Optional Chrome / Chromium executable for PDF export")
    args = parser.parse_args()
    exports = DEMO / "exports"
    exports.mkdir(exist_ok=True)
    source = DEMO / "defense.html"
    source.write_text(html(slides), encoding="utf-8")
    pptx(slides, exports / "astramis-defense.pptx")
    if args.chrome:
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run(
                [args.chrome, "--headless", "--disable-background-networking",
                 f"--user-data-dir={profile}", "--no-pdf-header-footer",
                 f"--print-to-pdf={exports / 'astramis-defense.pdf'}", source.as_uri()],
                check=True,
            )
