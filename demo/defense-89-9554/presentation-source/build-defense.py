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
     "BRIEF.md · RANK1_FINAL_REPORT.md · origin/top0-89-9554 · e31321f")
text(102, 226, 685, "Что под щёткой?", 46, INK, True)
text(104, 290, 650, "Робот решает: убрать или объехать.", 26, INK, True)
box(102, 356, 622, 103, PLUM)
text(124, 375, 582, "Из кадра RGB-D — рамка, свойства", 26, LIGHT)
text(124, 413, 582, "предмета и действие робота.", 26, LIGHT)
box(795, 208, 435, 306)
box(795, 208, 435, 7, GOLD)
text(820, 243, 388, "1-Е МЕСТО · ПУБЛИЧНЫЙ ЛИДЕРБОРД", 18, PLUM, True)
text(817, 282, 390, "89,9554", 72, INK, True)
text(820, 382, 380, "балла из 100", 26, PLUM, True)
text(820, 444, 380, "Сабмит №17448", 26, MUTED)
box(305, 551, 925, 57, PAPER)
text(323, 564, 891, "До 500 г включительно — убрать. Тяжелее — объехать.", 26, INK, True)
text(415, 623, 810, "Команда Astramis · Научная орбита 2026  |  22 ноября, около 12:50 МСК", 18, INK, True)

page(2, "РЕШЕНИЕ И ПРОВЕРКА", "От кадра к решению",
     "README.md · work/astra_cv.py · RANK1_FINAL_REPORT.md:14–18")
for i, (cap, label, note) in enumerate((
    ("01  РАМКИ", "YOLO11-seg", "Маски из RGB-кадра"),
    ("02  АТРИБУТЫ", "DINO + CLIP", "Материал и форма"),
    ("03  ОБЪЁМ", "Карта глубины", "Сверка между ракурсами"),
    ("04  ДЕЙСТВИЕ", "Убрать / объехать", "Масса и постобработка"),
)):
    x = 50+i*298
    box(x, 175, 286, 154)
    box(x, 175, 286, 5, MAUVE)
    text(x+14, 195, 257, cap, 18, MUTED, True)
    text(x+14, 230, 257, label, 26, INK, True)
    text(x+14, 277, 257, note, 18, MUTED)
box(50, 351, 1180, 131, PLUM)
text(71, 376, 1138, "Модели проверяли на 5 фолдах по предметам.", 26, LIGHT, True)
text(71, 420, 1138, "Поправки к базовому CSV подбирали отдельно, по публичному баллу.", 26, LIGHT)
text(50, 502, 1138, "ЭТАПЫ 0–8", 18, INK, True)
box(50, 540, 1180, 79)
for i, (label, status) in enumerate((
    ("0 Бейзлайн", "ориентир"), ("1 Среда", "офлайн"), ("2 Сплит", "по ID"),
    ("3 Кропы", "из рамок"), ("4 Материал", "DINO + CLIP"),
    ("5 Объём", "глубина"), ("6 Решение", "take"),
    ("7 Замеры", "OOF / тест"), ("8 Сборка", "CSV / SHA"),
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

page(4, "ПУБЛИЧНЫЙ ЛИДЕРБОРД", "Две правки подняли балл до 89,9554",
     "ЖУРНАЛ.md:73 · RANK1_FINAL_REPORT.md:4–18 · побайтовая сверка финального CSV")
box(50, 193, 1180, 339)
text(80, 219, 190, "ВЕРСИЯ", 18, MUTED, True)
text(281, 219, 739, "ЧТО ИЗМЕНИЛИ", 18, MUTED, True)
text(1024, 219, 180, "БАЛЛ ИЗ 100", 18, MUTED, True)
for i, (version, change, result) in enumerate((
    ("v46", "База: несколько ракурсов и постобработка", "89,0606"),
    ("v52", "Калибровка объёмов; цельный пластик ×1,25", "89,6817"),
    ("Финал", "Объём пластика ×1,10; в одном кадре — только bulk", "89,9554"),
)):
    y = 259+i*79
    if i == 2:
        box(72, y-7, 1136, 66, "F5E1A0")
    if i == 1:
        box(80, y-10, 1118, 1, HAIR)
    text(80, y+9, 187, version, 26, INK, True)
    text(281, y+9, 739, change, 26, INK, i == 2)
    text(1024, y+5, 182, result, 31, INK, True)
text(50, 552, 1180, "35 кадров с правками. Масса и действие оставшихся гипотез не менялись.", 26, INK, True)
text(50, 609, 1180, "Отрыв от 2-го места: 0,2737 балла. Бейзлайн ≈59 — ориентир из условия.", 18, INK)

page(5, "ЭКСПЕРИМЕНТЫ И ВЫВОД", "Не каждая поправка помогла",
     "ЖУРНАЛ.md:63,74 · RANK1_FINAL_REPORT.md:14–18 · независимого замера для финала нет")
for i, (cap, line) in enumerate((
    ("ОБЪЁМ ПО КАТАЛОГУ", "86,6963: хуже оценки по глубине"),
    ("БОЛЬШЕ СГЛАЖИВАНИЯ · v47", "88,9657: ниже 89,0606 у v46"),
    ("ВАРИАНТ «ПЛАСТИК ×1,35»", "89,5224: ниже победного балла"),
    ("ВАРИАНТ «МЕТАЛЛ ×1,10»", "89,8419: ниже победного балла"),
)):
    x, y = 50+(i%2)*535, 163+(i//2)*128
    box(x, y, 517, 110)
    text(x+18, y+15, 480, cap, 18, PLUM, True)
    text(x+18, y+56, 480, line, 26, INK)
text(50, 434, 1040, "Это публичные эксперименты, а не независимая валидация.", 18, INK, True)
box(50, 483, 1052, 143, PLUM)
text(68, 501, 1015, "ПЕРВОЕ МЕСТО · ПОБЕДНЫЙ CSV ВОСПРОИЗВЕДЁН ПОБАЙТНО", 18, GOLD, True)
text(68, 537, 1015, "Следующий шаг: проверить калибровку на новых проходах.", 26, LIGHT, True)
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
    deck.core_properties.subject = "top0-89-9554 · e31321f"
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
    source = DEMO / "index.html"
    source.write_text(html(slides), encoding="utf-8")
    pptx(slides, exports / "astramis-89-9554.pptx")
    if args.chrome:
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run(
                [args.chrome, "--headless", "--disable-background-networking",
                 f"--user-data-dir={profile}", "--no-pdf-header-footer",
                 f"--print-to-pdf={exports / 'astramis-89-9554.pdf'}", source.as_uri()],
                check=True,
            )
