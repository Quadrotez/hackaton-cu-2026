#!/usr/bin/env python3
"""
Скрипт для получения списка всех отправок и баллов с платформы StackMoreLayers.
Использует cookies из браузера Zen (или переданные вручную).

Использование:
  python3 get_submissions.py
  python3 get_submissions.py --sort-score
  python3 get_submissions.py --csv submissions.csv
  python3 get_submissions.py --json submissions.json
  python3 get_submissions.py --competition astramis
"""

import os
import sys
import glob
import shutil
import sqlite3
import argparse
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import requests
except ImportError:
    print("Ошибка: библиотека 'requests' не установлена. Установите её: pip install requests", file=sys.stderr)
    sys.exit(1)


def find_zen_cookies_db() -> Optional[str]:
    """Ищет cookies.sqlite в профилях браузера Zen."""
    search_patterns = [
        os.path.expanduser("~/.config/zen/*/cookies.sqlite"),
        os.path.expanduser("~/.zen/*/cookies.sqlite"),
        os.path.expanduser("~/.var/app/app.zen_browser.zen/.zen/*/cookies.sqlite"),
        os.path.expanduser("~/.var/app/app.zen_browser.zen/.config/zen/*/cookies.sqlite"),
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Возвращаем самый свежий файл базы данных
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]
    return None


def extract_cookies_from_sqlite(db_path: str, domain: str = "stackmorelayers.ai") -> Dict[str, str]:
    """Безопасно копирует и читает cookies из SQLite БД Zen браузера."""
    tmp_db = f"/tmp/zen_cookies_{os.getpid()}_{int(time.time())}.sqlite"
    try:
        shutil.copy2(db_path, tmp_db)
        conn = sqlite3.connect(tmp_db)
        cursor = conn.cursor()
        
        query = "SELECT name, value FROM moz_cookies WHERE host LIKE ?"
        cursor.execute(query, (f"%{domain}%",))
        cookies = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return cookies
    except Exception as e:
        print(f"Предупреждение: ошибка чтения cookies из {db_path}: {e}", file=sys.stderr)
        return {}
    finally:
        if os.path.exists(tmp_db):
            try:
                os.remove(tmp_db)
            except OSError:
                pass


def parse_submissions_from_html(html_text: str) -> List[Dict[str, Any]]:
    """Парсит сабмиты из Next.js hydration payload (self.__next_f.push) в HTML."""
    import re
    pushes = re.findall(r'self\.__next_f\.push\(\[1,\s*\"(.*?)\"\]\)', html_text, re.DOTALL)
    for p in pushes:
        if "submissions" in p and "safe_metrics" in p:
            try:
                decoded = json.loads('"' + p + '"')
                idx = decoded.find('"submissions":[')
                if idx != -1:
                    sub_str = decoded[idx + len('"submissions":'):]
                    cnt = 0
                    end_idx = 0
                    in_str = False
                    esc = False
                    for j, ch in enumerate(sub_str):
                        if esc:
                            esc = False
                            continue
                        if ch == '\\':
                            esc = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if not in_str:
                            if ch == '[':
                                cnt += 1
                            elif ch == ']':
                                cnt -= 1
                                if cnt == 0:
                                    end_idx = j + 1
                                    break
                    if end_idx > 0:
                        return json.loads(sub_str[:end_idx])
            except Exception:
                continue
    return []


def fetch_submissions(competition: str, cookies: Dict[str, str], max_retries: int = 3) -> List[Dict[str, Any]]:
    """Запрашивает отправки через REST API с запасным вариантом через HTML."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    
    session = requests.Session()
    session.headers.update(headers)
    session.cookies.update(cookies)
    
    api_url = f"https://stackmorelayers.ai/api/platform/v1/competitions/{competition}/submissions"
    page_url = f"https://stackmorelayers.ai/competitions/{competition}/submissions"
    
    # 1. Попытка через официальный API эндпоинт
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(api_url, timeout=(10, 30))
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "results" in data:
                    return data["results"]
                elif isinstance(data, list):
                    return data
            elif resp.status_code in (401, 403):
                print("Ошибка авторизации (401/403). Проверьте актуальность cookies в Zen браузере.", file=sys.stderr)
                return []
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                print(f"API запрос не удался после {max_retries} попыток ({e}). Пробуем загрузку через HTML страницу...", file=sys.stderr)
            time.sleep(1)

    # 2. Фолбэк через HTML страницу Next.js
    try:
        resp = session.get(page_url, timeout=(10, 30))
        if resp.status_code == 200:
            subs = parse_submissions_from_html(resp.text)
            if subs:
                return subs
    except Exception as e:
        print(f"Ошибка загрузки HTML страницы: {e}", file=sys.stderr)

    return []


def get_primary_score(submission: Dict[str, Any]) -> Optional[float]:
    """Извлекает основной балл из структуры сабмита."""
    # 1. safe_metrics
    for metric in submission.get("safe_metrics", []):
        if metric.get("is_primary") or "score" in metric:
            try:
                return float(metric["score"])
            except (ValueError, TypeError):
                pass

    # 2. scores
    for sc in submission.get("scores", []):
        if sc.get("is_primary") or "score" in sc:
            try:
                return float(sc["score"])
            except (ValueError, TypeError):
                pass

    return None


def format_datetime(dt_str: str) -> str:
    """Форматирует ISO дату в удобный локальный формат."""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%d.%m.%Y %H:%M:%S")
    except Exception:
        return dt_str[:19].replace("T", " ")


def print_submissions_table(submissions: List[Dict[str, Any]]) -> None:
    """Красиво выводит список отправок в консоль."""
    if not submissions:
        print("Отправки не найдены.")
        return

    # Находим максимальный балл для подсветки
    scored_values = []
    for s in submissions:
        sc = get_primary_score(s)
        if sc is not None:
            scored_values.append(sc)
    
    best_score = max(scored_values) if scored_values else None

    max_fn_len = max(len(str(s.get("filename", ""))) for s in submissions)
    fn_col_w = max(35, min(max_fn_len, 60))

    # Заголовок таблицы
    header_fmt = f"{{:<7}} | {{:<12}} | {{:<8}} | {{:<20}} | {{:<5}} | {{:<{fn_col_w}}}"
    row_fmt    = f"{{:<7}} | {{:<12}} | {{:<8}} | {{:<20}} | {{:<5}} | {{:<{fn_col_w}}}"
    divider = "-" * (58 + fn_col_w)

    print(f"\nНайдено отправок: {len(submissions)}")
    if best_score is not None:
        print(f"Лучший результат: \033[1;32m{best_score:.6f}\033[0m")
    print(divider)
    print(header_fmt.format("ID", "Балл (Score)", "Статус", "Дата и время", "Финал", "Имя файла"))
    print(divider)

    for item in submissions:
        sid = str(item.get("id", ""))
        status = str(item.get("status", ""))
        fn = str(item.get("filename", ""))
        is_final = "✓" if item.get("is_selected_for_final") else ""
        dt_str = format_datetime(item.get("created_when", ""))
        
        score_val = get_primary_score(item)
        if score_val is not None:
            score_str = f"{score_val:.6f}"
            if best_score is not None and abs(score_val - best_score) < 1e-9:
                score_str = f"\033[1;32m{score_str} ★\033[0m"
        else:
            score_str = "—"

        print(row_fmt.format(sid, score_str, status, dt_str, is_final, fn[:fn_col_w]))

    print(divider + "\n")


def main():
    parser = argparse.ArgumentParser(description="Получение всех отправок и баллов с платформы StackMoreLayers")
    parser.add_argument("-c", "--competition", default="astramis", help="Кодовое имя соревнования (по умолчанию: astramis)")
    parser.add_argument("--sort-score", action="store_true", help="Сортировать по убыванию балла")
    parser.add_argument("--sort-date", action="store_true", help="Сортировать по дате (по умолчанию)")
    parser.add_argument("--csv", metavar="FILE", help="Сохранить результаты в CSV файл")
    parser.add_argument("--json", metavar="FILE", help="Сохранить результаты в JSON файл")
    parser.add_argument("--cookie-db", help="Путь к файлу cookies.sqlite (если не найден автоматически)")
    args = parser.parse_args()

    # Поиск базы cookies
    db_path = args.cookie_db or find_zen_cookies_db()
    if not db_path or not os.path.exists(db_path):
        print("Ошибка: не удалось найти cookies.sqlite для браузера Zen.", file=sys.stderr)
        print("Укажите путь вручную с помощью флага --cookie-db /path/to/cookies.sqlite", file=sys.stderr)
        sys.exit(1)

    cookies = extract_cookies_from_sqlite(db_path)
    if "sml_platform_session" not in cookies:
        print("Предупреждение: сессионная cookie 'sml_platform_session' не найдена в базе Zen.", file=sys.stderr)
        print("Убедитесь, что вы авторизованы на сайте https://stackmorelayers.ai в браузере Zen.", file=sys.stderr)

    submissions = fetch_submissions(args.competition, cookies)
    if not submissions:
        print("Не удалось получить отправки. Проверьте подключение и статус авторизации.")
        sys.exit(1)

    # Сортировка
    if args.sort_score:
        submissions.sort(key=lambda s: (get_primary_score(s) is not None, get_primary_score(s) or 0), reverse=True)
    elif args.sort_date or True:
        # По умолчанию по дате от новых к старым
        submissions.sort(key=lambda s: s.get("created_when", ""), reverse=True)

    # Вывод в консоль
    print_submissions_table(submissions)

    # Сохранение в CSV
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score", "status", "created_when", "is_final", "filename", "owner"])
            for s in submissions:
                sc = get_primary_score(s)
                writer.writerow([
                    s.get("id"),
                    sc if sc is not None else "",
                    s.get("status"),
                    s.get("created_when"),
                    s.get("is_selected_for_final", False),
                    s.get("filename"),
                    s.get("owner"),
                ])
        print(f"Результаты сохранены в CSV: {args.csv}")

    # Сохранение в JSON
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)
        print(f"Полные данные сохранены в JSON: {args.json}")


if __name__ == "__main__":
    main()
