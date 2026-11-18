import os
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MSK = timezone(timedelta(hours=3))
MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}
FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
]

DATE_RU_RE = re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS_RU.values())})\b", re.I)
DATE_DOT_RE = re.compile(r"\b(\d{2})\.(\d{2})(?:\.(\d{4}))?\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")


def fetch_soup(url: str, timeout: int = 20) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        logging.info("Загружено %s (%d байт) | %s", url, len(r.text), title[:60])
        return soup
    except Exception as e:
        logging.warning("Ошибка загрузки %s: %s", url, e)
        return None


def norm_date(text: str, year: int) -> str | None:
    m = DATE_RU_RE.search(text)
    if m:
        day = int(m.group(1))
        month = next((k for k, v in MONTHS_RU.items() if v == m.group(2).lower()), None)
        if month and 1 <= day <= 31:
            return f"{day:02d}.{month:02d}.{year}"
    m = DATE_DOT_RE.search(text)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else year
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{d:02d}.{mo:02d}.{y}"
    return None


def parse_standings() -> str:
    for url in ("https://text.khl.ru/standings/", "https://www.sports.ru/hockey/tournament/khl/table/"):
        soup = fetch_soup(url)
        if not soup:
            continue
        out = []
        for idx, tbl in enumerate(soup.find_all("table"), 1):
            rows = [
                " | ".join(c for c in [re.sub(r"\s+", " ", td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])] if c)[:120]
                for tr in tbl.find_all("tr")
            ]
            rows = [r for r in rows if r.count("|") >= 4]
            if len(rows) >= 6:
                out.append(f"=== Конференция {idx} ===\n" + "\n".join(rows))
        if out:
            return "\n\n".join(out[:2])
    return "Турнирная таблица недоступна."


def parse_protocol(url: str) -> str:
    soup = fetch_soup(url, timeout=12)
    if not soup:
        return ""
    lines = [re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").splitlines() if l.strip()]
    keys = ("Броски в створ", "Штраф", "Вбрасывания", "Большинство", "Голы в большинстве")
    res = []
    for i, line in enumerate(lines):
        if len(line) < 35 and any(k.lower() in line.lower() for k in keys):
            prev_v = lines[i - 1] if i > 0 and len(lines[i - 1]) < 20 else ""
            next_v = lines[i + 1] if i + 1 < len(lines) and len(lines[i + 1]) < 20 else ""
            res.append(f"{prev_v} — {line} — {next_v}".strip(" —"))
    return "; ".join(dict.fromkeys(res))[:400]


def parse_games(target_dot: str, year: int, mode: str) -> list[dict]:
    games = []
    # 1. Текстовая версия КХЛ
    soup = fetch_soup("https://text.khl.ru/calendar/")
    if soup:
        links = {
            re.sub(r"\s+", " ", a.get_text(" ", strip=True)): urljoin("https://text.khl.ru", a["href"])
            for a in soup.find_all("a", href=re.compile(r"/game/\d+/\d+"))
        }
        lines = [re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").splitlines() if l.strip()]
        dates, active, chunk = [], False, []
        for line in lines:
            if len(line) <= 45 and not TIME_RE.search(line):
                dt = norm_date(line, year)
                if dt:
                    if dt not in dates:
                        dates.append(dt)
                    if dt == target_dot:
                        active = True
                        continue
                    elif active:
                        break
            if active:
                if "все права защищены" in line.lower():
                    break
                if TIME_RE.fullmatch(line) and chunk:
                    s = " | ".join(chunk)
                    u = next((v for k, v in links.items() if k and k in s), "")
                    games.append({"summary": s[:350], "url": u, "stats": ""})
                    chunk = [line]
                else:
                    chunk.append(line)
        if chunk and len(" | ".join(chunk)) >= 8:
            s = " | ".join(chunk)
            u = next((v for k, v in links.items() if k and k in s), "")
            games.append({"summary": s[:350], "url": u, "stats": ""})
        logging.info("Даты на text.khl.ru: %s", ", ".join(dates[:12]) or "нет")

    # 2. Резерв: Sports.ru
    if not games:
        soup = fetch_soup("https://www.sports.ru/hockey/tournament/khl/calendar/")
        if soup:
            active_dt, dates, seen = None, [], set()
            for el in soup.find_all(["h2", "h3", "h4", "tr"]):
                if el.name in ("h2", "h3", "h4"):
                    dt = norm_date(el.get_text(" ", strip=True), year)
                    if dt:
                        active_dt = dt
                        if dt not in dates:
                            dates.append(dt)
                    continue
                cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in el.find_all(["th", "td"]) if c.get_text(strip=True)]
                if not cells:
                    continue
                dt0 = norm_date(cells[0], year)
                if dt0:
                    active_dt = dt0
                    if dt0 not in dates:
                        dates.append(dt0)
                row = " | ".join(cells)
                if active_dt == target_dot and len(row) >= 10 and row not in seen and (TIME_RE.search(row) or ":" in row or "-" in row):
                    seen.add(row)
                    a = el.find("a", href=re.compile(r"/(match|stat)/"))
                    games.append({"summary": row[:350], "url": urljoin("https://www.sports.ru", a["href"]) if a else "", "stats": ""})
            logging.info("Даты на Sports.ru: %s", ", ".join(dates[:12]) or "нет")

    if mode == "evening":
        for g in games:
            if g["url"]:
                g["stats"] = parse_protocol(g["url"])
                time.sleep(0.4)
    return games


def ask_llm(system_p: str, user_p: str) -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    last_err = None
    for model in FREE_MODELS:
        try:
            logging.info("Запрос к модели: %s", model)
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_p}, {"role": "user", "content": user_p}],
                temperature=0.6,
                max_tokens=3000,
                extra_headers={"HTTP-Referer": "https://github.com", "X-Title": "KHL Agent"},
            )
            txt = re.sub(r"<think>.*?</think>", "", (r.choices[0].message.content or ""), flags=re.S).strip()
            if len(txt) > 100:
                return txt
        except Exception as e:
            last_err = e
            logging.warning("Модель %s недоступна: %s", model, e)
            time.sleep(2)
    raise RuntimeError(f"Нет доступных моделей OpenRouter: {last_err}")


def send_telegram(raw_text: str) -> None:
    token, chat_id = os.environ["TELEGRAM_TOKEN"], os.environ["CHANNEL_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", raw_text)
    txt = re.sub(r"^#{1,3}\s*(.+)$", r"<b>\1</b>", txt, flags=re.M).strip()

    chunks, cur = [], ""
    for para in txt.split("\n\n"):
        cand = f"{cur}\n\n{para}".strip() if cur else para
        if len(cand) <= 3900:
            cur = cand
        else:
            if cur:
                chunks.append(cur)
            cur = para[:3900]
    if cur:
        chunks.append(cur)

    for i, chunk in enumerate(chunks, 1):
        resp = requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}, timeout=20)
        if not resp.ok:
            plain = re.sub(r"<[^>]+>", "", chunk)
            requests.post(url, json={"chat_id": chat_id, "text": plain, "disable_web_page_preview": True}, timeout=20).raise_for_status()
        logging.info("Опубликована часть %d/%d", i, len(chunks))
        time.sleep(1)


def main() -> None:
    now = datetime.now(MSK)
    forced = os.getenv("RUN_MODE", "auto").strip().lower()
    mode = forced if forced in ("morning", "evening") else ("morning" if now.hour < 16 else "evening")
    date_dot = now.strftime("%d.%m.%Y")
    date_label = f"{now.day} {MONTHS_RU[now.month]} {now.year}"
    logging.info("Старт агента | %s МСК | Режим: %s", now.strftime("%d.%m.%Y %H:%M"), mode)

    games = parse_games(date_dot, now.year, mode)
    if not games:
        logging.info("На %s матчей КХЛ не найдено. Выход.", date_dot)
        return

    logging.info("Найдено матчей: %d", len(games))
    standings = parse_standings()
    games_str = "\n".join(f"{i}. {g['summary']}" + (f" | Стат: {g['stats']}" if g["stats"] else "") for i, g in enumerate(games, 1))

    sys_p = (
        "Ты хоккейный аналитик Telegram-канала о КХЛ. Пиши ярко и по делу без выдуманных фактов. "
        "Используй ТОЛЬКО HTML-теги Telegram: <b>, <i>, <blockquote>. Никакого Markdown (** или #)."
    )
    if mode == "morning":
        usr_p = (
            f"Сделай утренний разбор матчей КХЛ на {date_label}.\nТАБЛИЦА:\n{standings}\n\nМАТЧИ:\n{games_str}\n\n"
            f"Формат:\n🏒 <b>Игровой день КХЛ — {date_label}</b>\nПо каждому матчу: время МСК, места в таблице и очки, "
            "<b>Плюсы и минусы команд</b>, <b>Ключ к матчу</b>.\nВ конце: <blockquote>Главный матч дня и вопрос подписчикам.</blockquote>"
        )
    else:
        usr_p = (
            f"Сделай вечерний итог матчей КХЛ за {date_label}.\nТАБЛИЦА:\n{standings}\n\nРЕЗУЛЬТАТЫ:\n{games_str}\n\n"
            f"Формат:\n🚨 <b>Итоги игрового дня КХЛ — {date_label}</b>\nПо каждому матчу: счет, ход игры и блок "
            "<b>Разбор игры и что пошло не так</b> (причины поражения проигравшей команды).\nВ конце: <blockquote>Главный итог вечера.</blockquote>"
        )

    post = ask_llm(sys_p, usr_p)
    send_telegram(post)


if __name__ == "__main__":
    main()
