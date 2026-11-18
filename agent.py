import os
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

MSK = timezone(timedelta(hours=3))

MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
]

MONTHS_PATTERN = "|".join(MONTHS_RU.values())
DATE_RU_RE = re.compile(rf"\b(\d{{1,2}})\s+({MONTHS_PATTERN})\b", re.IGNORECASE)
DATE_SHORT_RE = re.compile(r"\b(\d{2})\.(\d{2})(?:\.(\d{4}))?\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")


def get_target_date() -> datetime:
    custom = os.getenv("TEST_DATE", "").strip()
    if custom:
        return datetime.strptime(custom, "%d.%m.%Y").replace(tzinfo=MSK)
    return datetime.now(MSK)


def detect_mode(msk_now: datetime) -> str:
    forced = os.getenv("RUN_MODE", "auto").strip().lower()
    if forced in ("morning", "evening"):
        return forced
    return "morning" if msk_now.hour < 16 else "evening"


def fetch_soup(url: str, timeout: int = 20) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else "без заголовка"
        logging.info("Загружено %s (%d байт) | %s", url, len(resp.text), title[:65])
        return soup
    except Exception as exc:
        logging.warning("Не удалось загрузить %s: %s", url, exc)
        return None


def normalize_date_token(text: str, default_year: int) -> str | None:
    """Извлекает дату из строки и приводит к единому виду ДД.ММ.ГГГГ."""
    m_ru = DATE_RU_RE.search(text)
    if m_ru:
        day = int(m_ru.group(1))
        month_name = m_ru.group(2).lower()
        month = next((k for k, v in MONTHS_RU.items() if v == month_name), None)
        if month and 1 <= day <= 31:
            return f"{day:02d}.{month:02d}.{default_year}"

    m_dot = DATE_SHORT_RE.search(text)
    if m_dot:
        day, month = int(m_dot.group(1)), int(m_dot.group(2))
        year = int(m_dot.group(3)) if m_dot.group(3) else default_year
        if 1 <= day <= 31 and 1 <= month <= 12:
            return f"{day:02d}.{month:02d}.{year}"

    return None


def parse_standings() -> str:
    """Парсит турнирную таблицу КХЛ."""
    sources = [
        "https://text.khl.ru/standings/",
        "https://www.sports.ru/hockey/tournament/khl/table/",
        "https://www.khl.ru/standings/",
    ]
    for url in sources:
        soup = fetch_soup(url)
        if not soup:
            continue

        tables_out = []
        for idx, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if len(cells) >= 5:
                    rows.append(" | ".join(cells[:12]))
            if len(rows) >= 6:
                title = "Таблица / Конференция 1" if idx == 0 else f"Таблица / Конференция {idx + 1}"
                tables_out.append(f"=== {title} ===\n" + "\n".join(rows))

        if tables_out:
            logging.info("Турнирная таблица получена из %s", url)
            return "\n\n".join(tables_out[:2])

    return "Турнирная таблица временно недоступна."


def extract_game_protocol_stats(game_url: str) -> str:
    """Догружает командную статистику из протокола матча для вечернего анализа."""
    if not game_url:
        return ""
    soup = fetch_soup(game_url, timeout=12)
    if not soup:
        return ""

    text = soup.get_text("\n", strip=True)
    keywords = (
        "Броски в створ", "Броски", "Штраф", "Вбрасывания",
        "Силовые приемы", "Блокированные броски", "Большинство", "Голы в большинстве",
    )
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines() if l.strip()]
    extracted = []
    for i, line in enumerate(lines):
        if any(k.lower() == line.lower() or (len(line) < 35 and k.lower() in line.lower()) for k in keywords):
            prev_val = lines[i - 1] if i > 0 and len(lines[i - 1]) < 20 else ""
            next_val = lines[i + 1] if i + 1 < len(lines) and len(lines[i + 1]) < 20 else ""
            item = f"{prev_val} — {line} — {next_val}".strip(" —")
            if item not in extracted:
                extracted.append(item)
    return "; ".join(extracted[:8])


def parse_khl_text_calendar(target_dot: str, year: int) -> list[dict]:
    """Построчный разбор текстовой версии календаря КХЛ (text.khl.ru)."""
    url = "https://text.khl.ru/calendar/"
    soup = fetch_soup(url)
    if not soup:
        return []

    # Собираем все ссылки на протоколы матчей со страницы
    game_links = {}
    for a in soup.find_all("a", href=re.compile(r"/game/\d+/\d+")):
        href = urljoin(url, a["href"])
        label = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
        if label:
            game_links[label] = href

    raw_lines = [re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").splitlines()]
    lines = [l for l in raw_lines if l]

    spotted_dates = []
    in_target_day = False
    day_lines = []

    for line in lines:
        # Заголовок даты обычно короткий и содержит день + месяц (без счёта и длинного текста)
        if len(line) <= 45 and not TIME_RE.search(line):
            dt_str = normalize_date_token(line, year)
            if dt_str:
                if dt_str not in spotted_dates:
                    spotted_dates.append(dt_str)
                if dt_str == target_dot:
                    in_target_day = True
                    continue
                elif in_target_day:
                    # Начался следующий игровой день — останавливаем сбор
                    break

        if in_target_day:
            if any(stop in line.lower() for stop in ("© кхл", "все права защищены", "партнеры лиги")):
                break
            day_lines.append(line)

    logging.info("Даты на text.khl.ru: %s", ", ".join(spotted_dates[:12]) or "не найдены")

    if not day_lines:
        return []

    # Группируем собранные строки дня по матчам (ориентир — время начала HH:MM)
    games = []
    current_chunk = []
    for item in day_lines:
        if TIME_RE.fullmatch(item) and current_chunk:
            summary = " | ".join(current_chunk)
            matched_url = next((u for lbl, u in game_links.items() if lbl in summary), "")
            games.append({"summary": summary[:350], "url": matched_url, "stats": ""})
            current_chunk = [item]
        else:
            current_chunk.append(item)

    if current_chunk:
        summary = " | ".join(current_chunk)
        if len(summary) >= 8:
            matched_url = next((u for lbl, u in game_links.items() if lbl in summary), "")
            games.append({"summary": summary[:350], "url": matched_url, "stats": ""})

    return games


def parse_sports_ru_calendar(target_dot: str, year: int) -> list[dict]:
    """Разбор календаря КХЛ на Sports.ru с учетом rowspan и заголовков дат."""
    url = "https://www.sports.ru/hockey/tournament/khl/calendar/"
    soup = fetch_soup(url)
    if not soup:
        return []

    active_date = None
    spotted_dates = []
    games = []
    seen = set()

    for elem in soup.find_all(["h2", "h3", "h4", "tr"]):
        if elem.name in ("h2", "h3", "h4"):
            dt_str = normalize_date_token(elem.get_text(" ", strip=True), year)
            if dt_str:
                active_date = dt_str
                if dt_str not in spotted_dates:
                    spotted_dates.append(dt_str)
            continue

        # Строка таблицы <tr>
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in elem.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if not cells:
            continue

        # Проверяем, указана ли дата в первой ячейке строки (при rowspan она есть только у 1-го матча дня)
        first_cell_date = normalize_date_token(cells[0], year)
        if first_cell_date:
            active_date = first_cell_date
            if first_cell_date not in spotted_dates:
                spotted_dates.append(first_cell_date)

        if active_date != target_dot:
            continue

        row_text = " | ".join(cells)
        if len(row_text) < 10 or row_text in seen:
            continue
        # Отсекаем шапки таблиц без времени и команд
        if not TIME_RE.search(row_text) and ":" not in row_text and "-" not in row_text:
            continue

        seen.add(row_text)
        match_url = ""
        for a in elem.find_all("a", href=True):
            if "/match/" in a["href"] or "/stat/" in a["href"]:
                match_url = urljoin(url, a["href"])
                break

        games.append({"summary": row_text[:350], "url": match_url, "stats": ""})

    logging.info("Даты на Sports.ru: %s", ", ".join(spotted_dates[:12]) or "не найдены")
    return games


def parse_today_games(target_dt: datetime, mode: str) -> list[dict]:
    target_dot = target_dt.strftime("%d.%m.%Y")
    logging.info("Ищем матчи на дату: %s", target_dot)

    games = parse_khl_text_calendar(target_dot, target_dt.year)
    if not games:
        games = parse_sports_ru_calendar(target_dot, target_dt.year)

    for idx, g in enumerate(games, 1):
        logging.info("Матч %d: %s", idx, g["summary"])

    if mode == "evening" and games:
        for game in games:
            if game["url"]:
                game["stats"] = extract_game_protocol_stats(game["url"])
                time.sleep(0.5)

    return games


def build_prompt(target_dt: datetime, mode: str, standings: str, games: list[dict]) -> tuple[str, str]:
    date_label = f"{target_dt.day} {MONTHS_RU[target_dt.month]} {target_dt.year}"
    games_text = "\n".join(
        f"{idx}. {g['summary']}"
        + (f"\n   Статистика протокола: {g['stats']}" if g["stats"] else "")
        for idx, g in enumerate(games, 1)
    )

    system_prompt = (
        "Ты — профессиональный хоккейный аналитик и автор Telegram-канала о КХЛ. "
        "Пиши живым, экспертным языком без воды, канцелярита и выдуманных фактов. "
        "Опирайся строго на переданные данные о матчах и турнирной таблице.\n"
        "ПРАВИЛА ФОРМАТИРОВАНИЯ (СТРОГО):\n"
        "- Используй ТОЛЬКО HTML-теги Telegram: <b>жирный</b>, <i>курсив</i>, <blockquote>цитата</blockquote>.\n"
        "- ЗАПРЕЩЕНО использовать Markdown (**звездочки**, ### заголовки, списки через *).\n"
        "- Для списков используй аккуратные эмодзи или тире (—).\n"
        "- Не используй символы < и > в тексте, кроме разрешенных тегов."
    )

    if mode == "morning":
        user_prompt = (
            f"Дата: {date_label}. Сделай утренний анонс и аналитический разбор всех матчей игрового дня КХЛ.\n\n"
            f"ТУРНИРНАЯ ТАБЛИЦА КХЛ:\n{standings}\n\n"
            f"МАТЧИ НА СЕГОДНЯ:\n{games_text}\n\n"
            "Структура поста:\n"
            f"1. Яркий заголовок: 🏒 <b>Игровой день КХЛ — {date_label}</b> и краткое вступление (1–2 предложения).\n"
            "2. По каждому матчу отдельный блок:\n"
            "   — Время начала (МСК), афиша и актуальные места команд в таблице + очки.\n"
            "   — <b>Плюсы и минусы соперников:</b> за счет чего каждая команда может забрать игру и где ее уязвимое место.\n"
            "   — <b>Ключ к матчу:</b> короткий вывод, на что смотреть в этой встрече.\n"
            "3. В конце — <blockquote>Главный матч дня и короткий вопрос подписчикам для комментариев.</blockquote>"
        )
    else:
        user_prompt = (
            f"Дата: {date_label}. Сделай вечерний итоговый обзор всех завершенных матчей дня в КХЛ.\n\n"
            f"ТУРНИРНАЯ ТАБЛИЦА КХЛ:\n{standings}\n\n"
            f"РЕЗУЛЬТАТЫ И ПРОТОКОЛЫ ДНЯ:\n{games_text}\n\n"
            "Структура поста:\n"
            f"1. Заголовок: 🚨 <b>Итоги игрового дня КХЛ — {date_label}</b>.\n"
            "2. По каждому матчу отдельный блок:\n"
            "   — Итоговый счет (с отметкой ОТ или Б, если были) и как складывалась игра по периодам.\n"
            "   — <b>Разбор игры и что пошло не так:</b> подробно разбери, почему проигравшая команда уступила "
            "(провал конкретного периода, слабая реализация бросков, удаления/меньшинство, игра вратаря или потеря концентрации).\n"
            "3. В конце — <blockquote>Главный итог вечера (сенсация дня или важная перестановка в таблице).</blockquote>"
        )

    return system_prompt, user_prompt


def generate_analysis(system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    last_error = None
    for model_id in FREE_MODELS:
        try:
            logging.info("Запрос к модели OpenRouter: %s", model_id)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                max_tokens=3200,
                extra_headers={
                    "HTTP-Referer": "https://github.com",
                    "X-Title": "KHL Telegram Analyst",
                },
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if len(content) > 100:
                logging.info("Успешный ответ от %s (%d символов)", model_id, len(content))
                return content
        except Exception as exc:
            last_error = exc
            logging.warning("Модель %s недоступна (%s), пробуем следующую...", model_id, exc)
            time.sleep(2)

    raise RuntimeError(f"Все бесплатные модели OpenRouter вернули ошибку. Последняя: {last_error}")


def sanitize_telegram_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<i>\1</i>", text)
    text = re.sub(r"^#{1,3}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    allowed = ("b", "strong", "i", "em", "u", "s", "code", "pre", "blockquote")
    tag_pattern = re.compile(r"</?([a-zA-Z0-9]+)(?:\s[^>]*)?>")

    def replace_tag(match: re.Match) -> str:
        tag_name = match.group(1).lower()
        if tag_name in allowed:
            is_closing = match.group(0).startswith("</")
            return f"</{tag_name}>" if is_closing else f"<{tag_name}>"
        return ""

    text = tag_pattern.sub(replace_tag, text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_chunks(text: str, max_len: int = 3900) -> list[str]:
    if len(text) <= max_len:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(para) <= max_len:
                current = para
            else:
                for i in range(0, len(para), max_len):
                    chunks.append(para[i:i + max_len])
                current = ""

    if current:
        chunks.append(current)
    return chunks


def send_to_telegram(text: str) -> None:
    token = os.environ["TELEGRAM_TOKEN"]
    channel_id = os.environ["CHANNEL_ID"]
    api_url = f"https://api.telegram.org/bot{token}/sendMessage"

    clean_text = sanitize_telegram_html(text)
    chunks = split_chunks(clean_text)

    for idx, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": channel_id,
            "text": chunk
