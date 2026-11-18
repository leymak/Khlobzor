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
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

# Актуальный список бесплатных моделей OpenRouter с автоматическим фолбэком
FREE_MODELS = [
    "openrouter/free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "qwen/qwen3.8-27b:free",
    "google/gemma-4-31b-it:free",
]


def get_msk_now() -> datetime:
    return datetime.now(MSK)


def detect_mode(msk_now: datetime) -> str:
    forced = os.getenv("RUN_MODE", "auto").strip().lower()
    if forced in ("morning", "evening"):
        return forced
    return "morning" if msk_now.hour < 16 else "evening"


def fetch_html(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_standings() -> str:
    """Парсит турнирную таблицу КХЛ (Запад и Восток)."""
    url = "https://www.khl.ru/standings/"
    try:
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        tables_out = []

        for idx, table in enumerate(soup.find_all("table")):
            rows = []
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if len(cells) >= 6:
                    rows.append(" | ".join(cells[:12]))
            if len(rows) >= 5:
                title = "Конференция Запад" if idx == 0 else (
                    "Конференция Восток" if idx == 1 else f"Таблица {idx + 1}"
                )
                tables_out.append(f"=== {title} ===\n" + "\n".join(rows))

        if tables_out:
            return "\n\n".join(tables_out[:2])

        # Резервное извлечение текста страницы, если структура таблиц изменилась
        body_text = soup.get_text("\n", strip=True)
        lines = [line for line in body_text.splitlines() if line]
        return "\n".join(lines[:140])
    except Exception as exc:
        logging.warning("Ошибка при загрузке таблицы КХЛ: %s", exc)
        return "Турнирная таблица временно недоступна."


def extract_game_protocol_stats(game_url: str) -> str:
    """Догружает командную статистику из протокола матча для вечернего анализа."""
    try:
        html = fetch_html(game_url, timeout=12)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        keywords = (
            "Броски в створ", "Штраф", "Вбрасывания",
            "Силовые приемы", "Блокированные броски", "Голы в большинстве",
        )
        lines = text.splitlines()
        extracted = []
        for i, line in enumerate(lines):
            if any(k.lower() in line.lower() for k in keywords):
                prev_val = lines[i - 1] if i > 0 else ""
                next_val = lines[i + 1] if i + 1 < len(lines) else ""
                extracted.append(f"{prev_val} — {line} — {next_val}".strip(" —"))
        return "; ".join(extracted[:8])
    except Exception:
        return ""


def parse_today_games(msk_now: datetime, mode: str) -> list[dict]:
    """Парсит матчи КХЛ на текущую дату по МСК."""
    url = "https://www.khl.ru/calendar/"
    date_ru = f"{msk_now.day} {MONTHS_RU[msk_now.month]}"
    date_dot = msk_now.strftime("%d.%m.%Y")
    logging.info("Ищем матчи на дату: %s (%s)", date_ru, date_dot)

    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    games = []
    seen_urls = set()

    # Ищем карточки матчей / ссылки на игры /game/
    for link in soup.find_all("a", href=re.compile(r"/game/\d+/\d+")):
        href = urljoin("https://www.khl.ru", link.get("href", ""))
        if href in seen_urls:
            continue

        # Поднимаемся к родительскому контейнеру матча и блока даты
        card = link
        for _ in range(5):
            if card.parent:
                card = card.parent

        block_text = card.get_text(" ", strip=True)
        parent_section = card.parent.get_text(" ", strip=True) if card.parent else block_text

        # Проверяем принадлежность сегодняшнему игровому дню
        if date_ru not in parent_section and date_dot not in parent_section:
            continue

        seen_urls.add(href)
        clean_info = re.sub(r"\s+", " ", block_text).strip()
        if len(clean_info) < 8:
            continue

        game_entry = {
            "summary": clean_info[:350],
            "url": href,
            "stats": "",
        }
        if mode == "evening":
            game_entry["stats"] = extract_game_protocol_stats(href)
            time.sleep(0.5)

        games.append(game_entry)

    return games


def build_prompt(msk_now: datetime, mode: str, standings: str, games: list[dict]) -> tuple[str, str]:
    date_label = f"{msk_now.day} {MONTHS_RU[msk_now.month]} {msk_now.year}"
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
            # Убираем блоки размышлений <think>...</think>, если модель их вернула
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
    """Приводит ответ модели к валидному Telegram HTML."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<i>\1</i>", text)
    text = re.sub(r"^#{1,3}\s*(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # Оставляем только разрешенные Telegram теги
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
    """Разбивает длинный пост по абзацам, не разрывая HTML-теги внутри блоков."""
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
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        resp = requests.post(api_url, json=payload, timeout=20)
        if not resp.ok:
            logging.warning("Ошибка HTML в части %d (%s), отправка без разметки...", idx, resp.text)
            plain_chunk = re.sub(r"<[^>]+>", "", chunk)
            fallback_payload = {
                "chat_id": channel_id,
                "text": plain_chunk,
                "disable_web_page_preview": True,
            }
            fb_resp = requests.post(api_url, json=fallback_payload, timeout=20)
            fb_resp.raise_for_status()
        logging.info("Часть %d/%d успешно опубликована в %s", idx, len(chunks), channel_id)
        time.sleep(1)


def main() -> None:
    msk_now = get_msk_now()
    mode = detect_mode(msk_now)
    logging.info("Запуск КХЛ-агента | Время МСК: %s | Режим: %s", msk_now.strftime("%Y-%m-%d %H:%M"), mode)

    games = parse_today_games(msk_now, mode)
    if not games:
        logging.info("На сегодня (%s) матчей КХЛ не найдено. Публикация пропущена.", msk_now.strftime("%d.%m.%Y"))
        return

    logging.info("Найдено матчей на сегодня: %d", len(games))
    standings = parse_standings()
    system_prompt, user_prompt = build_prompt(msk_now, mode, standings, games)
    post_text = generate_analysis(system_prompt, user_prompt)
    send_to_telegram(post_text)


if __name__ == "__main__":
    main()
