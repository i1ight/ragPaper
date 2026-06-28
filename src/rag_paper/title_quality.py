from __future__ import annotations

import re
from pathlib import Path
from typing import Any

URL_RE = re.compile(r"https?://|www\.|[a-z0-9-]+\.(com|cn|net|org|top|xyz)\b", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]+")

AD_KEYWORDS = (
    "google自动化",
    "外推软件",
    "最新版",
    "top233",
    "seo",
    "代发",
    "引流",
    "推广",
    "广告",
    "下载",
    "破解",
    "免费版",
    "官网",
    "加微信",
    "telegram",
    "whatsapp",
)


def is_trusted_title(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    title = " ".join(value.strip().split())
    if len(title) < 3 or len(title) > 260:
        return False

    lowered = title.lower()
    if URL_RE.search(title):
        return False
    if any(keyword in lowered for keyword in AD_KEYWORDS):
        return False
    if _symbol_ratio(title) > 0.28:
        return False

    words = WORD_RE.findall(title)
    has_cjk = bool(CJK_RE.search(title))
    if len(words) < 1 and not has_cjk:
        return False

    return True


def clean_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def title_from_file_name(file_name: str) -> str:
    return clean_title(Path(file_name).stem.replace("_", " ").replace("-", " "))


def best_title(*candidates: Any, file_name: str = "") -> str:
    for candidate in candidates:
        title = clean_title(candidate)
        if is_trusted_title(title):
            return title
    return title_from_file_name(file_name) if file_name else ""


def _symbol_ratio(value: str) -> float:
    visible = [char for char in value if not char.isspace()]
    if not visible:
        return 1.0
    symbols = [
        char
        for char in visible
        if not char.isalnum() and not CJK_RE.match(char) and char not in ":-,.;()[]/"
    ]
    return len(symbols) / len(visible)
