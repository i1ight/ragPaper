from __future__ import annotations

import re
from pathlib import Path
from typing import Any

URL_RE = re.compile(r"https?://|www\.|[a-z0-9-]+\.(com|cn|net|org|top|xyz)\b", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]+")

# Matches a line that is (almost) just a section heading, e.g. "Abstract",
# "1. Introduction", "References:". These are never the paper's title even though
# they may otherwise pass is_trusted_title.
SECTION_HEADER_RE = re.compile(
    r"^\s*\d*\.?\s*("
    r"abstract|keywords?|introduction|related works?|background|"
    r"methods?|materials and methods|results?|experiments?|"
    r"discussion|conclusions?|references?|bibliography|"
    r"acknowledg(e?)ments?|contents?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

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


_FILENAME_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def parse_filename_title_and_year(file_name: str) -> tuple[str, int | None]:
    """Extract a ``(title, year)`` pair from a Zotero-style filename.

    Zotero names files ``"Authors - Year - Title.pdf"``; when that pattern is
    detected, the title (everything after the year) and the year are returned.
    This is far more reliable than inferring the title from PDF body text, which
    often lands on boilerplate such as ``"Published as a conference paper at ICLR
    ..."``. Returns ``("", None)`` when the name does not match the pattern.
    """
    stem = Path(str(file_name).replace("\\", "/")).stem
    parts = [segment.strip() for segment in stem.split(" - ")]
    if len(parts) >= 3 and _FILENAME_YEAR_RE.match(parts[1]):
        try:
            year = int(parts[1])
        except ValueError:
            return "", None
        return " - ".join(parts[2:]).strip(), year
    return "", None


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


# Title-inference tuning. Titles live near the top of the first page and have a
# moderate length; these bound the candidate window and the acceptable length.
TITLE_SCAN_LINES = 12
TITLE_LINE_MIN = 12
TITLE_LINE_MAX = 200


def pick_title_line(lines, *, file_name: str = "") -> str:
    """Pick the most title-like line from the top of a document.

    Titles sit at the top of the first page, so this returns the first early line
    that *looks like* a title — trusted, within a sane length band, and not a
    section heading — rather than the longest line (which is frequently an
    affiliation, an abstract sentence, or a funding acknowledgment below the
    title). Falls back to a filename-derived title when nothing qualifies.

    The result is only used as a provider query hint; any match is verified
    against it downstream, so a bad guess here yields a skipped enrichment, not a
    wrong association.
    """
    seen = 0
    for raw in lines:
        line = clean_title(raw)
        if not line or line.startswith("[Page "):
            continue
        seen += 1
        if seen > TITLE_SCAN_LINES:
            break
        if _looks_like_title(line):
            return line
    return title_from_file_name(file_name) if file_name else ""


def _looks_like_title(text: str) -> bool:
    if not (TITLE_LINE_MIN <= len(text) <= TITLE_LINE_MAX):
        return False
    if SECTION_HEADER_RE.match(text):
        return False
    return is_trusted_title(text)
