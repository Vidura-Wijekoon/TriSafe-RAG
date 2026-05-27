"""Text normalization and language detection for Sinhala, Tamil, and English."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

LanguageCode = Literal["si", "ta", "en", "mixed", "unknown"]

# Sinhala block: U+0D80-U+0DFF, Tamil block: U+0B80-U+0BFF, Basic Latin: U+0000-U+007F.
_SINHALA_RE = re.compile(r"[඀-෿]")
_TAMIL_RE = re.compile(r"[஀-௿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WS_RE = re.compile(r"\s+")
_BIDI_CTRL = re.compile(r"[​-‏‪-‮﻿]")


def normalize_text(text: str) -> str:
    """Apply Unicode NFC, strip bidi control chars, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _BIDI_CTRL.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def detect_language(text: str, mixed_threshold: float = 0.2) -> LanguageCode:
    """Lightweight script-based detection.

    The heuristic counts characters per script and returns the dominant one.
    When two scripts each contribute more than ``mixed_threshold`` of the
    detected characters, the text is labeled ``"mixed"`` (code-switched input).
    """
    if not text:
        return "unknown"

    counts = {
        "si": len(_SINHALA_RE.findall(text)),
        "ta": len(_TAMIL_RE.findall(text)),
        "en": len(_LATIN_RE.findall(text)),
    }
    total = sum(counts.values())
    if total == 0:
        return "unknown"

    ratios = {k: v / total for k, v in counts.items()}
    sorted_ratios = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
    top, second = sorted_ratios[0], sorted_ratios[1]
    if second[1] >= mixed_threshold and top[1] < 0.85:
        return "mixed"
    return top[0]  # type: ignore[return-value]
