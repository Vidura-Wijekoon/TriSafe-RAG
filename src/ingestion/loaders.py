"""Document loaders for plain text, markdown, HTML, and PDF sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Supported source extensions handled by the loader registry below.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}


@dataclass
class Document:
    """A raw source document before normalization or chunking."""

    doc_id: str
    text: str
    source_path: str
    language_hint: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def load_text_file(path: Path) -> Document:
    """Read a single .txt or .md file from disk."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    return Document(
        doc_id=path.stem,
        text=raw,
        source_path=str(path),
        metadata={"size_bytes": path.stat().st_size},
    )


def load_html_file(path: Path) -> Document:
    """Strip tags from a static HTML page and return the visible text."""
    try:
        from bs4 import BeautifulSoup  # optional dependency
    except ImportError:  # pragma: no cover
        logger.warning("bs4 not installed; falling back to raw HTML for %s", path)
        return load_text_file(path)

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return Document(doc_id=path.stem, text=text, source_path=str(path))


def load_pdf_file(path: Path) -> Document:
    """Extract text from a PDF using pypdf if available."""
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError:  # pragma: no cover
        logger.warning("pypdf not installed; skipping PDF %s", path)
        return Document(doc_id=path.stem, text="", source_path=str(path))

    reader = PdfReader(str(path))
    pages: List[str] = [page.extract_text() or "" for page in reader.pages]
    return Document(
        doc_id=path.stem,
        text="\n".join(pages),
        source_path=str(path),
        metadata={"n_pages": len(pages)},
    )


_LOADERS = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".html": load_html_file,
    ".htm": load_html_file,
    ".pdf": load_pdf_file,
}


def load_directory(root: str | Path, recursive: bool = True) -> Iterable[Document]:
    """Yield Document objects for every supported file under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Ingestion root does not exist: {root}")

    walker = root.rglob("*") if recursive else root.iterdir()
    for fp in walker:
        if not fp.is_file():
            continue
        loader = _LOADERS.get(fp.suffix.lower())
        if loader is None:
            continue
        try:
            yield loader(fp)
        except Exception as exc:  # surface but do not halt the batch
            logger.error("Failed to load %s: %s", fp, exc)
