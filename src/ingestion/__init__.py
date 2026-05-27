"""Document ingestion: loaders, normalizers, language detection, chunking."""

from src.ingestion.loaders import load_directory, load_text_file
from src.ingestion.normalize import normalize_text, detect_language
from src.ingestion.chunker import chunk_text, Chunk

__all__ = [
    "load_directory",
    "load_text_file",
    "normalize_text",
    "detect_language",
    "chunk_text",
    "Chunk",
]
