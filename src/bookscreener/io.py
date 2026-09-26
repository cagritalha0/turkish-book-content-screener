"""Reading books from disk."""

from __future__ import annotations

from pathlib import Path


def read_book(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    if suffix == ".txt":
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "cp1254", "iso-8859-9"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {path.name}; save it as UTF-8.")
    raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf or .txt.")
