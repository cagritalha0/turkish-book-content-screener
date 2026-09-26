"""Charts and a downloadable PDF report for a :class:`BookReport`."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analyzer import BookReport  # noqa: E402

_FONT_NAME = "DejaVuSans"


def _register_unicode_font() -> str:
    """Helvetica has no ş/ğ/ı glyphs; DejaVu Sans ships with matplotlib and does."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if _FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(font_path)))
    return _FONT_NAME


def risk_histogram(report: BookReport, threshold: float, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(report.sentence_probs, bins=20, range=(0, 1), color="#4C72B0", edgecolor="white")
    ax.axvline(threshold, color="#C44E52", linestyle="--", label=f"threshold = {threshold}")
    ax.set_xlabel("Sentence risk score")
    ax.set_ylabel("Sentences")
    ax.set_title("Distribution of sentence risk scores")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "risk_histogram.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def pdf_report(report: BookReport, title: str, out_dir: Path, max_examples: int = 10) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    font = _register_unicode_font()
    path = out_dir / "analysis_report.pdf"
    width, height = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    y = height - 60

    def line(text: str, size: int = 11, gap: int = 16):
        nonlocal y
        c.setFont(font, size)
        for chunk in simpleSplit(text, font, size, width - 100) or [""]:
            if y < 60:
                c.showPage()
                y = height - 60
                c.setFont(font, size)
            c.drawString(50, y, chunk)
            y -= gap

    line(f"Content screening report: {title}", size=15, gap=24)
    line(f"Sentences analysed: {report.total_sentences}")
    line(f"Risky sentences: {len(report.risky_sentences)} ({report.risk_ratio:.1%})")
    line(f"Ateşman readability: {report.atesman} ({report.atesman_band})")
    line(f"Verdict: {report.verdict}")
    if report.reasons:
        line("Themes found in risky sentences: " + ", ".join(f"{k} ({v})" for k, v in report.reasons.items()))
    line("")
    line("Highest-scoring sentences:", size=12, gap=20)
    for sentence in report.risky_sentences[:max_examples]:
        line(f"• {sentence}", size=10, gap=14)
    line("")
    line("Automated screening only - a person should review flagged books.", size=9)
    c.save()
    return path


def new_output_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="bookscreener_"))
