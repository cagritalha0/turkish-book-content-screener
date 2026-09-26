"""Gradio demo: upload a Turkish book (.pdf / .txt) and get a content-screening report.

python app.py --model-dir models/berturk-offensive [--meta-model models/meta_catboost.cbm]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from bookscreener import BookAnalyzer, load_lexicon
from bookscreener.io import read_book
from bookscreener.report import new_output_dir, pdf_report, risk_histogram
from bookscreener.scorer import BertScorer


def load_meta_model(path: str | None):
    if not path:
        return None
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(path)
    return model


def build_app(analyzer: BookAnalyzer) -> gr.Blocks:
    def run(file):
        if file is None:
            raise gr.Error("Please upload a .pdf or .txt file.")
        path = Path(file if isinstance(file, str) else file.name)
        try:
            report = analyzer.analyze(read_book(path))
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        out_dir = new_output_dir()
        summary = "\n".join(
            [
                f"Sentences analysed: {report.total_sentences}",
                f"Risky sentences: {len(report.risky_sentences)} ({report.risk_ratio:.1%})",
                f"Ateşman readability: {report.atesman} ({report.atesman_band})",
                f"Verdict: {report.verdict}",
            ]
            + (["Themes: " + ", ".join(f"{k} ({v})" for k, v in report.reasons.items())] if report.reasons else [])
        )
        examples = "\n\n".join(report.risky_sentences[:10]) or "No risky sentences found."
        chart = risk_histogram(report, analyzer.sentence_threshold, out_dir)
        pdf = pdf_report(report, path.name, out_dir)
        return summary, examples, str(chart), str(pdf)

    with gr.Blocks(title="Turkish Book Content Screener", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "## 📚 Turkish Book Content Screener\n"
            "Upload a book to check whether it may contain content unsuitable for children."
        )
        with gr.Row():
            file_input = gr.File(label="Book (.pdf or .txt)", file_types=[".pdf", ".txt"])
            button = gr.Button("Analyse", variant="primary")
        with gr.Row():
            summary = gr.Textbox(label="Summary", lines=6, interactive=False)
            examples = gr.Textbox(label="Highest-scoring sentences", lines=6, interactive=False)
        with gr.Row():
            chart = gr.Image(label="Sentence risk distribution")
            pdf = gr.File(label="PDF report")
        gr.Markdown("_Automated screening only. A person should review any flagged book._")
        button.click(run, inputs=file_input, outputs=[summary, examples, chart, pdf])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-dir", required=True, help="Fine-tuned BERTurk directory (scripts/train_bert.py output)"
    )
    parser.add_argument("--meta-model", help="Optional CatBoost meta-model (scripts/train_meta.py output)")
    parser.add_argument("--lexicon", default=None, help="Custom lexicon file")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    lexicon = load_lexicon(args.lexicon) if args.lexicon else load_lexicon()
    analyzer = BookAnalyzer(BertScorer(args.model_dir), lexicon, meta_model=load_meta_model(args.meta_model))
    build_app(analyzer).launch(share=args.share)


if __name__ == "__main__":
    main()
