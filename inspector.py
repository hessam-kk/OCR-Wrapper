"""pdf-inspector engine: fast text extraction from text-based PDFs."""

from pathlib import Path


def inspector_to_markdown(pdf_path: Path) -> str:
    try:
        import pdf_inspector
    except ImportError:
        raise RuntimeError(
            "pdf-inspector is not installed. Run: pip install pdf-inspector"
        ) from None

    r = pdf_inspector.process_pdf(str(pdf_path))
    if r.markdown is None:
        raise ValueError(
            f"pdf-inspector could not extract text (pdf_type={r.pdf_type}); "
            "this PDF likely needs OCR - use the Bina engine"
        )
    return r.markdown
