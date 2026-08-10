"""Page collection: image folders and PDF rendering."""

from pathlib import Path

import fitz  # PyMuPDF

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PDF_DPI = 150


def get_page_images(input_dir: Path):
    pages = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not pages:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    return pages


def get_pdf_images(pdf_path: Path, tmp_dir: Path, dpi: int = PDF_DPI):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        out = tmp_dir / f"page_{i + 1:04d}.png"
        pix.save(out)
        pages.append(out)
    doc.close()
    return pages
