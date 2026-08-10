"""Batch OCR using the Bina-0.1 Persian OCR vision-language model.

Usage:
    python book_ocr_batch.py --input_dir ./book_pages --output_file book_transcript.md
    python book_ocr_batch.py --pdf book.pdf --output_file book_transcript.md
    python book_ocr_batch.py --gui
"""

import argparse
import sys
import tempfile
import tkinter as tk
from pathlib import Path

from tqdm import tqdm

from gui import OCRApp
from model import ENGINES, MODEL_ID, load_model, model_cache_info, repo_size_gb, write_inspector_transcript
from ocr import FORMATS, run_ocr_pages, transcribe_page
from pages import get_page_images, get_pdf_images
from normalize import get_normalizer, normalize_transcribe
from chrome_ocr_engine import chrome_transcribe_page, get_screenai_engine
from windows_ocr import get_ocr_engine, oneocr_transcribe_page


def main():
    parser = argparse.ArgumentParser(description="Batch OCR a folder of book page images or a PDF file.")
    parser.add_argument("--input_dir", help="Folder containing page images")
    parser.add_argument("--pdf", help="Path to a PDF file")
    parser.add_argument("--output_file", default="book_transcript", help="Transcript output base name (extension added per format)")
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=["md"], help="Output formats to write (md txt epub pdf azw3; epub/pdf/azw3 need calibre)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens generated per page")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages")
    parser.add_argument("--engine", choices=ENGINES, default="bina", help="OCR engine to use (bina: vision model, inspector: fast text extraction, PDF only, oneocr: Windows Snipping Tool OCR)")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    parser.add_argument("--normalize", action="store_true", help="Normalize Persian text with hazm (reinserts half-spaces/ZWNJ)")
    parser.add_argument("--direction", choices=("rtl", "ltr"), default="rtl", help="Text direction of the exported markdown")
    args = parser.parse_args()

    # Launch GUI if --gui or no CLI args provided
    if args.gui or (not args.input_dir and not args.pdf):
        root = tk.Tk()
        OCRApp(root)
        root.mainloop()
        return

    if not args.input_dir and not args.pdf:
        parser.error("Either --input_dir or --pdf is required.")
    if args.input_dir and args.pdf:
        parser.error("Use either --input_dir or --pdf, not both.")
    if args.engine == "inspector" and not args.pdf:
        parser.error("--engine inspector requires --pdf (pdf-inspector only processes PDFs).")

    output_base = Path(args.output_file)

    if args.engine == "inspector":
        write_inspector_transcript(Path(args.pdf), output_base, args.formats, args.direction)
        return

    pdf_tmp_dir = None

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            parser.error(f"PDF file not found: {pdf_path}")
        pdf_tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_pdf_"))
        pages = get_pdf_images(pdf_path, pdf_tmp_dir)
    else:
        input_dir = Path(args.input_dir)
        pages = get_page_images(input_dir)

    if args.limit:
        pages = pages[: args.limit]

    cached, _ = model_cache_info()
    if not cached:
        print(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{repo_size_gb():.1f} GB).")
        try:
            confirm = input("Download it now? [y/N] ")
        except EOFError:
            confirm = ""
        if confirm.strip().lower() not in ("y", "yes"):
            print("Aborted - model not downloaded.")
            sys.exit(0)

    if args.engine == "oneocr":
        engine = get_ocr_engine()
        transcribe = lambda p: oneocr_transcribe_page(engine, p)
    elif args.engine == "chrome":
        engine = get_screenai_engine()
        transcribe = lambda p: chrome_transcribe_page(engine, p)
    else:
        processor, model, device = load_model(force_cpu=args.cpu)
        transcribe = lambda p: transcribe_page(processor, model, p, args.max_new_tokens)

    if args.normalize:
        transcribe = normalize_transcribe(transcribe, get_normalizer())
        print("[INFO] Persian normalization enabled (hazm)")

    def show_progress(i, tot, elapsed, name):
        tqdm.write(f"[{i}/{tot}] {name} - {elapsed:.2f}s")

    run_ocr_pages(
        transcribe, pages, output_base, args.formats,
        direction=args.direction,
        log=print,
        progress=show_progress,
    )

    if pdf_tmp_dir:
        for f in pdf_tmp_dir.iterdir():
            f.unlink()
        pdf_tmp_dir.rmdir()


if __name__ == "__main__":
    main()
