"""Batch OCR using the Bina-0.1 Persian OCR vision-language model.

Usage:
    python book_ocr_batch.py --input_dir ./book_pages --output_file book_transcript.md
    python book_ocr_batch.py --pdf book.pdf --output_file book_transcript.md
    python book_ocr_batch.py --gui
"""

import argparse
import itertools
import tempfile
import tkinter as tk
from pathlib import Path

from tqdm import tqdm

from chrome_ocr_engine import chrome_transcribe_page, get_screenai_engine
from gui import OCRApp
from model import (
    ENGINES,
    MODEL_ID,
    load_model,
    model_cache_info,
    repo_size_gb,
    write_inspector_transcript,
)
from normalize import get_normalizer, normalize_transcribe
from ocr import FORMATS, run_ocr_pages, transcribe_page, write_outputs
from pages import get_page_images, get_pdf_images
from pdf_batch import (
    create_pdf_jobs,
    pdfs_in_folder,
    process_pdf_jobs,
    validate_pdf_paths,
)
from windows_ocr import get_ocr_engine, oneocr_transcribe_page


def _confirm_bina_download() -> bool:
    cached, _ = model_cache_info()
    if cached:
        return True

    print(f"[INFO] Model {MODEL_ID} is not downloaded yet (~{repo_size_gb():.1f} GB).")
    try:
        confirm = input("Download it now? [y/N] ")
    except EOFError:
        confirm = ""
    return confirm.strip().lower() in ("y", "yes")


def _create_transcriber(args):
    """Initialize the selected OCR engine once for the entire command."""
    if args.engine == "oneocr":
        engine = get_ocr_engine()
        transcribe = lambda path: oneocr_transcribe_page(engine, path)
    elif args.engine == "chrome":
        engine = get_screenai_engine()
        transcribe = lambda path: chrome_transcribe_page(engine, path)
    else:
        if not _confirm_bina_download():
            print("Aborted - model not downloaded.")
            return None
        processor, model, _device = load_model(force_cpu=args.cpu)
        transcribe = lambda path: transcribe_page(processor, model, path, args.max_new_tokens)

    if args.normalize:
        transcribe = normalize_transcribe(transcribe, get_normalizer())
        print("[INFO] Persian normalization enabled (hazm)")
    return transcribe


def _process_pdf_with_ocr(args, pdf_path, output_base, transcribe, progress=None):
    """Render and process one PDF with an already initialized OCR engine."""
    with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as temp_dir:
        pages, total_pages = get_pdf_images(pdf_path, Path(temp_dir), dpi=args.dpi)
        if args.limit:
            pages = itertools.islice(pages, args.limit)
            total_pages = min(total_pages, args.limit)

        print(f"[INFO] Found {total_pages} pages to process.")
        workers = args.workers
        if args.engine == "bina" and workers > 1:
            print("[INFO] bina engine is single-device - ignoring --workers")
            workers = 1

        run_ocr_pages(
            transcribe,
            pages,
            output_base,
            args.formats,
            direction=args.direction,
            total=total_pages,
            workers=workers,
            parallel_mode="process" if args.engine == "chrome" else "thread",
            parallel_engine=args.engine,
            log=print,
            progress=progress,
        )


def _run_pdf_batch(args, pdf_paths):
    jobs = create_pdf_jobs(
        pdf_paths,
        Path(args.output_dir),
        separate_folders=args.save_in_folders,
    )
    print(f"[INFO] Batch contains {len(jobs)} PDF files.")

    if args.skip_ocr:
        process = lambda job, _index, _total: write_outputs(None, job.output_base, args.formats, print, args.direction)
    elif args.engine == "inspector":
        process = lambda job, _index, _total: write_inspector_transcript(
            job.input_path,
            job.output_base,
            args.formats,
            args.direction,
        )
    else:
        transcribe = _create_transcriber(args)
        if transcribe is None:
            return 1

        def process(job, _index, _total):
            _process_pdf_with_ocr(
                args,
                job.input_path,
                job.output_base,
                transcribe,
                progress=lambda i, total, elapsed, name: tqdm.write(f"[{i}/{total}] {name} - {elapsed:.2f}s"),
            )

    summary = process_pdf_jobs(jobs, process)
    print("\n=== Batch summary ===")
    print(f"PDFs succeeded: {summary.succeeded}/{len(jobs)}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    if summary.failures:
        print(f"PDFs failed: {len(summary.failures)}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="Batch OCR a folder of book page images or a PDF file.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input_dir", help="Folder containing page images")
    input_group.add_argument("--pdf", help="Path to one PDF file")
    input_group.add_argument("--pdfs", nargs="+", help="Paths to multiple PDF files")
    input_group.add_argument("--pdf_dir", help="Folder containing PDF files")
    parser.add_argument(
        "--output_file",
        default="book_transcript",
        help="Transcript output base name (extension added per format)",
    )
    parser.add_argument(
        "--output_dir",
        default="transcripts",
        help="Output folder for --pdfs or --pdf_dir",
    )
    parser.add_argument(
        "--save_in_folders",
        action="store_true",
        help="Put each batch transcript in its own subfolder",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=["md"],
        help="Output formats to write (md txt epub pdf azw3; epub/pdf/azw3 need calibre)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens generated per page")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering DPI for OCR engines")
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="bina",
        help="OCR engine to use (bina: vision model, inspector: fast text extraction, PDF only, oneocr: Windows Snipping Tool OCR)",
    )
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize Persian text with hazm (reinserts half-spaces/ZWNJ)",
    )
    parser.add_argument(
        "--direction",
        choices=("rtl", "ltr"),
        default="rtl",
        help="Text direction of the exported markdown",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel page workers (2-8 speed up oneocr/chrome; bina stays 1)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="If the .md exists, skip OCR and only convert to the selected formats",
    )
    args = parser.parse_args()

    # Launch GUI if --gui or no CLI args provided
    has_input = args.input_dir or args.pdf or args.pdfs or args.pdf_dir
    if args.gui or not has_input:
        root = tk.Tk()
        OCRApp(root)
        root.mainloop()
        return 0

    if args.engine == "inspector" and args.input_dir:
        parser.error("--engine inspector requires --pdf (pdf-inspector only processes PDFs).")

    if args.pdfs or args.pdf_dir:
        try:
            pdf_paths = validate_pdf_paths(Path(path) for path in args.pdfs) if args.pdfs else pdfs_in_folder(Path(args.pdf_dir))
            return _run_pdf_batch(args, pdf_paths)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            parser.error(str(exc))

    output_base = Path(args.output_file)

    if args.skip_ocr:
        write_outputs(None, output_base, args.formats, print, args.direction)
        print("[INFO] OCR skipped - re-exported from existing markdown")
        return 0

    if args.engine == "inspector":
        write_inspector_transcript(Path(args.pdf), output_base, args.formats, args.direction)
        return 0

    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            parser.error(f"PDF file not found: {pdf_path}")

        transcribe = _create_transcriber(args)
        if transcribe is None:
            return 1
        _process_pdf_with_ocr(
            args,
            pdf_path,
            output_base,
            transcribe,
            progress=lambda i, total, elapsed, name: tqdm.write(f"[{i}/{total}] {name} - {elapsed:.2f}s"),
        )
        return 0

    input_dir = Path(args.input_dir)
    pages = get_page_images(input_dir)
    total_pages = len(pages)
    if args.limit:
        pages = pages[: args.limit]
        total_pages = len(pages)
    print(f"[INFO] Found {total_pages} pages to process.")

    transcribe = _create_transcriber(args)
    if transcribe is None:
        return 1

    def show_progress(i, tot, elapsed, name):
        tqdm.write(f"[{i}/{tot}] {name} - {elapsed:.2f}s")

    if args.engine == "bina" and args.workers > 1:
        print("[INFO] bina engine is single-device - ignoring --workers")
        args.workers = 1
    # chrome's DLL races across threads but is safe across processes
    parallel_mode = "process" if args.engine == "chrome" else "thread"

    run_ocr_pages(
        transcribe,
        pages,
        output_base,
        args.formats,
        direction=args.direction,
        total=total_pages,
        workers=args.workers,
        parallel_mode=parallel_mode,
        parallel_engine=args.engine,
        log=print,
        progress=show_progress,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
