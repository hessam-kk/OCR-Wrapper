"""Batch OCR using the Bina-0.1 Persian OCR vision-language model.

Usage:
    python main.py --input_dir ./book_pages --output_file book_transcript.md
    python main.py --pdf book.pdf --output_file book_transcript.md
    python main.py --gui
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from gui import launch_gui
from model import ENGINES
from ocr import FORMATS, run_ocr_pages
from pages import get_page_images
from pdf_batch import beside_input, create_jobs
from pdf_pipeline import process_pdf
from transcriber import build_transcriber, uses_page_transcriber


def confirm_download(size_gb):
    """Console prompt before pulling the model (a piped run declines)."""
    try:
        answer = input("Download it now? [y/N] ")
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def make_transcriber(args):
    """Transcriber for one file or an entire batch.

    None means either that the engine works on whole documents (pdf-inspector)
    or that the model download was declined.
    """
    if not uses_page_transcriber(args.engine):
        return None
    return build_transcriber(
        args.engine,
        normalize=args.normalize,
        force_cpu=args.cpu,
        max_new_tokens=args.max_new_tokens,
        log=print,
        confirm_download=confirm_download,
    )


def run_pdf_batch(args, parser):
    if args.skip_ocr:
        parser.error("--skip-ocr is not supported with --pdfs")
    jobs = create_jobs((Path(path) for path in args.pdfs), args.output_dir)
    transcribe = make_transcriber(args)
    if transcribe is None and uses_page_transcriber(args.engine):
        return 1

    failures = []
    for index, job in enumerate(jobs, start=1):
        print(f"\n=== PDF {index}/{len(jobs)}: {job.input_path.name} ===")
        try:
            process_pdf(
                job.input_path, job.output_base, args.formats, args.direction,
                args.engine, transcribe=transcribe, limit=args.limit,
                workers=args.workers, log=print,
            )
        except Exception as error:
            failures.append(job.input_path)
            print(f"[ERROR] {job.input_path.name}: {error}")

    print(f"\nPDFs succeeded: {len(jobs) - len(failures)}/{len(jobs)}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="Batch OCR a folder of book page images or a PDF file.")
    parser.add_argument("--input_dir", help="Folder containing page images")
    parser.add_argument("--pdf", help="Path to a PDF file")
    parser.add_argument("--pdfs", nargs="+", help="Paths to multiple PDF files")
    parser.add_argument("--output_file", default="book_transcript", help="Transcript output base name (extension added per format)")
    parser.add_argument("--same_dir", action="store_true",
                        help="Write outputs next to the input: beside the PDF, or inside the image folder")
    parser.add_argument("--output_dir", default="transcripts",
                        help="Output folder for --pdfs (one subfolder per PDF)")
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=["md"], help="Output formats to write (md txt epub pdf azw3; epub/pdf/azw3 need calibre)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max tokens generated per page")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages")
    parser.add_argument("--engine", choices=ENGINES, default="bina", help="OCR engine to use (bina: vision model, inspector: fast text extraction, PDF only, oneocr: Windows Snipping Tool OCR)")
    parser.add_argument("--gui", action="store_true", help="Launch the GUI instead of CLI")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")
    parser.add_argument("--normalize", action="store_true", help="Normalize Persian text with hazm (reinserts half-spaces/ZWNJ)")
    parser.add_argument("--direction", choices=("rtl", "ltr"), default="rtl", help="Text direction of the exported markdown")
    parser.add_argument("--workers", type=int, default=1, help="Parallel page workers (2-8 speed up oneocr/chrome; bina stays 1)")
    parser.add_argument("--skip-ocr", action="store_true", help="If the .md exists, skip OCR and only convert to the selected formats")
    args = parser.parse_args()

    # Launch GUI if --gui or no CLI args provided
    if args.gui or (not args.input_dir and not args.pdf and not args.pdfs):
        return launch_gui()

    if sum(bool(source) for source in (args.input_dir, args.pdf, args.pdfs)) != 1:
        parser.error("Use exactly one of --input_dir, --pdf, or --pdfs.")
    if args.engine == "inspector" and args.input_dir:
        parser.error("--engine inspector requires --pdf (pdf-inspector only processes PDFs).")
    if args.same_dir and args.pdfs:
        parser.error("--same_dir is not supported with --pdfs (use --output_dir).")
    if args.pdfs:
        return run_pdf_batch(args, parser)

    output_base = Path(args.output_file)
    if args.same_dir:
        output_base = beside_input(
            output_base, args.pdf or args.input_dir, source_is_dir=bool(args.input_dir),
        )

    if args.skip_ocr:
        from ocr import write_outputs
        write_outputs(None, output_base, args.formats, print, args.direction)
        print("[INFO] OCR skipped - re-exported from existing markdown")
        return

    if args.pdf:
        transcribe = make_transcriber(args)
        if transcribe is None and uses_page_transcriber(args.engine):
            return 1
        process_pdf(
            args.pdf, output_base, args.formats, args.direction, args.engine,
            transcribe=transcribe, limit=args.limit, workers=args.workers,
            log=print,
        )
        return 0

    pages = get_page_images(Path(args.input_dir))
    if args.limit:
        pages = pages[:args.limit]
    total_pages = len(pages)
    print(f"[INFO] Found {total_pages} pages to process.")

    transcribe = make_transcriber(args)
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
        transcribe, pages, output_base, args.formats,
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
