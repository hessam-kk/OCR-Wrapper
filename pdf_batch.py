"""Small helpers for applying the existing PDF pipeline to several files."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Some Windows applications still fail near the legacy 260-character limit.
MAX_OUTPUT_PATH = 240

@dataclass(frozen=True)
class PDFJob:
    input_path: Path
    output_base: Path


def beside_input(output_base, source, source_is_dir: bool = False) -> Path:
    """Move an output base next to its input.

    A PDF's outputs land in that PDF's own directory; an image folder's outputs
    land inside that folder. Only the name of `output_base` is kept, so an
    absolute path picked in the GUI does not drag outputs away from the input.
    """
    source = Path(source)
    directory = source if source_is_dir else source.parent
    return directory / Path(output_base).name


def validate_pdfs(paths: Iterable[Path]) -> list[Path]:
    """Validate and de-duplicate explicitly selected PDFs."""
    pdfs = []
    seen = set()
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        identity = path.resolve()
        if identity not in seen:
            seen.add(identity)
            pdfs.append(path)
    if not pdfs:
        raise ValueError("Select at least one PDF file")
    return pdfs


def create_jobs(paths: Iterable[Path], output_dir: Path) -> list[PDFJob]:
    """Build short, collision-free paths: one folder per PDF."""
    output_dir = Path(output_dir).resolve()
    jobs = []
    used_names = set()
    for pdf in validate_pdfs(paths):
        name = _document_name(pdf, output_dir, used_names)
        used_names.add(name.casefold())
        jobs.append(PDFJob(pdf, output_dir / name / "transcript"))
    return jobs


def _document_name(pdf: Path, output_dir: Path, used: set[str]) -> str:
    # transcript.ebook.md is the longest file written into the folder, so it
    # decides how much room a shortened folder name may take.
    suffix = "\\transcript.ebook.md"
    maximum = min(100, MAX_OUTPUT_PATH - len(str(output_dir)) - len(suffix) - 1)
    if maximum < 20:
        raise ValueError("Output folder path is too long; choose a shorter output folder")

    original = pdf.stem.rstrip(" .") or "document"
    candidate = original
    needs_hash = len(candidate) > maximum or candidate.casefold() in used
    if needs_hash:
        digest = hashlib.sha1(str(pdf.resolve()).encode("utf-8")).hexdigest()[:8]
        candidate = f"{original[:maximum - 9].rstrip()}-{digest}"
    return candidate
