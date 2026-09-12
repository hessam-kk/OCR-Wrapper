"""PDF batch input discovery and output naming."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class PDFJob:
    """A PDF input paired with the output path base used by exporters."""

    input_path: Path
    output_base: Path


@dataclass(frozen=True)
class BatchSummary:
    """Outcome of a batch run."""

    completed: int
    failures: tuple[tuple[Path, str], ...]
    stopped: bool

    @property
    def succeeded(self) -> int:
        return self.completed - len(self.failures)


def pdfs_in_folder(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted by filename."""
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"PDF folder not found: {folder}")

    pdfs = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {folder}")
    return pdfs


def validate_pdf_paths(paths: Iterable[Path]) -> list[Path]:
    """Validate explicit PDF paths and remove duplicates without reordering."""
    pdfs: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")

        identity = path.resolve()
        if identity not in seen:
            seen.add(identity)
            pdfs.append(path)

    if not pdfs:
        raise ValueError("At least one PDF file is required")
    return pdfs


def create_pdf_jobs(
    pdf_paths: Iterable[Path],
    output_dir: Path,
    *,
    separate_folders: bool = False,
) -> list[PDFJob]:
    """Create deterministic ``<pdf stem>_transcript`` output paths."""
    pdfs = validate_pdf_paths(pdf_paths)
    output_dir = Path(output_dir)
    jobs: list[PDFJob] = []
    output_names: dict[str, Path] = {}

    for pdf_path in pdfs:
        transcript_name = f"{pdf_path.stem}_transcript"
        name_key = transcript_name.casefold()
        if name_key in output_names:
            first = output_names[name_key]
            raise ValueError("PDFs must have unique filenames to avoid overwriting output: " f"{first} and {pdf_path}")
        output_names[name_key] = pdf_path

        output_base = output_dir / transcript_name
        if separate_folders:
            output_base /= transcript_name
        jobs.append(PDFJob(pdf_path, output_base))

    return jobs


def process_pdf_jobs(
    jobs: Iterable[PDFJob],
    process: Callable[[PDFJob, int, int], None],
    *,
    log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
) -> BatchSummary:
    """Process PDF jobs independently so one bad document does not end a batch."""
    jobs = list(jobs)
    failures: list[tuple[Path, str]] = []
    completed = 0
    if log is None:
        log = print

    for index, job in enumerate(jobs, start=1):
        if should_stop():
            return BatchSummary(completed, tuple(failures), stopped=True)

        log(f"\n=== PDF {index}/{len(jobs)}: {job.input_path.name} ===")
        try:
            process(job, index, len(jobs))
        except Exception as exc:
            failures.append((job.input_path, str(exc)))
            log(f"[ERROR] {job.input_path.name}: {exc}")
        completed += 1

    return BatchSummary(completed, tuple(failures), stopped=False)
