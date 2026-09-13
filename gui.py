"""Qt (PySide6) desktop GUI for batch OCR.

The OCR pipeline runs on a background thread. It never touches a widget: the
worker reports progress back through Qt signals, and the queued connections
deliver them on the GUI thread.
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import settings_store
from model import MODEL_ID
from ocr import FORMATS, run_ocr_pages, write_outputs
from pages import get_page_images
from pdf_batch import BY_TYPE, PER_PDF, beside_input, create_jobs, place_pagemap
from pdf_pipeline import process_pdf
from transcriber import build_transcriber, uses_page_transcriber

DPI_CHOICES = ("150", "200", "300", "400")
PDF_FILTER = "PDF files (*.pdf);;All files (*)"


def _pdf_count_label(count):
    """Label for the read-only input field when several PDFs are selected."""
    return f"{count} PDF file{'s' if count != 1 else ''} selected"


def _radio_value(buttons, default):
    """Value of the checked radio button in a {value: button} mapping."""
    for value, button in buttons.items():
        if button.isChecked():
            return value
    return default


@dataclass(frozen=True)
class OCRSettings:
    """A snapshot of the form, taken on the GUI thread before a run starts."""

    input_type: str
    input_path: Path
    pdf_paths: list[Path]
    output_base: Path
    output_dir: str
    batch_layout: str
    formats: list[str]
    direction: str
    engine: str
    force_cpu: bool
    normalize: bool
    max_tokens: int
    page_limit: int
    dpi: int
    workers: int
    skip_ocr: bool


class OCRWorker(QObject):
    """Runs one OCR job off the GUI thread and reports back via signals."""

    log = Signal(str)
    status = Signal(str)
    progress = Signal(int, int)  # page, total
    failed = Signal(str)
    finished = Signal()
    download_prompt = Signal(float)  # model size in GB

    def __init__(self, settings: OCRSettings):
        super().__init__()
        self.settings = settings
        self.running = True
        self._download_answer = None
        self._download_answered = threading.Event()

    # --- control (GUI thread) ---

    def stop(self):
        self.running = False

    def answer_download(self, agreed):
        """Deliver the user's answer to the worker blocked in _ask_download."""
        self._download_answer = agreed
        self._download_answered.set()

    # --- worker thread ---

    def run(self):
        try:
            self._run()
        except Exception as error:
            self.log.emit(f"FATAL: {error}")
            self.failed.emit(str(error))
        finally:
            self.running = False
            self.finished.emit()

    def _should_stop(self):
        return not self.running

    def _ask_download(self, size_gb):
        """Ask the GUI thread whether to download the model, and wait for it."""
        self._download_answer = None
        self._download_answered.clear()
        self.download_prompt.emit(size_gb)
        self._download_answered.wait()
        return bool(self._download_answer)

    def _run(self):
        settings = self.settings

        if settings.skip_ocr:
            self.status.emit("Re-exporting...")
            write_outputs(
                None, settings.output_base, settings.formats,
                log=self.log.emit, direction=settings.direction,
            )
            self.status.emit("Done")
            return

        if settings.input_type == "pdfs":
            self._run_pdf_batch()
        elif settings.input_type == "pdf":
            self._run_single_pdf()
        else:
            self._run_image_folder()

    def _run_single_pdf(self):
        settings = self.settings
        if not settings.input_path.is_file():
            self.failed.emit(f"PDF not found: {settings.input_path}")
            return
        if uses_page_transcriber(settings.engine):
            transcribe = self._transcriber()
            if transcribe is None:
                return
            self.log.emit(
                f"Rendering PDF pages at {settings.dpi} DPI (lazy, page by page)..."
            )
        else:
            transcribe = None

        process_pdf(
            settings.input_path, settings.output_base, settings.formats,
            settings.direction, settings.engine,
            transcribe=transcribe, dpi=settings.dpi,
            limit=settings.page_limit or None, workers=settings.workers,
            log=self.log.emit, progress=self._page_progress(),
            should_stop=self._should_stop,
        )
        if not self._should_stop():
            self._document_completed()
        self.status.emit("Done")

    def _run_image_folder(self):
        settings = self.settings
        if not settings.input_path.is_dir():
            self.failed.emit(f"Folder not found: {settings.input_path}")
            return
        if not uses_page_transcriber(settings.engine):
            self.failed.emit(
                "pdf-inspector only processes PDF files. "
                "Pick a PDF or switch to another engine."
            )
            return

        pages = get_page_images(settings.input_path)
        if settings.page_limit:
            pages = pages[:settings.page_limit]
        total = len(pages)
        self.log.emit(f"Found {total} pages to process.")

        transcribe = self._transcriber()
        if transcribe is None:
            return

        workers = settings.workers
        if settings.engine == "bina" and workers > 1:
            self.log.emit("[INFO] bina engine is single-device - ignoring workers")
            workers = 1

        self.progress.emit(0, total)
        self.status.emit(f"Processing 0/{total} pages...")
        run_ocr_pages(
            transcribe, pages, settings.output_base, settings.formats,
            direction=settings.direction, total=total, workers=workers,
            # chrome's DLL races across threads but is safe across processes
            parallel_mode="process" if settings.engine == "chrome" else "thread",
            parallel_engine=settings.engine, log=self.log.emit,
            progress=self._page_progress(), should_stop=self._should_stop,
        )
        if not self._should_stop():
            self._document_completed()
        self.status.emit("Done")

    def _run_pdf_batch(self):
        settings = self.settings
        jobs = create_jobs(settings.pdf_paths, settings.output_dir, settings.batch_layout)
        if uses_page_transcriber(settings.engine):
            transcribe = self._transcriber()
            if transcribe is None:
                return
        else:
            transcribe = None

        failures = []
        completed = 0
        for index, job in enumerate(jobs, start=1):
            if self._should_stop():
                break
            self.log.emit(f"\n=== PDF {index}/{len(jobs)}: {job.input_path.name} ===")
            self.status.emit(f"PDF {index}/{len(jobs)}")
            try:
                process_pdf(
                    job.input_path, job.output_base, settings.formats,
                    settings.direction, settings.engine,
                    transcribe=transcribe, dpi=settings.dpi,
                    limit=settings.page_limit or None, workers=settings.workers,
                    log=self.log.emit,
                    progress=self._page_progress(index, len(jobs)),
                    should_stop=self._should_stop,
                )
                place_pagemap(job)
            except Exception as error:
                failures.append(job.input_path)
                self.log.emit(f"[ERROR] {job.input_path.name}: {error}")
            completed += 1
            if not self._should_stop():
                self._document_completed()

        self.log.emit(f"\nPDFs succeeded: {completed - len(failures)}/{len(jobs)}")
        if self._should_stop():
            self.status.emit("Stopped")
        elif failures:
            self.status.emit("Done with errors")
        else:
            self.status.emit("Done")

    def _document_completed(self):
        """Fill the progress bar once a document is done.

        Engines that report no per-page progress (pdf-inspector is a single
        call) would otherwise leave the bar sitting at zero.
        """
        self.progress.emit(1, 1)

    def _page_progress(self, document=None, documents=None):
        """Per-page callback shared by the single-file and batch pipelines."""
        def report(page, total, elapsed, name):
            self.progress.emit(page, total)
            if document is None:
                self.status.emit(f"Processing {page}/{total} pages... ({elapsed:.1f}s)")
            else:
                self.status.emit(
                    f"PDF {document}/{documents}, page {page}/{total} ({elapsed:.1f}s)"
                )

        return report

    def _transcriber(self):
        """Page transcriber for the selected engine, or None to stop cleanly.

        The engine wiring lives in transcriber.py so the CLI and the GUI build
        their engines the same way; None means the download was declined.
        """
        settings = self.settings
        return build_transcriber(
            settings.engine,
            normalize=settings.normalize,
            force_cpu=settings.force_cpu,
            max_new_tokens=settings.max_tokens,
            log=self.log.emit,
            confirm_download=self._ask_download,
            status=self.status.emit,
        )


class OCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR-Wrapper")
        self.resize(820, 700)

        self.worker = None
        self.thread = None
        self.running = False
        self.pdf_paths = []
        self._start_time = None

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick_timer)

        self._restore_settings()

    # --- UI construction ---

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self._build_input_box())
        layout.addWidget(self._build_output_box())
        layout.addWidget(self._build_options_box())
        layout.addWidget(self._build_progress_box())
        layout.addWidget(self._build_log_box(), 1)
        layout.addLayout(self._build_button_row())

    def _build_input_box(self):
        box = QGroupBox("Input Source")
        grid = QGridLayout(box)

        self.input_type_buttons = {
            "pdf": QRadioButton("PDF File"),
            "pdfs": QRadioButton("Batch PDFs"),
            "dir": QRadioButton("Image Folder"),
        }
        self.input_group = self._exclusive(self.input_type_buttons)
        self.input_type_buttons["pdf"].setChecked(True)
        for column, button in enumerate(self.input_type_buttons.values()):
            button.clicked.connect(self._input_type_changed)
            grid.addWidget(button, 0, column, Qt.AlignmentFlag.AlignLeft)

        path_row = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setReadOnly(True)
        path_row.addWidget(self.input_path, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_input)
        path_row.addWidget(browse)
        grid.addLayout(path_row, 1, 0, 1, len(self.input_type_buttons))
        grid.setColumnStretch(1, 1)
        return box

    def _build_output_box(self):
        box = QGroupBox("Output")
        grid = QGridLayout(box)

        self.output_label = QLabel("Transcript:")
        grid.addWidget(self.output_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        self.output_file = QLineEdit("book_transcript")
        grid.addWidget(self.output_file, 0, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        grid.addWidget(browse, 0, 2)

        # Checkboxes get their own rows: sharing a row with the entry makes the
        # window far wider than it needs to be.
        location_row = QHBoxLayout()
        self.folder_check = QCheckBox("Save in folder")
        self.folder_check.setChecked(True)
        location_row.addWidget(self.folder_check)
        self.same_dir_check = QCheckBox("Save next to input")
        self.same_dir_check.setToolTip(
            "Write the transcript beside the source PDF, or inside the image "
            "folder, instead of the current directory. The name above is kept."
        )
        location_row.addWidget(self.same_dir_check)
        location_row.addStretch(1)
        grid.addLayout(location_row, 1, 0, 1, 3)

        self.skip_ocr_check = QCheckBox("Skip OCR (re-export from existing md)")
        grid.addWidget(self.skip_ocr_check, 2, 0, 1, 3, Qt.AlignmentFlag.AlignLeft)

        grid.addWidget(QLabel("Formats:"), 3, 0, Qt.AlignmentFlag.AlignLeft)
        formats_row = QHBoxLayout()
        self.format_boxes = {}
        for fmt in FORMATS:
            check = QCheckBox(fmt)
            check.setChecked(fmt == "md")
            self.format_boxes[fmt] = check
            formats_row.addWidget(check)
        formats_row.addStretch(1)
        grid.addLayout(formats_row, 3, 1, 1, 2)

        self.layout_row = QWidget()
        layout_row = QHBoxLayout(self.layout_row)
        layout_row.setContentsMargins(0, 0, 0, 0)
        layout_row.addWidget(QLabel("Batch layout:"))
        self.layout_buttons = {
            PER_PDF: QRadioButton("One folder per PDF"),
            BY_TYPE: QRadioButton("Group into markdown/ and pagemaps/"),
        }
        self.layout_group = self._exclusive(self.layout_buttons)
        self.layout_buttons[PER_PDF].setChecked(True)
        for button in self.layout_buttons.values():
            layout_row.addWidget(button)
        layout_row.addStretch(1)
        grid.addWidget(self.layout_row, 4, 0, 1, 3)

        grid.setColumnStretch(1, 1)
        return box

    def _build_options_box(self):
        """Rows are independent layouts, not one grid.

        A single grid shares column widths across every row, which made these
        option rows as wide as the longest one.
        """
        box = QGroupBox("Options")
        column = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max tokens:"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(64, 4096)
        self.max_tokens.setValue(1024)
        row.addWidget(self.max_tokens)
        row.addSpacing(16)
        row.addWidget(QLabel("Page limit:"))
        self.page_limit = QSpinBox()
        self.page_limit.setRange(0, 99999)
        row.addWidget(self.page_limit)
        row.addSpacing(16)
        row.addWidget(QLabel("DPI:"))
        self.dpi_box = QComboBox()
        self.dpi_box.addItems(DPI_CHOICES)
        self.dpi_box.setCurrentText("300")
        row.addWidget(self.dpi_box)
        row.addStretch(1)
        column.addLayout(row)

        self.engine_buttons = {
            "chrome": QRadioButton("Chrome (Screen AI)"),
            "oneocr": QRadioButton("Windows (oneocr)"),
            "bina": QRadioButton("bina (OCR)"),
            "inspector": QRadioButton("pdf-inspector"),
        }
        self.engine_group = self._exclusive(self.engine_buttons)
        self.engine_buttons["chrome"].setChecked(True)
        engine_grid = QGridLayout()
        engine_grid.addWidget(QLabel("Engine:"), 0, 0, Qt.AlignmentFlag.AlignLeft)
        for index, button in enumerate(self.engine_buttons.values()):
            engine_grid.addWidget(button, index // 2, 1 + index % 2, Qt.AlignmentFlag.AlignLeft)
        engine_grid.setColumnStretch(2, 1)
        column.addLayout(engine_grid)

        row = QHBoxLayout()
        row.addWidget(QLabel("Device:"))
        self.device_buttons = {
            "cuda": QRadioButton("GPU"),
            "cpu": QRadioButton("CPU"),
        }
        self.device_group = self._exclusive(self.device_buttons)
        self.device_buttons["cuda" if torch.cuda.is_available() else "cpu"].setChecked(True)
        for button in self.device_buttons.values():
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)

        row = QHBoxLayout()
        self.normalize_check = QCheckBox("Normalize Persian (half-space)")
        self.normalize_check.setChecked(True)
        row.addWidget(self.normalize_check)
        row.addSpacing(16)
        row.addWidget(QLabel("Workers:"))
        self.workers = QSpinBox()
        self.workers.setRange(1, 8)
        row.addWidget(self.workers)
        row.addSpacing(16)
        row.addWidget(QLabel("Dir:"))
        self.direction_buttons = {
            "rtl": QRadioButton("RTL"),
            "ltr": QRadioButton("LTR"),
        }
        self.direction_group = self._exclusive(self.direction_buttons)
        self.direction_buttons["rtl"].setChecked(True)
        for button in self.direction_buttons.values():
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)
        return box

    def _build_progress_box(self):
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        return box

    def _build_log_box(self):
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        layout.addWidget(self.log)
        return box

    def _build_button_row(self):
        row = QHBoxLayout()
        self.timer_label = QLabel("0s")
        row.addWidget(self.timer_label)
        row.addStretch(1)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.stop_btn)
        self.start_btn = QPushButton("Start OCR")
        self.start_btn.clicked.connect(self._start)
        row.addWidget(self.start_btn)
        return row

    def _exclusive(self, buttons):
        """Make a {value: button} mapping mutually exclusive.

        Qt radio buttons sharing a parent widget are auto-exclusive, so the
        engine, device, and direction buttons would otherwise all be one group.
        """
        group = QButtonGroup(self)
        for button in buttons.values():
            group.addButton(button)
        return group

    # --- widgets -> settings ---

    def _input_type(self):
        return _radio_value(self.input_type_buttons, "pdf")

    def _input_path(self):
        return Path(self.input_path.text().strip())

    def _output_base(self):
        base = Path(self.output_file.text())
        if self.same_dir_check.isChecked():
            base = beside_input(base, self._input_path(), self._input_type() == "dir")
        if self.folder_check.isChecked():
            base = base / base.name
        return base

    def _settings(self):
        return OCRSettings(
            input_type=self._input_type(),
            input_path=Path(self.input_path.text().strip()),
            pdf_paths=list(self.pdf_paths),
            output_base=self._output_base(),
            output_dir=self.output_file.text().strip() or "transcripts",
            batch_layout=_radio_value(self.layout_buttons, PER_PDF),
            formats=[fmt for fmt, box in self.format_boxes.items() if box.isChecked()],
            direction=_radio_value(self.direction_buttons, "rtl"),
            engine=_radio_value(self.engine_buttons, "chrome"),
            force_cpu=_radio_value(self.device_buttons, "cuda") == "cpu",
            normalize=self.normalize_check.isChecked(),
            max_tokens=self.max_tokens.value(),
            page_limit=self.page_limit.value(),
            dpi=int(self.dpi_box.currentText()),
            workers=self.workers.value(),
            skip_ocr=self.skip_ocr_check.isChecked(),
        )

    # --- input/output selection ---

    def _input_type_changed(self):
        self.pdf_paths = []
        self.input_path.setText("")
        is_batch = self._input_type() == "pdfs"
        self.output_label.setText("Output folder:" if is_batch else "Transcript:")
        if is_batch:
            self.output_file.setText("transcripts")
            self.folder_check.setVisible(False)
            # Batch inputs can live in different directories, so "next to input"
            # is ambiguous there - the output folder picker covers it.
            self.same_dir_check.setVisible(False)
            self.skip_ocr_check.setVisible(False)
            self.layout_row.setVisible(True)
        else:
            self.folder_check.setVisible(True)
            self.same_dir_check.setVisible(True)
            self.skip_ocr_check.setVisible(True)
            self.layout_row.setVisible(False)

    def _browse_input(self):
        input_type = self._input_type()
        if input_type == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select image folder")
        elif input_type == "pdfs":
            paths, _ = QFileDialog.getOpenFileNames(self, "Select PDF files", "", PDF_FILTER)
            if paths:
                self.pdf_paths = [Path(path) for path in paths]
                self.input_path.setText(_pdf_count_label(len(paths)))
                self.output_file.setText(str(self.pdf_paths[0].parent / "transcripts"))
            return
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select PDF file", "", PDF_FILTER)
        if path:
            self.input_path.setText(path)
            if input_type == "pdf":
                self.output_file.setText(Path(path).stem + "_transcript")

    def _browse_output(self):
        if self._input_type() == "pdfs":
            path = QFileDialog.getExistingDirectory(self, "Select output folder")
            if path:
                self.output_file.setText(path)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript as", self.output_file.text(),
            "Markdown (*.md);;Text (*.txt);;All files (*)",
        )
        if path:
            self.output_file.setText(Path(path).stem)

    # --- settings persistence ---

    def _settings_payload(self):
        """Form values worth remembering.

        Raw widget state, not the derived output path, so restoring does not
        apply the folder options a second time.
        """
        return {
            "input_type": self._input_type(),
            "input_path": self.input_path.text().strip(),
            "pdf_paths": [str(path) for path in self.pdf_paths],
            "output_file": self.output_file.text().strip(),
            "folder_check": self.folder_check.isChecked(),
            "same_dir_check": self.same_dir_check.isChecked(),
            "skip_ocr": self.skip_ocr_check.isChecked(),
            "formats": [fmt for fmt, box in self.format_boxes.items() if box.isChecked()],
            "batch_layout": _radio_value(self.layout_buttons, PER_PDF),
            "engine": _radio_value(self.engine_buttons, "chrome"),
            "device": _radio_value(self.device_buttons, "cuda"),
            "direction": _radio_value(self.direction_buttons, "rtl"),
            "normalize": self.normalize_check.isChecked(),
            "max_tokens": self.max_tokens.value(),
            "page_limit": self.page_limit.value(),
            "dpi": self.dpi_box.currentText(),
            "workers": self.workers.value(),
        }

    def _save_settings(self):
        try:
            settings_store.save(self._settings_payload())
        except OSError as error:
            # Forgetting the form is not worth blocking the run or the exit.
            self._log(f"[WARN] Could not save settings: {error}")

    def _restore_settings(self):
        """Apply the saved form, falling back to the defaults for anything odd.

        Unknown or malformed values are skipped, so a settings file written by
        an older version can never leave the form in an impossible state.
        """
        payload = settings_store.load()
        if payload.get("input_type") in self.input_type_buttons:
            self.input_type_buttons[payload["input_type"]].setChecked(True)
        # _input_type_changed() clears the path fields, so the remembered input
        # is restored after it has run.
        self._input_type_changed()

        for key, buttons in (
            ("engine", self.engine_buttons),
            ("device", self.device_buttons),
            ("direction", self.direction_buttons),
            ("batch_layout", self.layout_buttons),
        ):
            if payload.get(key) in buttons:
                buttons[payload[key]].setChecked(True)

        for key, widget in (
            ("folder_check", self.folder_check),
            ("same_dir_check", self.same_dir_check),
            ("skip_ocr", self.skip_ocr_check),
            ("normalize", self.normalize_check),
        ):
            if isinstance(payload.get(key), bool):
                widget.setChecked(payload[key])

        formats = payload.get("formats")
        if isinstance(formats, list):
            wanted = [fmt for fmt in formats if fmt in self.format_boxes]
            # Never restore an empty selection: Start would refuse to run.
            if wanted:
                for fmt, box in self.format_boxes.items():
                    box.setChecked(fmt in wanted)

        for key, widget in (
            ("max_tokens", self.max_tokens),
            ("page_limit", self.page_limit),
            ("workers", self.workers),
        ):
            value = payload.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                widget.setValue(value)  # QSpinBox clamps to its own range
        if payload.get("dpi") in DPI_CHOICES:
            self.dpi_box.setCurrentText(payload["dpi"])

        if isinstance(payload.get("output_file"), str) and payload["output_file"]:
            self.output_file.setText(payload["output_file"])
        self._restore_input_path(payload)

    def _restore_input_path(self, payload):
        saved_pdfs = payload.get("pdf_paths")
        if isinstance(saved_pdfs, list):
            names = [value for value in saved_pdfs if isinstance(value, str)]
            # A remembered input may have moved or been renamed since last run.
            self.pdf_paths = [Path(name) for name in names if Path(name).is_file()]
            dropped = len(names) - len(self.pdf_paths)
            if dropped > 0:
                self._log(f"[INFO] Skipped {dropped} saved PDF(s) that no longer exist.")

        if self._input_type() == "pdfs" and self.pdf_paths:
            self.input_path.setText(_pdf_count_label(len(self.pdf_paths)))
        elif isinstance(payload.get("input_path"), str):
            self.input_path.setText(payload["input_path"])

    def closeEvent(self, event):
        """Remember the form so the next launch starts where this one left off."""
        self._save_settings()
        super().closeEvent(event)

    # --- GUI-thread slots ---

    def _log(self, message):
        self.log.appendPlainText(message)
        scrollbar = self.log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_status(self, text):
        self.status_label.setText(text)

    def _on_progress(self, page, total):
        if total > 0:
            self.progress.setMaximum(total)
        self.progress.setValue(page)

    def _on_failed(self, message):
        self.status_label.setText("Error")
        # Keep the reason in the log too: the dialog is dismissed and gone.
        self._log(f"[ERROR] {message}")
        QMessageBox.critical(self, "Error", message)

    def _on_download_prompt(self, size_gb):
        answer = QMessageBox.question(
            self,
            "Model not downloaded",
            f"Model {MODEL_ID} is not cached locally.\n\n"
            f"Download size: ~{size_gb:.1f} GB\n\n"
            "Download it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if self.worker is not None:
            self.worker.answer_download(answer == QMessageBox.StandardButton.Yes)

    def _on_finished(self):
        self.running = False
        self.timer.stop()
        self._finalize_timer()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # --- run control ---

    def _start(self):
        input_type = self._input_type()
        engine = _radio_value(self.engine_buttons, "chrome")

        if input_type == "pdfs":
            if not self.pdf_paths:
                QMessageBox.warning(
                    self, "Missing input", "Please select one or more PDF files."
                )
                return
        elif not self.input_path.text().strip():
            QMessageBox.warning(
                self, "Missing input", "Please select an input folder or PDF file."
            )
            return

        if not any(box.isChecked() for box in self.format_boxes.values()):
            QMessageBox.warning(
                self, "No formats selected", "Pick at least one output format."
            )
            return

        if input_type == "dir" and engine == "inspector":
            QMessageBox.warning(
                self, "Unsupported engine",
                "pdf-inspector only processes PDF files. Pick a PDF or switch to "
                "another engine.",
            )
            return

        self._save_settings()
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting...")
        self._log("Starting OCR...")

        self._start_time = time.time()
        self.timer.start(1000)

        self.worker = OCRWorker(self._settings())
        self.worker.log.connect(self._log)
        self.worker.status.connect(self._on_status)
        self.worker.progress.connect(self._on_progress)
        self.worker.download_prompt.connect(self._on_download_prompt)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.thread = threading.Thread(target=self.worker.run, daemon=True)
        self.thread.start()

    def _stop(self):
        self.running = False
        if self.worker is not None:
            self.worker.stop()
        self._finalize_timer()
        self._log("Stopping after current page...")
        self.stop_btn.setEnabled(False)

    def _tick_timer(self):
        if self._start_time is not None:
            self.timer_label.setText(f"{int(time.time() - self._start_time)}s")

    def _finalize_timer(self):
        self.timer.stop()
        self._tick_timer()


def launch_gui():
    """Start the Qt event loop and show the main window."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = OCRApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(launch_gui())
