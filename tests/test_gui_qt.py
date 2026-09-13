"""Headless smoke tests for the PySide6 GUI.

The GUI is driven offscreen through a real Qt event loop: the OCR worker still
runs on its own thread and reports back through queued signals, so these tests
exercise the same signal round-trip the desktop app relies on.

Only the pdf-inspector engine is covered - it needs no model download and no
Windows-only DLL. Text direction is LTR because the RTL path reshapes
Arabic-script runs with arabic-reshaper, which is an optional extra.
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Must be set before Qt is imported so no display/plugin is needed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import fitz
    import pdf_inspector  # noqa: F401  (pdf-inspector engine, exercised below)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    import gui
    import settings_store
except ImportError as error:  # pragma: no cover - depends on the environment
    QApplication = None
    QMessageBox = None
    gui = None
    SKIP_REASON = f"GUI test dependencies unavailable: {error}"
else:
    SKIP_REASON = ""


@unittest.skipIf(gui is None, SKIP_REASON)
class QtGuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One QApplication per process; widgets need it to exist first.
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Point the settings store at a throwaway directory: the tests must not
        # read, or overwrite, the real user's saved settings.
        override = patch.dict(
            os.environ, {settings_store.ENV_OVERRIDE: str(self.root / "config")}
        )
        override.start()
        self.addCleanup(override.stop)
        self.window = gui.OCRApp()
        # LTR keeps arabic-reshaper out of the picture (optional extra).
        self.window.direction_buttons["ltr"].setChecked(True)
        self.window.engine_buttons["inspector"].setChecked(True)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    # --- helpers ---

    def _text_pdf(self, name="paper.pdf", directory=None):
        """A small PDF with a real text layer."""
        path = Path(directory or self.root) / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "OCR wrapper test page.")
        doc.save(path)
        doc.close()
        return path

    def _textless_pdf(self, name="scanned.pdf"):
        """A PDF with no text layer - pdf-inspector rejects it."""
        path = self.root / name
        doc = fitz.open()
        page = doc.new_page()
        page.draw_rect(fitz.Rect(72, 72, 300, 200), color=(0, 0, 0), width=2)
        doc.save(path)
        doc.close()
        return path

    def _select_single_pdf(self, pdf, output_base, in_folder=False):
        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(pdf))
        self.window.output_file.setText(str(output_base))
        self.window.folder_check.setChecked(in_folder)
        for fmt, box in self.window.format_boxes.items():
            box.setChecked(fmt == "md")

    def _select_batch(self, pdfs, output_dir):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.window.pdf_paths = list(pdfs)
        self.window.input_path.setText(gui._pdf_count_label(len(pdfs)))
        self.window.output_file.setText(str(output_dir))

    def _start_and_wait(self, timeout=120, window=None):
        window = window or self.window
        window._start()
        self.assertIsNotNone(window.thread, "the worker thread was never started")
        deadline = time.time() + timeout
        while window.thread.is_alive():
            self.app.processEvents()
            time.sleep(0.01)
            if time.time() > deadline:
                self.fail("the OCR worker did not finish in time")
        # Let the queued progress/finished signals reach the GUI thread.
        for _ in range(50):
            self.app.processEvents()
            time.sleep(0.005)

    # --- tests ---

    def test_launch_gui_enters_and_leaves_the_event_loop(self):
        """main.py launches the app through this entry point."""
        QTimer.singleShot(0, self.app.quit)
        self.assertEqual(gui.launch_gui(), 0)

    def test_radio_groups_are_independent(self):
        """Engine, device, and direction radios must not uncheck each other."""
        self.window.engine_buttons["bina"].setChecked(True)
        self.assertFalse(self.window.engine_buttons["chrome"].isChecked())
        self.assertTrue(
            self.window.device_buttons["cuda"].isChecked()
            or self.window.device_buttons["cpu"].isChecked()
        )
        self.assertTrue(self.window.direction_buttons["ltr"].isChecked())

        self.window.device_buttons["cpu"].setChecked(True)
        self.assertTrue(self.window.engine_buttons["bina"].isChecked())

    def test_batch_input_toggles_the_batch_widgets(self):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.assertFalse(self.window.folder_check.isVisibleTo(self.window))
        self.assertEqual(self.window.output_label.text(), "Output folder:")

        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.assertTrue(self.window.folder_check.isVisibleTo(self.window))
        self.assertEqual(self.window.output_label.text(), "Transcript:")

    def test_there_is_no_batch_layout_chooser(self):
        """Batch output is always one folder per PDF, so there is nothing to group."""
        self.assertFalse(hasattr(self.window, "layout_buttons"))
        self.assertFalse(hasattr(self.window, "layout_row"))

    def test_single_pdf_writes_markdown_and_finishes_clean(self):
        pdf = self._text_pdf()
        self._select_single_pdf(pdf, self.root / "single")

        self._start_and_wait()

        self.assertTrue((self.root / "single.md").is_file(), "no markdown was written")
        self.assertEqual(self.window.status_label.text(), "Done")
        self.assertEqual(self.window.progress.value(), self.window.progress.maximum())
        self.assertIn("Wrote single.md", self.window.log.toPlainText())
        # run control returns to the idle state
        self.assertTrue(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        self.assertFalse(self.window.timer.isActive())  # elapsed timer stopped
        self.assertFalse(self.window.running)

    def test_output_folder_option_nests_the_transcript(self):
        pdf = self._text_pdf()
        self._select_single_pdf(pdf, self.root / "book", in_folder=True)

        self._start_and_wait()

        self.assertTrue((self.root / "book" / "book.md").is_file())

    def test_save_next_to_input_is_off_by_default(self):
        self.assertFalse(self.window.same_dir_check.isChecked())

    def test_save_next_to_input_writes_beside_the_pdf(self):
        books = self.root / "books"
        books.mkdir()
        pdf = self._text_pdf("paper.pdf", books)
        # A bare name: without the option this would land in the CWD.
        self._select_single_pdf(pdf, "transcript")
        self.window.same_dir_check.setChecked(True)

        self._start_and_wait()

        self.assertTrue((books / "transcript.md").is_file())
        self.assertFalse((Path.cwd() / "transcript.md").exists())

    def test_save_next_to_input_creates_the_folder_beside_the_pdf(self):
        books = self.root / "books"
        books.mkdir()
        pdf = self._text_pdf("paper.pdf", books)
        self._select_single_pdf(pdf, "my_transcript", in_folder=True)
        self.window.same_dir_check.setChecked(True)

        self._start_and_wait()

        self.assertTrue((books / "my_transcript" / "my_transcript.md").is_file())

    def test_save_next_to_input_targets_the_image_folder_itself(self):
        pages = self.root / "pages"
        pages.mkdir()
        self.window.input_type_buttons["dir"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(pages))
        self.window.output_file.setText("pages_transcript")
        self.window.folder_check.setChecked(False)
        self.window.same_dir_check.setChecked(True)

        self.assertEqual(self.window._output_base(), pages / "pages_transcript")

        self.window.folder_check.setChecked(True)
        self.assertEqual(
            self.window._output_base(), pages / "pages_transcript" / "pages_transcript"
        )

    def test_save_next_to_input_is_hidden_for_batch(self):
        self.window.input_type_buttons["pdfs"].setChecked(True)
        self.window._input_type_changed()
        self.assertFalse(self.window.same_dir_check.isVisibleTo(self.window))

        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.assertTrue(self.window.same_dir_check.isVisibleTo(self.window))

    def test_batch_writes_one_folder_per_pdf(self):
        pdfs = [self._text_pdf("first.pdf"), self._text_pdf("second.pdf")]
        self._select_batch(pdfs, self.root / "out")

        self._start_and_wait()

        self.assertEqual(self.window.status_label.text(), "Done")
        self.assertTrue((self.root / "out" / "first" / "transcript.md").is_file())
        self.assertTrue((self.root / "out" / "second" / "transcript.md").is_file())
        self.assertIn("PDFs succeeded: 2/2", self.window.log.toPlainText())

    def test_no_pagemap_sidecar_is_left_behind(self):
        pdfs = [self._text_pdf("first.pdf"), self._text_pdf("second.pdf")]
        self._select_batch(pdfs, self.root / "out")

        self._start_and_wait()

        self.assertEqual(list((self.root / "out").rglob("*.pagemap.json")), [])

    def test_batch_keeps_going_when_one_document_fails(self):
        pdfs = [self._textless_pdf(), self._text_pdf()]
        self._select_batch(pdfs, self.root / "out")

        with patch.object(QMessageBox, "critical") as critical:
            self._start_and_wait()

        self.assertEqual(self.window.status_label.text(), "Done with errors")
        log = self.window.log.toPlainText()
        self.assertIn("[ERROR] scanned.pdf", log)
        self.assertIn("PDFs succeeded: 1/2", log)
        # A per-document failure is logged, not fatal.
        critical.assert_not_called()
        self.assertTrue((self.root / "out" / "paper" / "transcript.md").is_file())
        self.assertFalse((self.root / "out" / "scanned").exists())

    def test_start_without_input_warns_and_does_not_start(self):
        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)
        self.assertFalse(self.window.running)

    def test_start_without_formats_warns_and_does_not_start(self):
        self._select_single_pdf(self._text_pdf(), self.root / "single")
        for box in self.window.format_boxes.values():
            box.setChecked(False)

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)

    def test_inspector_with_image_folder_is_rejected(self):
        image_dir = self.root / "pages"
        image_dir.mkdir()
        self.window.input_type_buttons["dir"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(image_dir))
        self.window.engine_buttons["inspector"].setChecked(True)

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)

    def test_batch_refuses_to_start_with_no_pdfs_selected(self):
        self._select_batch([], self.root / "out")

        with patch.object(QMessageBox, "warning") as warning:
            self.window._start()

        warning.assert_called_once()
        self.assertIsNone(self.window.thread)

    # --- remembered settings ---

    def _restart(self):
        """A second window, as if the app had been closed and reopened."""
        window = gui.OCRApp()
        self.addCleanup(window.deleteLater)
        return window

    def test_first_launch_keeps_the_defaults(self):
        """With no settings file the form looks exactly as it always did."""
        first = self._restart()

        self.assertFalse(settings_store.settings_path().exists())
        self.assertEqual(first._input_type(), "pdf")
        self.assertTrue(first.folder_check.isChecked())
        self.assertFalse(first.same_dir_check.isChecked())
        self.assertEqual(first.output_file.text(), "book_transcript")
        self.assertEqual(first.dpi_box.currentText(), "300")
        self.assertEqual(first.max_tokens.value(), 1024)
        self.assertEqual(first.workers.value(), 1)
        self.assertEqual(first.page_limit.value(), 0)
        self.assertTrue(first.format_boxes["md"].isChecked())
        self.assertTrue(first.direction_buttons["rtl"].isChecked())
        self.assertTrue(first.normalize_check.isChecked())
        self.assertFalse(first.skip_ocr_check.isChecked())

    def test_a_run_remembers_the_form(self):
        pdf = self._text_pdf()
        self._select_single_pdf(pdf, self.root / "single")
        self.window.device_buttons["cpu"].setChecked(True)
        self.window.same_dir_check.setChecked(True)
        self.window.normalize_check.setChecked(False)
        self.window.max_tokens.setValue(2048)
        self.window.page_limit.setValue(5)
        self.window.dpi_box.setCurrentText("200")
        self.window.workers.setValue(3)

        self._start_and_wait()

        saved = settings_store.load()
        self.assertEqual(saved["engine"], "inspector")
        self.assertEqual(saved["input_path"], str(pdf))
        self.assertEqual(saved["formats"], ["md"])
        self.assertEqual(saved["device"], "cpu")
        self.assertTrue(saved["same_dir_check"])
        self.assertFalse(saved["normalize"])
        self.assertEqual(saved["max_tokens"], 2048)
        self.assertEqual(saved["page_limit"], 5)
        self.assertEqual(saved["dpi"], "200")
        self.assertEqual(saved["workers"], 3)

    def test_closing_the_window_remembers_the_form(self):
        self.window.engine_buttons["oneocr"].setChecked(True)
        self.window.skip_ocr_check.setChecked(True)

        self.window.close()

        saved = settings_store.load()
        self.assertEqual(saved["engine"], "oneocr")
        self.assertTrue(saved["skip_ocr"])

    def test_the_next_launch_restores_the_form(self):
        self.window.input_type_buttons["pdf"].setChecked(True)
        self.window._input_type_changed()
        self.window.input_path.setText(str(self.root / "paper.pdf"))
        self.window.output_file.setText("my_transcript")
        self.window.folder_check.setChecked(False)
        self.window.same_dir_check.setChecked(True)
        self.window.engine_buttons["oneocr"].setChecked(True)
        self.window.device_buttons["cpu"].setChecked(True)
        self.window.direction_buttons["ltr"].setChecked(True)
        self.window.normalize_check.setChecked(False)
        self.window.max_tokens.setValue(2048)
        self.window.workers.setValue(3)
        self.window.dpi_box.setCurrentText("400")
        for fmt, box in self.window.format_boxes.items():
            box.setChecked(fmt == "txt")
        self.window._save_settings()

        restored = self._restart()

        self.assertEqual(restored._input_type(), "pdf")
        self.assertEqual(restored.input_path.text(), str(self.root / "paper.pdf"))
        self.assertEqual(restored.output_file.text(), "my_transcript")
        self.assertFalse(restored.folder_check.isChecked())
        self.assertTrue(restored.same_dir_check.isChecked())
        self.assertTrue(restored.engine_buttons["oneocr"].isChecked())
        self.assertTrue(restored.device_buttons["cpu"].isChecked())
        self.assertTrue(restored.direction_buttons["ltr"].isChecked())
        self.assertFalse(restored.normalize_check.isChecked())
        self.assertEqual(restored.max_tokens.value(), 2048)
        self.assertEqual(restored.workers.value(), 3)
        self.assertEqual(restored.dpi_box.currentText(), "400")
        self.assertTrue(restored.format_boxes["txt"].isChecked())
        self.assertFalse(restored.format_boxes["md"].isChecked())
        # The remembered options still resolve to the same output path.
        self.assertEqual(
            restored._output_base(), self.root / "my_transcript"
        )

    def test_batch_inputs_are_remembered(self):
        pdf = self._text_pdf()
        settings_store.save({
            "input_type": "pdfs",
            "pdf_paths": [str(pdf)],
            "output_file": str(self.root / "out"),
        })

        restored = self._restart()

        self.assertEqual(restored._input_type(), "pdfs")
        self.assertEqual(restored.pdf_paths, [pdf])
        self.assertEqual(restored.input_path.text(), "1 PDF file selected")
        self.assertEqual(restored.output_file.text(), str(self.root / "out"))
        self.assertFalse(restored.same_dir_check.isVisibleTo(restored))

    def test_stale_batch_pdfs_are_dropped_on_restore(self):
        pdf = self._text_pdf()
        settings_store.save({
            "input_type": "pdfs",
            "pdf_paths": [str(self.root / "gone.pdf"), str(pdf)],
        })

        restored = self._restart()

        self.assertEqual(restored.pdf_paths, [pdf])
        self.assertEqual(restored.input_path.text(), "1 PDF file selected")
        self.assertIn("no longer exist", restored.log.toPlainText())

    def test_batch_restore_with_no_usable_pdfs_stays_empty(self):
        settings_store.save({
            "input_type": "pdfs",
            "pdf_paths": [str(self.root / "gone.pdf")],
        })

        restored = self._restart()

        self.assertEqual(restored.pdf_paths, [])
        self.assertEqual(restored.input_path.text(), "")

    def test_corrupt_settings_fall_back_to_the_defaults(self):
        path = settings_store.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")

        restored = self._restart()

        self.assertEqual(restored._input_type(), "pdf")
        self.assertTrue(restored.folder_check.isChecked())
        self.assertEqual(restored.dpi_box.currentText(), "300")
        self.assertEqual(restored.max_tokens.value(), 1024)

    def test_unknown_saved_values_are_ignored(self):
        settings_store.save({
            "input_type": "bogus",
            "engine": "nope",
            "device": "tpu",
            "dpi": "999",
            "workers": -5,
            "formats": ["docx"],
            "output_file": 42,
            "batch_layout": "by_type",  # written by older versions
        })

        restored = self._restart()

        self.assertEqual(restored._input_type(), "pdf")
        self.assertTrue(restored.engine_buttons["chrome"].isChecked())
        self.assertEqual(restored.dpi_box.currentText(), "300")
        self.assertEqual(restored.workers.value(), 1)  # clamped into range
        self.assertTrue(restored.format_boxes["md"].isChecked())
        self.assertEqual(restored.output_file.text(), "book_transcript")

    def test_a_remembered_input_that_vanished_fails_cleanly(self):
        """A restored input path can be stale; the run reports it, not crashes."""
        settings_store.save({
            "input_type": "pdf",
            "input_path": str(self.root / "gone.pdf"),
        })
        restored = self._restart()

        with patch.object(QMessageBox, "critical") as critical:
            self._start_and_wait(window=restored)

        self.assertEqual(restored.input_path.text(), str(self.root / "gone.pdf"))
        self.assertEqual(restored.status_label.text(), "Error")
        self.assertIn("PDF not found", restored.log.toPlainText())
        critical.assert_called_once()
        self.assertIn("PDF not found", critical.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
