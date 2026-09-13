"""CLI tests for --same_dir (write outputs next to the input)."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import fitz

    import main
except ImportError as error:  # pragma: no cover - depends on the environment
    main = None
    SKIP_REASON = f"CLI test dependencies unavailable: {error}"
else:
    SKIP_REASON = ""


@unittest.skipIf(main is None, SKIP_REASON)
class SameDirCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _text_pdf(self, name="paper.pdf"):
        """A small PDF with a real text layer (pdf-inspector rejects empty ones)."""
        path = self.root / name
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "OCR wrapper test page.")
        doc.save(path)
        doc.close()
        return path

    def _run_cli(self, *args):
        argv = ["main.py", *args]
        # The pipeline logs to stdout; keep the test output readable.
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return main.main()

    def test_same_dir_writes_beside_the_pdf(self):
        pdf = self._text_pdf()

        code = self._run_cli(
            "--pdf", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_file", "transcript", "--same_dir",
        )

        self.assertEqual(code, 0)
        self.assertTrue((self.root / "transcript.md").is_file())
        # Without the flag a bare --output_file would land in the working dir.
        self.assertFalse((Path.cwd() / "transcript.md").exists())

    def test_same_dir_leaves_no_pagemap_sidecar(self):
        pdf = self._text_pdf()

        self._run_cli(
            "--pdf", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_file", "transcript", "--same_dir",
        )

        self.assertTrue((self.root / "transcript.md").is_file())
        self.assertEqual(list(self.root.glob("*.pagemap.json")), [])

    def test_a_stale_pagemap_sidecar_is_cleaned_up(self):
        pdf = self._text_pdf()
        stale = self.root / "transcript.pagemap.json"
        stale.write_text("[0, 2]", encoding="utf-8")

        self._run_cli(
            "--pdf", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_file", "transcript", "--same_dir",
        )

        self.assertTrue((self.root / "transcript.md").is_file())
        self.assertFalse(stale.exists())

    def test_a_pdf_batch_leaves_no_pagemap_sidecar(self):
        pdf = self._text_pdf()

        code = self._run_cli(
            "--pdfs", str(pdf), "--engine", "inspector", "--direction", "ltr",
            "--output_dir", str(self.root / "out"),
        )

        self.assertEqual(code, 0)
        self.assertTrue((self.root / "out" / "paper" / "transcript.md").is_file())
        self.assertEqual(list((self.root / "out").rglob("*.pagemap.json")), [])

    def test_the_batch_layout_flag_is_gone(self):
        pdf = self._text_pdf()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                self._run_cli(
                    "--pdfs", str(pdf), "--batch_layout", "by_type",
                )

        self.assertEqual(raised.exception.code, 2)

    def test_same_dir_is_rejected_for_multiple_pdfs(self):
        pdf = self._text_pdf()

        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                self._run_cli("--pdfs", str(pdf), "--same_dir")

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--same_dir is not supported with --pdfs", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
