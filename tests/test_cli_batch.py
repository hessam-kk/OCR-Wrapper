import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from book_ocr_batch import _run_pdf_batch


class CLIBatchTests(unittest.TestCase):
    def test_inspector_batch_processes_every_pdf_with_matching_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            first = folder / "first.pdf"
            second = folder / "second.pdf"
            first.touch()
            second.touch()
            args = SimpleNamespace(
                output_dir=str(folder / "output"),
                save_in_folders=False,
                skip_ocr=False,
                engine="inspector",
                formats=["md"],
                direction="ltr",
            )

            with (
                patch("book_ocr_batch.write_inspector_transcript") as write,
                patch("builtins.print"),
            ):
                exit_code = _run_pdf_batch(args, [first, second])

            self.assertEqual(exit_code, 0)
            self.assertEqual(write.call_count, 2)
            self.assertEqual(write.call_args_list[0].args[0], first)
            self.assertEqual(
                write.call_args_list[0].args[1],
                folder / "output" / "first_transcript",
            )
            self.assertEqual(write.call_args_list[1].args[0], second)
            self.assertEqual(
                write.call_args_list[1].args[1],
                folder / "output" / "second_transcript",
            )

    def test_batch_returns_failure_when_one_pdf_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            first = folder / "first.pdf"
            second = folder / "second.pdf"
            first.touch()
            second.touch()
            args = SimpleNamespace(
                output_dir=str(folder / "output"),
                save_in_folders=False,
                skip_ocr=False,
                engine="inspector",
                formats=["md"],
                direction="ltr",
            )

            with (
                patch(
                    "book_ocr_batch.write_inspector_transcript",
                    side_effect=[RuntimeError("bad PDF"), None],
                ) as write,
                patch("builtins.print"),
            ):
                exit_code = _run_pdf_batch(args, [first, second])

            self.assertEqual(exit_code, 1)
            self.assertEqual(write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
