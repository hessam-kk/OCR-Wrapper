import tempfile
import unittest
from pathlib import Path

from pdf_batch import (
    create_pdf_jobs,
    pdfs_in_folder,
    process_pdf_jobs,
    validate_pdf_paths,
)


class PDFBatchTests(unittest.TestCase):
    def test_pdfs_in_folder_is_sorted_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "zeta.PDF").touch()
            (folder / "alpha.pdf").touch()
            (folder / "notes.txt").touch()

            self.assertEqual(
                [path.name for path in pdfs_in_folder(folder)],
                ["alpha.pdf", "zeta.PDF"],
            )

    def test_validate_pdf_paths_deduplicates_without_reordering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            first = folder / "first.pdf"
            second = folder / "second.pdf"
            first.touch()
            second.touch()

            self.assertEqual(
                validate_pdf_paths([second, first, second]),
                [second, first],
            )

    def test_create_pdf_jobs_uses_transcript_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            pdf = folder / "paper.pdf"
            pdf.touch()

            [job] = create_pdf_jobs([pdf], folder / "output")

            self.assertEqual(job.input_path, pdf)
            self.assertEqual(job.output_base, folder / "output" / "paper_transcript")

    def test_create_pdf_jobs_can_put_each_result_in_its_own_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            pdf = folder / "paper.pdf"
            pdf.touch()

            [job] = create_pdf_jobs([pdf], folder / "output", separate_folders=True)

            self.assertEqual(
                job.output_base,
                folder / "output" / "paper_transcript" / "paper_transcript",
            )

    def test_duplicate_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            first_dir = folder / "first"
            second_dir = folder / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "paper.pdf"
            second = second_dir / "paper.pdf"
            first.touch()
            second.touch()

            with self.assertRaisesRegex(ValueError, "unique filenames"):
                create_pdf_jobs([first, second], folder / "output")

    def test_process_pdf_jobs_continues_after_a_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            first = folder / "first.pdf"
            second = folder / "second.pdf"
            first.touch()
            second.touch()
            jobs = create_pdf_jobs([first, second], folder / "output")
            processed = []

            def process(job, _index, _total):
                processed.append(job.input_path.name)
                if job.input_path == first:
                    raise RuntimeError("cannot read PDF")

            summary = process_pdf_jobs(jobs, process, log=lambda _message: None)

            self.assertEqual(processed, ["first.pdf", "second.pdf"])
            self.assertEqual(summary.completed, 2)
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failures, ((first, "cannot read PDF"),))


if __name__ == "__main__":
    unittest.main()
