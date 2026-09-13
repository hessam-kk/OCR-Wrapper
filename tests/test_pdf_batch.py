import tempfile
import unittest
from pathlib import Path

from pdf_batch import (
    MAX_OUTPUT_PATH,
    beside_input,
    create_jobs,
    validate_pdfs,
)


class BesideInputTests(unittest.TestCase):
    def test_pdf_transcript_lands_in_the_pdf_directory(self):
        self.assertEqual(
            beside_input(Path("/books/out/transcript"), Path("/books/paper.pdf")),
            Path("/books/transcript"),
        )

    def test_image_folder_transcript_lands_inside_that_folder(self):
        self.assertEqual(
            beside_input(
                Path("/books/transcript"), Path("/scans/pages"), source_is_dir=True
            ),
            Path("/scans/pages/transcript"),
        )

    def test_an_absolute_output_path_keeps_only_its_name(self):
        self.assertEqual(
            beside_input(Path("/elsewhere/a/b/transcript"), Path("/books/paper.pdf")),
            Path("/books/transcript"),
        )


class PDFBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _pdf(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_each_pdf_gets_its_own_folder_with_a_fixed_name(self):
        pdf = self._pdf("paper.pdf")

        [job] = create_jobs([pdf], self.root / "output")

        self.assertEqual(job.input_path, pdf)
        self.assertEqual(job.output_base.name, "transcript")
        self.assertEqual(
            job.output_base.parent, (self.root / "output" / "paper").resolve()
        )

    def test_a_long_name_is_shortened_with_a_hash(self):
        pdf = self._pdf("long-paper-title-" * 12 + ".pdf")

        [job] = create_jobs([pdf], self.root / "output")

        folder = job.output_base.parent.name
        self.assertLess(len(folder), len(pdf.stem))
        self.assertRegex(folder, r"-[0-9a-f]{8}$")

    def test_generated_paths_fit_the_windows_limit(self):
        pdf = self._pdf("long-paper-title-" * 12 + ".pdf")

        [job] = create_jobs([pdf], self.root / "output")

        longest = job.output_base.with_suffix(".ebook.md")
        self.assertLessEqual(len(str(longest)), MAX_OUTPUT_PATH)

    def test_same_stem_in_different_folders_gets_distinct_folders(self):
        first = self._pdf("one/paper.pdf")
        second = self._pdf("two/paper.pdf")

        jobs = create_jobs([first, second], self.root / "output")

        self.assertEqual(len(jobs), 2)
        self.assertEqual(len({job.output_base for job in jobs}), 2)

    def test_validate_pdfs_drops_duplicates_and_keeps_order(self):
        first = self._pdf("first.pdf")
        second = self._pdf("second.pdf")

        self.assertEqual(
            validate_pdfs([second, first, Path(second)]), [second, first]
        )

    def test_validate_pdfs_rejects_a_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            validate_pdfs([self.root / "gone.pdf"])

    def test_validate_pdfs_rejects_a_non_pdf(self):
        other = self._pdf("notes.txt")

        with self.assertRaises(ValueError):
            validate_pdfs([other])

    def test_validate_pdfs_rejects_an_empty_selection(self):
        with self.assertRaises(ValueError):
            validate_pdfs([])


if __name__ == "__main__":
    unittest.main()
