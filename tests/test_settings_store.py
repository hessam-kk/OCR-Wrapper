"""Unit tests for the on-disk settings store (no Qt, no display needed)."""

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import settings_store


def without_override(**environment):
    """Environment with the override removed.

    Each keyword is set, or dropped from the environment when its value is
    None - which is how the platform fallbacks are exercised.
    """
    cleaned = dict(os.environ)
    cleaned.pop(settings_store.ENV_OVERRIDE, None)
    for key, value in environment.items():
        cleaned.pop(key, None)
        if value is not None:
            cleaned[key] = value
    return patch.dict(os.environ, cleaned, clear=True)


class SettingsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        # Every test runs against a throwaway config directory, never APPDATA.
        override = patch.dict(
            os.environ, {settings_store.ENV_OVERRIDE: str(self.root)}
        )
        override.start()
        self.addCleanup(override.stop)

    def _write_raw(self, text):
        path = settings_store.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    # --- where the file lives ---

    def test_config_dir_prefers_the_environment_override(self):
        self.assertEqual(settings_store.config_dir(), self.root)

    def test_settings_path_is_settings_json_inside_the_config_dir(self):
        self.assertEqual(
            settings_store.settings_path(), self.root / "settings.json"
        )

    def test_windows_uses_appdata(self):
        appdata = self.root / "Roaming"
        with without_override(APPDATA=str(appdata)):
            with patch.object(sys, "platform", "win32"):
                self.assertEqual(
                    settings_store.config_dir(), appdata / "OCR-Wrapper"
                )

    def test_windows_falls_back_when_appdata_is_unset(self):
        with without_override(APPDATA=None):
            with patch.object(sys, "platform", "win32"):
                self.assertEqual(
                    settings_store.config_dir(),
                    Path.home() / "AppData" / "Roaming" / "OCR-Wrapper",
                )

    def test_macos_uses_application_support(self):
        with without_override():
            with patch.object(sys, "platform", "darwin"):
                self.assertEqual(
                    settings_store.config_dir(),
                    Path.home() / "Library" / "Application Support" / "OCR-Wrapper",
                )

    def test_linux_uses_xdg_config_home(self):
        xdg = self.root / "config"
        with without_override(XDG_CONFIG_HOME=str(xdg)):
            with patch.object(sys, "platform", "linux"):
                self.assertEqual(
                    settings_store.config_dir(), xdg / "OCR-Wrapper"
                )

    def test_linux_falls_back_to_dot_config(self):
        with without_override(XDG_CONFIG_HOME=None):
            with patch.object(sys, "platform", "linux"):
                self.assertEqual(
                    settings_store.config_dir(),
                    Path.home() / ".config" / "OCR-Wrapper",
                )

    # --- reading and writing ---

    def test_load_returns_empty_dict_before_anything_is_saved(self):
        self.assertEqual(settings_store.load(), {})

    def test_save_then_load_round_trips_the_values(self):
        values = {"engine": "bina", "formats": ["md", "txt"], "workers": 4}
        path = settings_store.save(values)
        self.assertEqual(path, self.root / "settings.json")
        self.assertEqual(settings_store.load(), values)

    def test_save_creates_missing_folders(self):
        nested = self.root / "deep" / "nested"
        with patch.dict(os.environ, {settings_store.ENV_OVERRIDE: str(nested)}):
            settings_store.save({"engine": "chrome"})
            self.assertTrue((nested / "settings.json").is_file())

    def test_save_overwrites_the_previous_values(self):
        settings_store.save({"engine": "chrome", "workers": 1})
        settings_store.save({"engine": "oneocr"})
        self.assertEqual(settings_store.load(), {"engine": "oneocr"})

    def test_save_leaves_no_temporary_file_behind(self):
        settings_store.save({"engine": "chrome"})
        self.assertEqual([p.name for p in self.root.iterdir()], ["settings.json"])

    def test_saved_file_is_readable_json(self):
        settings_store.save({"engine": "chrome"})
        text = settings_store.settings_path().read_text(encoding="utf-8")
        self.assertEqual(json.loads(text), {"engine": "chrome"})

    def test_unicode_paths_survive_the_round_trip(self):
        values = {"input_path": "C:\\کتاب\\section ۱\\paper.pdf"}
        settings_store.save(values)
        self.assertEqual(settings_store.load()["input_path"], values["input_path"])

    # --- bad files are never fatal ---

    def test_malformed_json_is_ignored(self):
        self._write_raw("{ not json")
        self.assertEqual(settings_store.load(), {})

    def test_non_object_json_is_ignored(self):
        self._write_raw("[1, 2, 3]")
        self.assertEqual(settings_store.load(), {})

    def test_unreadable_file_is_ignored(self):
        settings_store.settings_path().mkdir()  # a directory, not a file
        self.assertEqual(settings_store.load(), {})

    def test_settings_can_be_saved_again_after_a_corrupt_file(self):
        self._write_raw("{ not json")
        settings_store.save({"engine": "oneocr"})
        self.assertEqual(settings_store.load(), {"engine": "oneocr"})


if __name__ == "__main__":
    unittest.main()
