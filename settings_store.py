"""Remember the GUI's last-used settings between runs.

The file lives in the per-user config directory:

    Windows   %APPDATA%\\OCR-Wrapper\\settings.json
    macOS     ~/Library/Application Support/OCR-Wrapper/settings.json
    Linux     $XDG_CONFIG_HOME/OCR-Wrapper/settings.json  (~/.config/...)

``OCR_WRAPPER_CONFIG_DIR`` overrides that location, which is how the tests keep
a run from reading or overwriting the real user's settings.

A corrupt or missing file is never fatal: the caller gets an empty mapping and
the GUI falls back to its defaults.
"""

import json
import os
import sys
from pathlib import Path

APP_NAME = "OCR-Wrapper"
SETTINGS_FILENAME = "settings.json"
ENV_OVERRIDE = "OCR_WRAPPER_CONFIG_DIR"


def config_dir() -> Path:
    """Directory that holds this app's per-user configuration."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(root) / APP_NAME


def settings_path() -> Path:
    return config_dir() / SETTINGS_FILENAME


def load() -> dict:
    """Saved settings, or an empty dict when there is nothing usable."""
    try:
        with open(settings_path(), encoding="utf-8") as handle:
            values = json.load(handle)
    except (OSError, ValueError):
        return {}
    return values if isinstance(values, dict) else {}


def save(values: dict) -> Path:
    """Write `values` to the settings file, creating its folder if needed."""
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target first: an interrupted save then cannot leave a
    # half-written file that the next launch refuses to parse.
    temporary = path.parent / f"{path.name}.tmp"
    temporary.write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)
    return path
