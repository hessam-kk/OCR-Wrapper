"""Chrome Screen AI OCR engine via the chrome-ocr wrapper."""

from pathlib import Path

from ocr import _register_engine_builder


def _chrome_builder(page_path):
    """Build a chrome transcribe closure inside a worker process."""
    engine = get_screenai_engine()
    return lambda p: chrome_transcribe_page(engine, p)


# Registered at import so spawned worker processes (Windows spawn re-imports
# modules) have the builder available for process-parallel runs.
_register_engine_builder("chrome", _chrome_builder)


def _chrome_builder(page_path):
    """Build a chrome transcribe closure inside a worker process."""
    engine = get_screenai_engine()
    return lambda p: chrome_transcribe_page(engine, p)


def get_screenai_engine():
    """Return a chrome_ocr.ScreenAIEngine, or raise a friendly error."""
    try:
        from chrome_ocr import ScreenAIEngine
    except ImportError:
        raise RuntimeError(
            "chrome-ocr is not installed. Run: "
            "git clone https://github.com/ayismas/chrome-ocr && cd chrome-ocr && pip install -e \".[pdf]\""
        ) from None
    try:
        return ScreenAIEngine()
    except Exception as e:
        raise RuntimeError(
            "Chrome Screen AI not found. Enable a screen-reader option in "
            "Chrome Settings -> Accessibility so chrome_screen_ai.dll downloads "
            "into %LOCALAPPDATA%\\Google\\Chrome\\User Data\\screen_ai\\"
        ) from e


def chrome_transcribe_page(engine, image_path: Path) -> str:
    from chrome_ocr import ocr_img

    return ocr_img(str(image_path), engine=engine).strip()
