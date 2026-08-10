"""Persian text post-processing (hazm normalization)."""


def get_normalizer():
    try:
        from hazm import Normalizer
    except ImportError:
        raise RuntimeError(
            "hazm is not installed. Run: pip install hazm"
        ) from None
    return Normalizer()


def normalize_transcribe(transcribe, normalizer):
    """Wrap a transcribe callable to normalize its output text."""
    def normalized(page_path):
        text = transcribe(page_path)
        if text.startswith("[ERROR"):
            return text
        return normalizer.normalize(text)
    return normalized
