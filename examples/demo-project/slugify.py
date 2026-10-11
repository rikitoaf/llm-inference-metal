"""Turn arbitrary titles into URL slugs."""
import re
import unicodedata


def slugify(text: str, max_len: int = 40) -> str:
    """Lower-case, strip accents, join words with single hyphens, cap at max_len."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "-", text)
    return text[:max_len]
