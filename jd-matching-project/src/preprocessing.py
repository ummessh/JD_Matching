import re
import string

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove punctuation, numbers, and extra spaces.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

