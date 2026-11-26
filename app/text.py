import re
import unicodedata

WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)
LEXEMES = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
BOILERPLATE = re.compile(r"(?im)^\s*(?:advertisement|read more|subscribe to our newsletter|all rights reserved)[.!]?\s*$")
LIMIT = 256
MIN_WORDS = 20


def normalize(text):
    return " ".join(BOILERPLATE.sub("", unicodedata.normalize("NFC", text)).split())


def combine(headline, body):
    return normalize(f"{headline}\n{body}")


def validate(text):
    words = [word for word in WORDS.findall(URL.sub("", text)) if word[0].isalpha()]
    if not words:
        raise ValueError("Enter a news headline or article containing words.")
    if len(words) < MIN_WORDS:
        raise ValueError(f"Enter at least {MIN_WORDS} words across the headline and article.")
    if not any("a" <= char.lower() <= "z" for char in text):
        raise ValueError("Enter English-language news text.")
