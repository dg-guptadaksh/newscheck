import re
from functools import lru_cache
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.text import URL

PATTERN = re.compile("[a-z]+")


def lexical_tokens(text):
    return [word for word in PATTERN.findall(URL.sub("", text.lower()))
            if word not in ENGLISH_STOP_WORDS]


def stemmed_tokens(text):
    return [stem(word) for word in lexical_tokens(text)]


@lru_cache(maxsize=100_000)
def stem(word):
    return stemmer().stem(word)


@lru_cache(maxsize=1)
def stemmer():
    from nltk.stem.snowball import EnglishStemmer
    return EnglishStemmer()


def vectorizer(stemming=True):
    return TfidfVectorizer(tokenizer=stemmed_tokens if stemming else lexical_tokens, token_pattern=None,
                           ngram_range=(1, 2), max_features=50000, sublinear_tf=True, dtype=np.float64)
