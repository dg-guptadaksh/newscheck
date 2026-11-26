import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from app.text import WORDS, combine, normalize

SEED = 42
LIAR = {"pants-fire": 1, "false": 1, "barely-true": 1, "half-true": 0, "mostly-true": 0, "true": 0}


def shingles(text):
    words = WORDS.findall(text.casefold())[:160]
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def near_duplicates(rows):
    # SimHash narrows candidates. Jaccard confirmation avoids discarding two
    # unrelated articles because their short fingerprints happen to collide.
    buckets = defaultdict(list)
    seen = set()
    for text, label in rows:
        words = WORDS.findall(text.casefold())
        if len(words) < 20:
            continue
        exact = hashlib.blake2b(" ".join(words).encode(), digest_size=16).digest()
        if exact in seen:
            continue
        seen.add(exact)
        terms = shingles(text)
        hashes = b"".join(hashlib.blake2b(word.encode(), digest_size=8).digest() for word in terms)
        bits = np.unpackbits(np.frombuffer(hashes, dtype=np.uint8)).reshape(-1, 64)
        signature = int.from_bytes(np.packbits(bits.sum(axis=0) * 2 > len(terms)).tobytes(), "big")
        keys = [(block, (signature >> (16 * block)) & 65535) for block in range(4)]
        candidates = {prior for key in keys for prior_signature, prior in buckets[key]
                      if (prior_signature ^ signature).bit_count() <= 3}
        duplicate = False
        for prior in candidates:
            other = shingles(prior)
            if len(terms & other) / len(terms | other) >= .80:
                duplicate = True
                break
        if duplicate:
            continue
        for key in keys:
            buckets[key].append((signature, text))
        yield text, label


def welfake(path, fake_value=1):
    csv.field_size_limit(2_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        return [(combine(row.get("title") or "", row.get("text") or ""), int(int(row["label"]) == fake_value))
                for row in rows if row.get("label") in {"0", "1"}]


def liar(path):
    with Path(path).open(encoding="utf-8", newline="") as source:
        return [(normalize(row[2]), LIAR[row[1]]) for row in csv.reader(source, delimiter="\t", quoting=csv.QUOTE_NONE)
                if row and row[1] in LIAR]


def covid(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return [(normalize(row["tweet"]), {"real": 0, "fake": 1}[row["label"].strip().lower()])
                for row in csv.DictReader(source)]


def split(rows):
    labels = [label for _, label in rows]
    train, rest = train_test_split(np.arange(len(rows)), test_size=.30, stratify=labels, random_state=SEED)
    validation, test = train_test_split(rest, test_size=.5, stratify=np.asarray(labels)[rest], random_state=SEED)
    return tuple(([rows[i][0] for i in subset], np.asarray(labels)[subset]) for subset in (train, validation, test))
