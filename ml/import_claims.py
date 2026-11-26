"""Normalize a ClaimReview JSON export or Data Commons JSON-LD sample."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlsplit


def _entries(value):
    if isinstance(value, list):
        for item in value:
            yield from _entries(item)
    elif isinstance(value, dict):
        if "claimReviewed" in value:
            yield value
        else:
            for item in value.values():
                yield from _entries(item)


def import_claims(source, limit):
    raw = source.read_text(encoding="utf-8")
    if raw.lstrip().startswith("<script"):
        docs = [json.loads(text) for text in re.findall(r"<script[^>]*>(.*?)</script>", raw, re.S)]
    else:
        try:
            docs = json.loads(raw)
        except json.JSONDecodeError:
            docs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    seen = set()
    output = []
    for row in _entries(docs):
        language = row.get("inLanguage", "en")
        if isinstance(language, str) and not language.casefold().startswith("en"):
            continue
        author = row.get("author") or {}
        rating = row.get("reviewRating") or {}
        if isinstance(author, list):
            author = author[0] if author else {}
        claim, url = row["claimReviewed"], row.get("url")
        publisher = author.get("name") if isinstance(author, dict) else author
        if not all(isinstance(value, str) and value.strip() for value in (claim, url, publisher)):
            continue
        if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).netloc:
            continue
        record = (claim, publisher, url)
        if record in seen:
            continue
        seen.add(record)
        label = rating.get("alternateName", "") if isinstance(rating, dict) else ""
        output.append({"claim": claim, "publisher": publisher, "url": url, "rating": str(label or "")})
        if len(output) == limit:
            break
    if not output:
        raise ValueError("No valid English ClaimReview records found")
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--limit", type=int, default=23180)
    args = cli.parse_args()
    if args.limit < 1:
        cli.error("Use a positive limit")
    claims = import_claims(args.source, args.limit)
    args.destination.write_text(json.dumps(claims, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(claims)} claims")
