"""Import English reviews from the organized claimreview-data release archive."""
import argparse
import io
import json
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

PUBLISHERS = {"politifact.com": "PolitiFact", "factcheck.org": "FactCheck.org",
              "snopes.com": "Snopes", "fullfact.org": "Full Fact", "afp.com": "AFP Fact Check",
              "apnews.com": "AP Fact Check", "reuters.com": "Reuters Fact Check"}


def records(stream):
    """Stream the release's array without reading the other languages into memory."""
    decoder = json.JSONDecoder()
    buffer = stream.read(65536).lstrip()
    if not buffer.startswith("["):
        raise ValueError("The archive member must contain a JSON array")
    buffer = buffer[1:]
    while True:
        buffer = buffer.lstrip(" \r\n\t,")
        if buffer.startswith("]"):
            return
        try:
            row, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            extra = stream.read(65536)
            if not extra:
                raise ValueError("Incomplete ClaimReview archive member") from None
            buffer += extra
            continue
        yield row
        buffer = buffer[end:]


def archive_claims(archive, seed, limit):
    rows = list(seed)
    seen = {(row["claim"].casefold(), row["url"]) for row in rows}
    with ZipFile(archive) as zipped:
        member, = [name for name in zipped.namelist() if name.endswith("/claim_reviews.json")]
        with io.TextIOWrapper(zipped.open(member), encoding="utf-8") as stream:
            for item in records(stream):
                if len(rows) >= limit:
                    break
                checker = item.get("fact_checker") or {}
                if checker.get("language") != "English":
                    continue
                url = item.get("review_url") or ""
                parts = urlsplit(url)
                if parts.scheme not in {"http", "https"} or not parts.hostname:
                    continue
                claim = next((text.strip() for text in item.get("claim_text", [])
                              if isinstance(text, str) and 12 <= len(text.strip()) <= 500), "")
                if not claim or (claim.casefold(), url) in seen:
                    continue
                reviews = item.get("reviews") or []
                rating = next((review.get("original_label") for review in reviews if
                               isinstance(review.get("original_label"), str) and
                               0 < len(review["original_label"]) <= 100), "")
                domain = checker.get("domain") or parts.hostname
                publisher = PUBLISHERS.get(domain, domain)
                seen.add((claim.casefold(), url))
                rows.append({"claim": claim, "publisher": publisher, "rating": rating, "url": url})
    return rows


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--seed", type=Path)
    cli.add_argument("--limit", type=int, default=23180)
    args = cli.parse_args()
    if args.limit < 1:
        cli.error("Use a positive limit")
    seed = json.loads(args.seed.read_text(encoding="utf-8")) if args.seed else []
    claims = archive_claims(args.archive, seed[:args.limit], args.limit)
    args.destination.write_text(json.dumps(claims, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(claims)} fact-check records")
