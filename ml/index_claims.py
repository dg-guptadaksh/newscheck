import argparse
import hashlib
import json
from pathlib import Path

from app.retrieval import MiniLM, claim_fingerprint


def build(folder, claims_file):
    import faiss
    rows = json.loads(claims_file.read_text(encoding="utf-8"))
    unique = {}
    for row in rows:
        unique.setdefault((row["claim"], row["publisher"], row["url"]), row)
    rows = list(unique.values())
    if not rows:
        raise ValueError("Provide at least one fact-check")
    encoder = MiniLM(folder)
    embeddings = encoder.encode([claim["claim"] for claim in rows])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(folder / "claims.faiss"))
    files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
             for name in ("model.onnx", "tokenizer.json", "claims.faiss")}
    (folder / "index.json").write_text(json.dumps({"embedding": "all-MiniLM-L6-v2",
         "claims": len(rows), "claims_sha256": claim_fingerprint(rows), "files": files}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("folder", type=Path)
    cli.add_argument("claims", type=Path)
    args = cli.parse_args()
    build(args.folder, args.claims)
