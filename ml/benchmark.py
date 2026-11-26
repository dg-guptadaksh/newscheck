"""Measure warm CPU classification, evidence retrieval and attribution separately."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from app.model import OnnxModel
from app.retrieval import SemanticRetriever


def timed(operation, repetitions):
    times = []
    for _ in range(repetitions):
        start = perf_counter()
        operation()
        times.append((perf_counter() - start) * 1000)
    return round(median(times), 2)


def retrieval_metrics(retriever, pairs):
    if not pairs:
        raise ValueError("Provide at least one matching article and fact-check URL")
    ranks = []
    for row in pairs:
        hits = retriever.search(row["text"], limit=10, minimum=-1)
        rank = next((i for i, hit in enumerate(hits, 1) if hit["url"] == row["url"]), None)
        ranks.append(rank)
    count = len(ranks)
    return {"samples": count, "recall_at_5": sum(1 for rank in ranks if rank and rank <= 5) / count,
            "recall_at_10": sum(1 for rank in ranks if rank) / count,
            "mrr_at_10": sum(1 / rank for rank in ranks if rank) / count}


def measure(encoder, retrieval, claims_file, repetitions, explanations, pairs=None):
    model = OnnxModel(encoder)
    retriever = SemanticRetriever(retrieval, json.loads(claims_file.read_text(encoding="utf-8")))
    paragraph = ("A local council released its annual transport report after a public session. "
                 "The published records include traffic counts, planned repairs and a timetable for further review. ")
    text = paragraph * 20
    scored = model.score(text)
    retriever.search(text)
    results = {"scored_tokens": scored.used, "inference_median_ms": timed(lambda: model.score(text), repetitions),
               "retrieval_median_ms": timed(lambda: retriever.search(text), repetitions),
               "onnx_bytes": (encoder / "model.int8.onnx").stat().st_size,
               "classification_repetitions": repetitions}
    if explanations:
        model.explain(model.score(paragraph))
        results["shap_median_ms"] = timed(lambda: model.explain(scored), explanations)
        results["shap_repetitions"] = explanations
    if pairs is not None:
        results["retrieval_quality"] = retrieval_metrics(retriever, pairs)
    return results


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--encoder", type=Path, default=Path("artifacts/encoder"))
    cli.add_argument("--retrieval", type=Path, default=Path("artifacts/retrieval"))
    cli.add_argument("--claims", type=Path, default=Path("data/claims.json"))
    cli.add_argument("--runs", type=int, default=30)
    cli.add_argument("--shap-runs", type=int, default=0)
    cli.add_argument("--pairs", type=Path, help="JSON array of objects containing text and matching url")
    args = cli.parse_args()
    if args.runs < 1 or args.shap_runs < 0:
        cli.error("Use a positive run count and a nonnegative SHAP run count")
    pairs = json.loads(args.pairs.read_text(encoding="utf-8")) if args.pairs else None
    print(json.dumps(measure(args.encoder, args.retrieval, args.claims, args.runs, args.shap_runs, pairs), indent=2))
