import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from app.model import lexical_window
from ml.baselines import vectorizer
from ml.calibration import fit_temperature, probability
from ml.corpora import SEED, covid, liar, near_duplicates, split, welfake
from ml.metrics import evaluate


def run(source, output, liar_path=None, covid_path=None, stemming=False):
    raw = welfake(source)
    rows = list(near_duplicates(raw))
    portions = split(rows)
    prepared = [([lexical_window(text, 256)[0] for text in texts], labels)
                for texts, labels in portions]
    (train, y_train), (validation, y_validation), (test, y_test) = prepared
    domains = {"WELFake": (test, y_test)}
    for name, loader, path in (("LIAR", liar, liar_path), ("Constraint COVID-19", covid, covid_path)):
        if path:
            rows = loader(path)
            domains[name] = ([lexical_window(t, 256)[0] for t, _ in rows], [label for _, label in rows])
    tfidf = vectorizer(stemming)
    design = tfidf.fit_transform(train)
    validation_design = tfidf.transform(validation)
    results = {}
    for name, model in (("TF-IDF + Logistic Regression", LogisticRegression(random_state=SEED, max_iter=1000)),
                        ("TF-IDF + Linear SVM", LinearSVC(random_state=SEED))):
        model.fit(design, y_train)
        margin = model.decision_function(validation_design)
        temperature = fit_temperature(margin, y_validation)
        measurements = {}
        for domain, (texts, labels) in domains.items():
            margin = model.decision_function(tfidf.transform(texts))
            measurements[domain] = evaluate(labels, probability(margin, temperature))
            measurements[domain]["calibration_before"] = evaluate(labels, probability(margin, 1))["calibration"]
        results[name] = {"temperature": temperature, "domains": measurements}
        if name.endswith("Regression"):
            output.mkdir(parents=True, exist_ok=True)
            joblib.dump((tfidf, model), output / "baseline.joblib", compress=3)
            (output / "baseline.json").write_text(json.dumps({"name": name, "version": "welfake-tfidf-v1",
                "mode": "baseline", "temperature": temperature, "max_tokens": 256,
                "token_unit": "lexical tokens", "attribution": "Weighted TF-IDF terms"}, indent=2))
    summary = {"seed": SEED, "source_rows": len(raw), "raw_label_counts": dict(Counter(y for _, y in raw)),
               "split_sizes": [len(y) for _, y in portions], "stemming": stemming, "models": results}
    (output / "baseline-evaluation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("welfake", type=Path)
    parser.add_argument("--liar", type=Path)
    parser.add_argument("--covid", type=Path)
    parser.add_argument("--stemming", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run(args.welfake, args.output, args.liar, args.covid, args.stemming)
