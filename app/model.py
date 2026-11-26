import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.text import LEXEMES, normalize
from ml.calibration import probability


@dataclass(frozen=True)
class Scored:
    text: str
    used: int
    total: int
    fake_probability: float
    token_ids: tuple[int, ...] = ()
    tokens: tuple[str, ...] = ()


def lexical_window(text, limit):
    positions = list(LEXEMES.finditer(text))
    if len(positions) > limit:
        text = text[:positions[limit - 1].end()]
    return text, min(len(positions), limit), len(positions)


class BaselineModel:
    def __init__(self, folder):
        self.meta = json.loads((folder / "baseline.json").read_text())
        self.vectorizer, self.model = joblib.load(folder / "baseline.joblib")
        self.features = self.vectorizer.get_feature_names_out()

    def score(self, text):
        text, used, total = lexical_window(text, self.meta["max_tokens"])
        margin = self.model.decision_function(self.vectorizer.transform([text]))[0]
        return Scored(text, used, total, float(probability(margin, self.meta["temperature"])))

    def explain(self, result):
        vector = self.vectorizer.transform([result.text])
        weights = vector.data * self.model.coef_[0, vector.indices]
        items = sorted(zip(vector.indices, weights), key=lambda item: abs(item[1]), reverse=True)
        return [{"token": self.features[index], "weight": float(value)}
                for index, value in items[:12] if value]


class OnnxModel:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.meta = json.loads((folder / "manifest.json").read_text())
        if self.meta["max_tokens"] != 256 or self.meta["temperature"] <= 0:
            raise ValueError("Invalid model manifest")
        for name in ("tokenizer.json", "model.int8.onnx"):
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != self.meta["files"][name]:
                raise ValueError(f"Encoder artifact checksum differs for {name}")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.mask = self.tokenizer.token_to_id("<mask>")
        if self.mask is None:
            raise ValueError("Token attribution requires a mask token")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(folder / "model.int8.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}

    def logits(self, ids, mask=None):
        inputs = {"input_ids": np.asarray(ids, dtype=np.int64)}
        inputs["attention_mask"] = np.ones_like(inputs["input_ids"]) if mask is None else mask
        return self.session.run(None, {k: v for k, v in inputs.items() if k in self.inputs})[0]

    def encode(self, text):
        full = self.tokenizer.encode(normalize(text))
        limit = self.meta["max_tokens"]
        if len(full.ids) <= limit:
            return full.ids, full.tokens, text, len(full.ids)
        # Keep the final special token, exactly as truncation with AutoTokenizer does.
        end = full.offsets[limit - 2][1]
        return full.ids[:limit - 1] + full.ids[-1:], full.tokens[:limit - 1] + full.tokens[-1:], text[:end], len(full.ids)

    def probabilities(self, ids):
        logits = self.logits(ids)
        return probability(logits[:, 1] - logits[:, 0], self.meta["temperature"])

    def score(self, text):
        ids, tokens, used_text, total = self.encode(text)
        return Scored(used_text, len(ids), total, float(self.probabilities([ids])[0]), tuple(ids), tuple(tokens))

    def explain(self, result):
        import shap

        ids, tokens = result.token_ids, result.tokens
        features = np.asarray(ids[1:-1])
        if not len(features):
            return []

        def mask_tokens(mask, values):
            return np.where(mask, values, self.mask).reshape(1, -1)

        def predict(values):
            batch = [[ids[0], *row, ids[-1]] for row in values]
            return self.probabilities(batch)

        explanation = shap.Explainer(predict, mask_tokens, algorithm="permutation", seed=42)(
            features[None, :], max_evals=2 * len(features) + 1, batch_size=16, silent=True
        )
        values = explanation.values[0]
        indices = np.argsort(-np.abs(values))[:12]
        return [{"token": re.sub(r"^[ĠĊ]", "", tokens[i + 1]), "weight": float(values[i])}
                for i in indices if values[i]]


def load_model(bundle, baseline_folder: Path):
    return OnnxModel(bundle) if bundle else BaselineModel(baseline_folder)
