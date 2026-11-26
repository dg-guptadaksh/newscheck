import hashlib
import json
import re

import numpy as np


def claim_fingerprint(claims):
    fields = ("claim", "publisher", "rating", "url")
    data = [[row[field] for field in fields] for row in claims]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def leading_claim(text):
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


class MiniLM:
    def __init__(self, folder):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(folder / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts):
        chunks = []
        for start in range(0, len(texts), 32):
            tokens = self.tokenizer.encode_batch(texts[start:start + 32])
            mask = np.asarray([item.attention_mask for item in tokens], dtype=np.int64)
            inputs = {"input_ids": np.asarray([item.ids for item in tokens], dtype=np.int64),
                      "attention_mask": mask, "token_type_ids": np.zeros_like(mask)}
            embedding = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0]
            pooled = (embedding * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
            chunks.append((pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)).astype("float32"))
        return np.vstack(chunks)


class SemanticRetriever:
    name = "MiniLM / FAISS"

    def __init__(self, folder, claims):
        import faiss
        meta = json.loads((folder / "index.json").read_text())
        if meta["claims_sha256"] != claim_fingerprint(claims):
            raise ValueError("Fact-check index and database contain different records")
        for file, checksum in meta["files"].items():
            if hashlib.sha256((folder / file).read_bytes()).hexdigest() != checksum:
                raise ValueError("Fact-check embedding bundle failed checksum verification")
        self.encoder = MiniLM(folder)
        self.index = faiss.read_index(str(folder / "claims.faiss"))
        if self.index.ntotal != len(claims) or self.index.d != 384:
            raise ValueError("Fact-check index has an unexpected shape")
        self.claims = claims

    def search(self, text, limit=5, minimum=.32):
        scores, indices = self.index.search(self.encoder.encode([leading_claim(text)]), limit)
        return [{**self.claims[i], "similarity": round(float(score), 4)}
                for i, score in zip(indices[0], scores[0]) if i >= 0 and score >= minimum]


class LexicalRetriever:
    name = "TF-IDF cosine similarity"

    def __init__(self, claims):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.claims = claims
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.vectors = self.vectorizer.fit_transform([row["claim"] for row in claims]) if claims else None

    def search(self, text):
        if self.vectors is None:
            return []
        scores = (self.vectors @ self.vectorizer.transform([leading_claim(text)]).T).toarray().ravel()
        return [{**self.claims[i], "similarity": round(float(scores[i]), 4)}
                for i in np.argsort(-scores, kind="stable")[:5] if scores[i] >= .12]
