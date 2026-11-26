import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from app.text import LIMIT, normalize
from ml.calibration import calibration, fit_temperature, probability


def export(checkpoint, output):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, attn_implementation="eager").eval()
    if model.config.model_type != "roberta" or model.config.num_hidden_layers != 6 or model.config.num_labels != 2:
        raise ValueError("Expected binary DistilRoBERTa classifier")
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer needs a mask token for attribution")

    class Wrapper(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    sample = tokenizer("Example text for export.", return_tensors="pt")
    fp32 = output / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(Wrapper(model).eval(), (sample["input_ids"], sample["attention_mask"]),
                          str(fp32), input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                        "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
    gc.collect()
    int8 = output / "model.int8.onnx"
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    fp32.unlink()


def margins(folder, texts, batch_size=16):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    tokenizer.enable_truncation(max_length=LIMIT)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(folder / "model.int8.onnx"), options, providers=["CPUExecutionProvider"])
    output = []
    for index in range(0, len(texts), batch_size):
        encoded = tokenizer.encode_batch([normalize(text) for text in texts[index:index + batch_size]])
        inputs = {"input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                  "attention_mask": np.asarray([item.attention_mask for item in encoded], dtype=np.int64)}
        logits = session.run(["logits"], inputs)[0]
        output.extend(logits[:, 1] - logits[:, 0])
    return np.asarray(output)


def bundle(folder, scores, labels, source, version, calibration_source):
    temperature = fit_temperature(scores, labels)
    filenames = ("model.int8.onnx", "tokenizer.json")
    meta = {"name": "DistilRoBERTa · ONNX int8", "mode": "encoder", "version": version,
            "checkpoint": source, "temperature": temperature, "max_tokens": LIMIT,
            "token_unit": "word-piece tokens", "attribution": "SHAP effects on Fake probability",
            "calibration_source": calibration_source, "calibration_samples": len(labels),
            "labels": ["Real", "Fake"],
            "files": {file: hashlib.sha256((folder / file).read_bytes()).hexdigest() for file in filenames}}
    (folder / "manifest.json").write_text(json.dumps(meta, indent=2))
    observed = {"samples": len(labels), "seed": 42, "source": calibration_source,
                "temperature": temperature, "before": calibration(probability(scores, 1), np.asarray(labels)),
                "after": calibration(probability(scores, temperature), np.asarray(labels))}
    (folder / "calibration.json").write_text(json.dumps(observed, indent=2))
    return temperature


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from ml.corpora import SEED, near_duplicates, split, welfake

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("welfake", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--calibrate-only", action="store_true", help="Use the existing ONNX and tokenizer files")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("Use at least two calibration samples")
    if not args.calibrate_only:
        export(args.checkpoint, args.output)
        print("Exported int8 model", flush=True)
    _, (texts, labels), _ = split(list(near_duplicates(welfake(args.welfake))))
    if args.samples < len(labels):
        texts, _, labels, _ = train_test_split(texts, labels, train_size=args.samples, random_state=SEED, stratify=labels)
    scores = margins(args.output, texts)
    temperature = bundle(args.output, scores, labels, args.source, args.version,
                         "WELFake validation subset, seed 42. Checkpoint training overlap is unknown.")
    print(f"Temperature {temperature:.4f} fitted on {len(labels)} samples", flush=True)
