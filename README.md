# NewsCheck

A full-stack implementation of the fake news detection minor project. The browser submits a headline, an article body, or both. FastAPI returns a **Real** or **Fake** prediction, temperature-scaled confidence, signed token attributions, and up to five related fact-checks. SQLite stores the fact-check records and optional screening history.

## Run the app

Use Python 3.12 on an x86-64 CPU with AVX2 for the bundled MiniLM model.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 3000
```

Open **http://127.0.0.1:3000** and select **Try a sample**, then **Screen article**. API documentation is at **http://127.0.0.1:3000/docs**. The bundled models, calibration and fact-check index work offline after dependency installation. The database is created on first start at `data/newscheck.sqlite3`.

History defaults to saving a headline or a short body preview and the returned result, including displayed attributions. The full body is not saved. Clear history in the browser or uncheck the save option before submitting. A random HTTP-only cookie scopes history to the browser. For independent installations, point `NEWSCHECK_DB` at a separate SQLite file.

## Report requirements

| Report IDs | Implementation |
| --- | --- |
| FR01–FR02 | Headline and body fields, 20-word minimum, non-text and size validation. |
| FR03–FR06 | Shared neural normalization, bundled tokenizer, 256 word-pieces including special tokens, explicit truncation counts, int8 ONNX CPU inference and checkpoint-specific temperature scaling. |
| FR07 | Signed SHAP permutation attributions on the exact scored token IDs. Each sign describes an effect on the Fake probability. |
| FR08–FR09 | MiniLM embeddings, versioned FAISS index, HTTPS/HTTP publisher links, textual ratings, similarity scores and a context-only caveat. |
| FR10–FR12 | Reproduction harness for all five candidates, seed-42 stratification, validation macro-F1 selection, two untouched transfer corpora, per-class metrics, confusion matrix, ROC-AUC and 15-bin ECE. |

Attribution and retrieval run concurrently. A failure in either component returns a warning with the label and confidence. The local 256-token benchmark measured a 58.53 ms median for classification and a 19.99 ms median for retrieval. Full SHAP attribution took 30.47 seconds on the same input, so the combined request exceeds the report’s 150 ms target when attribution is included. Measure warm inference, retrieval, and optional SHAP separately with `python -m ml.benchmark --shap-runs 1`. Pass `--pairs matches.json` to compute Recall@5, Recall@10 and MRR@10 from a JSON array of articles and their known matching fact-check URLs, using fields `text` and `url`.

## Included artifacts and reproducibility

- **Serving checkpoint.** The supplied report contains no weights or tokenizer. The app bundles a pinned public fine-tuned DistilRoBERTa checkpoint exported to ONNX int8. Its temperature was fitted to this quantized model using a stratified 512-row subset of the local WELFake validation split. Training overlap with the external checkpoint is unknown, so this fit does not independently substantiate how well confidence transfers. See [DistilRoBERTa checkpoint](https://huggingface.co/vikram71198/distilroberta-base-finetuned-fake-news-detection/tree/1df1a775036940eea52988c309d962c50c5c55c1). The public training recipe used a 512-token cap. Serving and calibration use the report’s 256-token cap.
- **Evidence.** The installed index contains 23,180 English publisher-linked reviews. It combines 101 Data Commons sample records with a filtered subset of Martino Mensio’s ClaimReview archive. This collection differs from the report’s and its 300 matched evaluation articles were unavailable. The ClaimReview archive and derived data are distributed under CC BY-NC-SA 4.0 for this academic project. The Data Commons sample retains original publisher attribution and links. See [ClaimReview release](https://github.com/MartinoMensio/claimreview-data/releases/tag/2026_11_25), [Data Commons sample via BERD](https://berd-platform.de/records/zyhb7-nmp30), and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **Measured results.** The two TF-IDF baselines were trained locally on WELFake and tested on WELFake, LIAR, and Constraint COVID-19. Their stored metrics cover the first 256 lexical tokens of each submission, matching the portable baseline runtime. The browser shows these measurements and the PDF’s published figures under their respective source labels.
- **Split provenance.** The verified Zenodo CSV has 72,134 rows. In this file label 0 occurs 35,028 times and includes Reuters reporting. Label 1 occurs 37,106 times and includes fabricated stories. This matches the report’s totals by class. The Zenodo prose describes the opposite encoding. The loader defaults to `1 = Fake`, with an override in the full harness. Short-article filtering and exact/near-duplicate removal (64-bit SimHash followed by trigram Jaccard confirmation) leave 62,620 usable articles and a 43,834 / 9,393 / 9,393 split. The public LIAR test file has 1,283 mapped statements. The report gives 1,267 and does not list its exclusions. See [WELFake](https://zenodo.org/records/4561253) and [LIAR](https://aclanthology.org/P17-2067/).
- **Full comparison.** The five-candidate workflow is provided as source. The complete GPU training run and hosted Llama evaluation require a suitable GPU and an API key. They were not executed for this delivery. The report’s 98.91% accuracy and calibration figures apply to its original checkpoint.

## Re-run the study

Use the public [WELFake CSV](https://zenodo.org/records/4561253/files/WELFake_Dataset.csv?download=1), [LIAR test TSV](https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv), and [Constraint COVID-19 labelled test CSV](https://huggingface.co/datasets/nanyy1025/covid_fake_news/resolve/079d310163b36c9a72fe327d905cf21a21b8d5a6/test.csv). The COVID-19 file has 2,140 records. The LIAR loader maps pants-fire, false, and barely-true to Fake.

```sh
python -m pip install -r requirements-report.txt
python -m ml.train_baseline WELFake_Dataset.csv --liar test.tsv --covid test.csv --stemming
export LLAMA_API_KEY=your_provider_key
python -m ml.experiment --welfake WELFake_Dataset.csv --liar test.tsv --covid test.csv \
  --llama-endpoint https://YOUR_PROVIDER/v1/chat/completions --output runs/report
```

Use a provider that serves Llama 3.1 8B Instruct and pass its model identifier with `--llama-model` if needed. The full run sends benchmark text only to this configured endpoint and caches its results for resume. Browser submissions always stay with the local service. Use `--resume` for an interrupted training run. Evaluation output includes per-domain calibration bins, confusion matrices and a fingerprint of the shared split. Copy `runs/report/evaluation.json` into `artifacts/` to display it in Methodology. Set `NEWSCHECK_BUNDLE=runs/report/encoder` to serve the newly trained and calibrated bundle.

## Update the fact-check index

Import a ClaimReview JSON, JSONL or Data Commons JSON-LD export with compatible rights, then rebuild the FAISS index alongside the bundled MiniLM model and tokenizer.

```sh
python -m ml.import_claims claimreviews.json data/new-claims.json
python -m ml.index_claims artifacts/retrieval data/new-claims.json
NEWSCHECK_CLAIMS=data/new-claims.json NEWSCHECK_DB=data/new-claims.sqlite3 \
  python -m uvicorn app.main:app --port 3000
```

To reproduce the supplied selection from the cited release ZIP, use `python -m ml.import_archive 2026_11_25.zip data/new-claims.json --seed data/seed-claims.json` before building the index.

The retrieval manifest fingerprints the complete ordered claim collection, MiniLM artifacts and FAISS index. The serving checkpoint separately fingerprints its ONNX weights and tokenizer. The app uses the included TF-IDF baseline if no encoder manifest is installed. You can select another organized artifact directory via `NEWSCHECK_ARTIFACTS`.

The tool screens English writing patterns. Similar wording is never treated as evidence that the submitted article is true or false.

## Container deployment

Build the image on an x86-64 host with AVX2. The image includes the native Python runtime, ONNX checkpoints and FAISS index. It seeds a fresh SQLite database at startup.

```sh
docker build -t newscheck .
docker run --rm -p 3000:7860 \
  --mount type=volume,source=newscheck-data,target=/var/lib/newscheck newscheck
```

For a hosted installation, route HTTPS to port 7860 and mount a persistent writable volume at `/var/lib/newscheck` for screening history. Use one replica for the SQLite database and give the upstream request timeout enough time for full SHAP attribution. The container health check verifies that the encoder and semantic retriever both loaded. The bundled ClaimReview dataset carries the noncommercial terms documented above.

### Railway

The included `railway.toml` selects the Dockerfile, one replica, and the status endpoint as a startup check. Mount a Railway volume at `/var/lib/newscheck` before the first deployment. The service listens on port 7860. Set the Railway service variable `PORT=7860` and route its public domain to that port.

### Vercel

Vercel uses `Dockerfile.vercel` to run the same ONNX and FAISS pipeline. Connect a PostgreSQL integration such as Neon to the project and let it inject `DATABASE_URL` into the relevant deployment environments. The app creates its screening-history table at startup. The local SQLite claim catalog is rebuilt from the bundled claim records and is disposable on Vercel.

After linking the project with the Vercel CLI, run `vercel deploy --archive=tgz` for a preview. Verify `/api/status`, screening, and saved history before publishing to production. Keep database credentials in the Vercel environment configuration.
