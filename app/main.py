import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from app.database import Database
from app.model import load_model
from app.retrieval import LexicalRetriever, SemanticRetriever
from app.text import MIN_WORDS, combine, normalize, validate

ROOT = Path(__file__).resolve().parents[1]
COOKIE = "newscheck_session"
LOG = logging.getLogger(__name__)
CONTEXT = "Related fact-checks are provided for context only. Similarity does not verify the submitted article."


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: str = Field(default="", max_length=500)
    body: str = Field(default="", max_length=30000)
    save: bool = True


def asset(setting, default):
    return Path(os.environ.get(setting, default))


def session(request, response):
    value = request.cookies.get(COOKIE, "")
    if len(value) == 43 and all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in value):
        return value
    value = secrets.token_urlsafe(32)
    response.set_cookie(COOKIE, value, max_age=60 * 60 * 24 * 30,
                        httponly=True, secure=request.url.scheme == "https", samesite="strict")
    return value


def optional(name, operation):
    try:
        return operation(), None
    except Exception:
        LOG.exception("%s unavailable", name)
        return [], f"{name} unavailable for this screening."


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        artifacts = asset("NEWSCHECK_ARTIFACTS", ROOT / "artifacts")
        configured = os.environ.get("NEWSCHECK_BUNDLE")
        bundle = Path(configured) if configured else artifacts / "encoder"
        model = load_model(bundle if configured or (bundle / "manifest.json").is_file() else None, artifacts)
        database = Database(asset("NEWSCHECK_DB", ROOT / "data" / "newscheck.sqlite3"))
        database.initialize(asset("NEWSCHECK_CLAIMS", ROOT / "data" / "claims.json"))
        history = database
        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            from app.postgres_history import PostgresHistory

            history = PostgresHistory(database_url)
            history.initialize()
        elif os.environ.get("NEWSCHECK_REQUIRE_PERSISTENT_HISTORY") == "1":
            raise RuntimeError("Set DATABASE_URL to the hosted PostgreSQL connection URL.")
        claims = database.claims()
        retrieval_folder = asset("NEWSCHECK_RETRIEVAL", artifacts / "retrieval")
        try:
            retriever = (SemanticRetriever(retrieval_folder, claims) if (retrieval_folder / "index.json").is_file()
                         else LexicalRetriever(claims))
        except Exception:
            LOG.exception("Fact-check index unavailable")
            retriever = None
        app.state.model = model
        app.state.db = history
        app.state.retriever = retriever
        app.state.claim_count = len(claims)
        app.state.artifacts = artifacts
        app.state.explanation_slots = asyncio.Semaphore(1)
        app.state.bundle = bundle
        yield

    app = FastAPI(title="NewsCheck", version="1.0", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        messages = {"headline": "Keep the headline under 501 characters.",
                    "body": "Keep the article under 30,001 characters."}
        fields = [error["loc"][-1] for error in exc.errors() if error["type"] == "string_too_long"]
        return JSONResponse({"detail": messages.get(fields[0], "Send a valid headline or article.") if fields
                             else "Send a valid headline or article."}, status_code=422)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/status")
    def status(request: Request):
        state = request.app.state
        return {"model": state.model.meta, "min_words": MIN_WORDS, "claims": state.claim_count,
                "retriever": state.retriever.name if state.retriever else None}

    @app.get("/api/evaluation")
    def evaluation(request: Request):
        path = request.app.state.artifacts / "evaluation.json"
        if not path.is_file():
            path = request.app.state.artifacts / "baseline-evaluation.json"
        local = json.loads(path.read_text()) if path.is_file() else None
        calibration_path = request.app.state.bundle / "calibration.json"
        serving = json.loads(calibration_path.read_text()) if calibration_path.is_file() else None
        benchmark_path = request.app.state.artifacts / "benchmark.json"
        runtime = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else None
        return {"measured": local, "serving_calibration": serving, "runtime_benchmark": runtime, "report": json.loads((ROOT / "data" / "report-metrics.json").read_text())}

    @app.post("/api/analyze")
    async def analyze(payload: Submission, request: Request, response: Response):
        state = request.app.state
        combined = combine(payload.headline, payload.body)
        try:
            validate(combined)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        scored = await asyncio.to_thread(state.model.score, combined)

        async def explain():
            async with state.explanation_slots:
                return await asyncio.to_thread(optional, "Token attribution", lambda: state.model.explain(scored))

        async def retrieve():
            if state.retriever is None:
                return [], "Fact-check retrieval is unavailable."
            query = normalize(payload.headline) or scored.text
            return await asyncio.to_thread(optional, "Fact-check retrieval", lambda: state.retriever.search(query))

        (attributions, attribution_error), (claims, retrieval_error) = await asyncio.gather(explain(), retrieve())
        if attribution_error or retrieval_error:
            attributions, claims = [], []
        fake = scored.fake_probability
        output = {"label": "Fake" if fake >= .5 else "Real", "confidence": max(fake, 1 - fake),
                  "probabilities": {"Real": 1 - fake, "Fake": fake}, "model": state.model.meta,
                  "tokens": {"used": scored.used, "total": scored.total, "truncated": scored.used < scored.total,
                             "unit": state.model.meta["token_unit"]},
                  "attributions": attributions, "claims": claims, "context": CONTEXT,
                  "warnings": [value for value in (attribution_error, retrieval_error) if value]}
        if payload.save:
            preview = normalize(payload.headline) or normalize(payload.body)[:100]
            output["id"] = await asyncio.to_thread(state.db.save, session(request, response), preview[:150], output)
        return output

    @app.get("/api/history")
    def history(request: Request, response: Response):
        return request.app.state.db.history(session(request, response))

    @app.delete("/api/history", status_code=204)
    def clear_history(request: Request, response: Response):
        request.app.state.db.clear(session(request, response))
        response.status_code = 204
        return response

    return app


app = create_app()
