"use strict";

const $ = (selector) => document.querySelector(selector);
const result = $("#result");
const emptyResult = result.firstElementChild.cloneNode(true);
const headline = $("#headline");
const body = $("#body");
const submit = $("#submit");
const error = $("#form-error");
const tabs = ["screening", "history", "methodology"];
let settings;
let evaluation;

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function percent(value, decimals = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(decimals)}%` : "—";
}

async function api(path, options) {
  const response = await fetch(`/api/${path}`, options);
  if (!response.ok) {
    const info = await response.json().catch(() => ({}));
    throw new Error(info.detail || "The request could not be completed. Try again.");
  }
  return response.status === 204 ? null : response.json();
}

function wordCount() {
  const input = `${headline.value} ${body.value}`.replace(/https?:\/\/\S+|www\.\S+/gi, "");
  const words = input.match(/\p{L}+(?:['’]\p{L}+)*/gu)?.length ?? 0;
  $("#word-count").textContent = `${words} ${words === 1 ? "word" : "words"}`;
}

function setError(message = "") {
  error.textContent = message;
  error.hidden = !message;
}

function showResult(data) {
  const content = element("div", undefined, "result-content");
  const verdict = element("div", undefined, "verdict");
  const badge = element("span", data.label, `badge ${data.label.toLowerCase()}`);
  verdict.append(badge, element("span", percent(data.confidence, 2), "score"));
  const bar = document.createElement("meter");
  bar.value = data.confidence;
  bar.setAttribute("aria-label", "Temperature-scaled model confidence");
  content.append(verdict, bar, element("p", `Calibrated model confidence · ${data.model.name}`));
  const token = data.tokens;
  if (token.truncated) {
    content.append(element("p", `Screened the first ${token.used} of ${token.total} ${token.unit}. The remaining text was outside the model’s input window.`, "warning"));
  }
  for (const warning of data.warnings) content.append(element("p", warning, "warning"));
  const attribution = element("section", undefined, "result-section");
  attribution.append(element("h3", "Token signals"));
  attribution.append(element("p", data.attributions.length
    ? `${data.model.attribution}. Positive values favor Fake and negative values favor Real.`
    : "No token signals were returned."));
  if (data.attributions.length) {
    const chips = element("div", undefined, "chips");
    for (const item of data.attributions) {
      const sign = item.weight > 0 ? "+" : "";
      const chip = element("span", `${item.token || "‹whitespace›"} ${sign}${item.weight.toFixed(3)}`,
        item.weight > 0 ? "chip fake" : "chip");
      chips.append(chip);
    }
    attribution.append(chips);
  }
  const evidence = element("section", undefined, "result-section");
  evidence.append(element("h3", "Related fact-checks"), element("p", data.context));
  if (!data.claims.length) evidence.append(element("p", "No sufficiently similar claims were found in the index."));
  for (const item of data.claims) {
    const card = element("article", undefined, "claim");
    const url = new URL(item.url, location.origin);
    if (["http:", "https:"].includes(url.protocol)) {
      const link = element("a", item.claim);
      link.href = url.href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      card.append(link);
    } else {
      card.append(element("p", item.claim));
    }
    const meta = [item.publisher, item.rating, `${percent(item.similarity, 0)} similarity`].filter(Boolean);
    card.append(element("p", meta.join(" · ")));
    evidence.append(card);
  }
  content.append(attribution, evidence);
  return content;
}

async function loadHistory() {
  const list = $("#history-list");
  const status = $("#history-status");
  $("#clear-history").disabled = true;
  status.textContent = "Loading…";
  list.replaceChildren();
  try {
    const rows = await api("history");
    if (location.hash !== "#history") return;
    status.textContent = rows.length ? "" : "No saved screenings in this browser.";
    $("#clear-history").disabled = !rows.length;
    for (const row of rows) {
      const detail = element("details");
      const summary = element("summary", row.headline);
      const date = new Date(row.created_at).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"});
      summary.append(element("span", ` · ${row.result.label} · ${date}`));
      const content = element("div", undefined, "history-result");
      content.append(showResult(row.result));
      detail.append(summary, content);
      list.append(detail);
    }
  } catch (exception) {
    status.textContent = exception.message;
  }
}

function addTable(headers, rows) {
  const container = element("div", undefined, "table-wrap");
  const table = element("table");
  const heading = element("tr");
  for (const title of headers) {
    const item = element("th", title);
    item.scope = "col";
    heading.append(item);
  }
  const thead = element("thead");
  thead.append(heading);
  const tbody = element("tbody");
  for (const row of rows) {
    const tr = element("tr");
    for (const value of row) tr.append(element("td", value));
    tbody.append(tr);
  }
  table.append(thead, tbody);
  container.append(table);
  return container;
}

function reliability(before, after) {
  const chart = element("canvas", "Confidence versus observed accuracy by bin.", "plot");
  chart.width = 350;
  chart.height = 230;
  chart.setAttribute("role", "img");
  chart.setAttribute("aria-label", "Reliability diagram showing confidence against observed accuracy for raw and scaled predictions");
  const ctx = chart.getContext("2d");
  const x = (v) => 37 + v * 292;
  const y = (v) => 190 - v * 174;
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#43564a";
  ctx.fillText("Observed accuracy", 37, 12);
  ctx.fillText("Model confidence", 133, 225);
  for (const t of [0, .25, .5, .75, 1]) {
    ctx.fillText(percent(t, 0), x(t) - 9, 207);
    ctx.fillText(percent(t, 0), 2, y(t) + 3);
  }
  ctx.strokeStyle = "#a6b2a2";
  ctx.beginPath();
  ctx.moveTo(x(0), y(1));
  ctx.lineTo(x(0), y(0));
  ctx.lineTo(x(1), y(0));
  ctx.stroke();
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x(0), y(0));
  ctx.lineTo(x(1), y(1));
  ctx.stroke();
  ctx.setLineDash([]);
  for (const [bins, color] of [[before, "#963b29"], [after, "#285744"]]) {
    ctx.fillStyle = color;
    for (const bin of bins) {
      ctx.beginPath();
      ctx.arc(x(bin.confidence), y(bin.accuracy), 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  return chart;
}

function drawEvaluation() {
  if (!evaluation) return;
  const metrics = $("#metrics");
  metrics.replaceChildren();
  const source = $("#metrics-source");
  if ($("#metrics-select").value === "report") {
    const report = evaluation.report;
    source.textContent = `${report.source}. Accuracy by domain.`;
    metrics.append(addTable(["Model", "WELFake", "Constraint COVID-19", "LIAR", "WELFake macro-F1"],
      report.models.map((row) => [row.name, ...["welfake", "covid", "liar", "macro_f1"].map((key) => `${row[key].toFixed(2)}%`)])));
    const inDomain = element("details");
    inDomain.append(element("summary", "Reported WELFake metrics and calibration"),
      addTable(["Model", "Fake precision", "Fake recall", "Fake F1", "ROC-AUC"],
        report.models.map((row) => [row.name, ...["fake_precision", "fake_recall", "fake_f1"].map((key) => `${row[key].toFixed(2)}%`), row.roc_auc.toFixed(4)])),
      addTable(["Model", "Temperature", "Raw ECE", "Scaled ECE"],
        report.calibration.map((row) => [row.name, row.temperature?.toFixed(2) ?? "—",
          row.ece_before.toFixed(4), row.ece_after?.toFixed(4) ?? "—"])));
    metrics.append(inDomain);
    return;
  }
  const measured = evaluation.measured;
  source.textContent = measured
    ? `Local evaluation · seed ${measured.seed} · train / validation / test ${measured.split_sizes.join(" / ")}.`
    : "No locally measured results are installed.";
  if (!measured) return;
  for (const [name, values] of Object.entries(measured.models)) {
    const details = element("details");
    details.append(element("summary", name));
    if (values.temperature) details.append(element("p", `Fitted temperature ${values.temperature.toFixed(3)}`));
    const entries = Object.entries(values.domains);
    details.append(addTable(["Corpus", "Accuracy", "Macro-F1", "ROC-AUC", "ECE", "Real→Fake", "Fake→Real"],
      entries.map(([domain, item]) => [domain, percent(item.accuracy), percent(item.macro_f1),
        item.roc_auc.toFixed(3), item.calibration.ece.toFixed(4), item.confusion_matrix[0][1], item.confusion_matrix[1][0]])));
    for (const [domain, item] of entries) {
      details.append(element("p", domain));
      details.append(addTable(["Class", "Precision", "Recall", "F1", "Samples"],
        Object.entries(item.per_class).map(([label, stats]) => [label,
          percent(stats.precision), percent(stats.recall), percent(stats["f1-score"]), stats.support])));
    }
    const reference = values.domains.WELFake;
    if (reference?.calibration_before) {
      details.append(element("p", "WELFake reliability diagram · raw (rust) and scaled (green)"));
      details.append(reliability(reference.calibration_before.bins, reference.calibration.bins));
    }
    metrics.append(details);
  }
}

async function methodology() {
  if (settings) {
    const model = settings.model;
    const data = {"Classifier": model.name, "Bundle": model.version, "Input limit": `${model.max_tokens} ${model.token_unit}`,
      "Calibration": `Temperature ${model.temperature.toFixed(3)}`, "Attribution": model.attribution,
      "Evidence index": settings.retriever ? `${settings.retriever} · ${settings.claims} claims` : "Unavailable"};
    if (model.calibration_source) data["Calibration data"] = model.calibration_source;
    const details = $("#model-details");
    details.replaceChildren();
    for (const [title, value] of Object.entries(data)) details.append(element("dt", title), element("dd", value));
  }
  try {
    evaluation ??= await api("evaluation");
    drawEvaluation();
    const observed = evaluation.serving_calibration;
    const section = $("#serving-calibration");
    section.replaceChildren();
    const runtime = evaluation.runtime_benchmark;
    if (runtime) {
      const detail = element("details");
      detail.append(element("summary", "Measured CPU latency"),
        element("p", `${runtime.inference_median_ms} ms inference · ${runtime.retrieval_median_ms} ms retrieval`),
        element("p", `Warm median over ${runtime.classification_repetitions} runs at ${runtime.scored_tokens} tokens.`));
      if (Number.isFinite(runtime.shap_median_ms)) {
        detail.append(element("p", `SHAP took ${(runtime.shap_median_ms / 1000).toFixed(1)} seconds in a separate run.`));
      }
      section.append(detail);
    }
    if (observed) {
      const detail = element("details");
      detail.append(element("summary", "Serving calibration"),
        element("p", observed.source),
        element("p", `Fit on ${observed.samples.toLocaleString()} rows · raw ECE ${observed.before.ece.toFixed(4)} · scaled ECE ${observed.after.ece.toFixed(4)}`),
        element("p", "Raw probabilities in rust · scaled probabilities in green"),
        reliability(observed.before.bins, observed.after.bins));
      section.append(detail);
    }
  } catch (exception) {
    $("#metrics-source").textContent = exception.message;
  }
}

function navigate() {
  const requested = location.hash.slice(1);
  const tab = tabs.includes(requested) ? requested : "screening";
  for (const name of tabs) {
    $(`#${name}`).hidden = name !== tab;
    const link = $(`[data-tab="${name}"]`);
    if (name === tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (tab === "history") loadHistory();
  if (tab === "methodology") methodology();
}

headline.addEventListener("input", wordCount);
body.addEventListener("input", wordCount);
$("#sample").addEventListener("click", () => {
  headline.value = "Did Michelle Obama say white people are what is wrong with America?";
  body.value = "A social media post attributes this statement to Michelle Obama. The post has circulated widely and readers are asking where the quotation originated.";
  wordCount();
  setError();
  headline.focus();
});
$("#reset").addEventListener("click", () => {
  headline.value = "";
  body.value = "";
  wordCount();
  setError();
  result.replaceChildren(emptyResult.cloneNode(true));
  headline.focus();
});
$("#article-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError();
  submit.disabled = true;
  $("#reset").disabled = true;
  $("#sample").disabled = true;
  submit.querySelector("span").textContent = "Screening…";
  result.setAttribute("aria-busy", "true");
  result.replaceChildren(element("div", "Screening text and looking up related claims…", "empty"));
  try {
    const data = await api("analyze", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({headline: headline.value, body: body.value, save: $("#save").checked})});
    result.replaceChildren(showResult(data));
  } catch (exception) {
    result.replaceChildren(emptyResult.cloneNode(true));
    setError(exception.message);
  } finally {
    submit.disabled = $("#reset").disabled = $("#sample").disabled = false;
    submit.querySelector("span").textContent = "Screen article";
    result.setAttribute("aria-busy", "false");
  }
});
$("#clear-history").addEventListener("click", async () => {
  if (!window.confirm("Clear all saved screenings for this browser?")) return;
  try {
    await api("history", {method: "DELETE"});
    await loadHistory();
  } catch (exception) {
    $("#history-status").textContent = exception.message;
  }
});
$("#metrics-select").addEventListener("change", drawEvaluation);
window.addEventListener("hashchange", navigate);
api("status").then((data) => {
  settings = data;
  $("#service-status").textContent = data.model.mode === "encoder" ? "Encoder ready" : "Baseline ready";
  if (location.hash === "#methodology") methodology();
}).catch(() => {
  $("#service-status").textContent = "Service unavailable";
  $(".service").classList.add("offline");
});
navigate();
