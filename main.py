from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, render_template_string, request, send_file
from werkzeug.utils import secure_filename

from src.ge_validator import run_validation
from src.llm1_ge_writer import run_llm_ge_writer
from src.llm2_analyst import run_analysis
from src.parser import parse_logs
from src.pattern_engine import run_pattern_engine
from src.visualizer import run_visualizer

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
MPLCONFIG_DIR = ROOT / ".cache" / "matplotlib"
PARSED_OUTPUT = OUTPUT_DIR / "parsed_logs.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_results.json"
LLM1_RESULTS_OUTPUT = OUTPUT_DIR / "validation_results_llm.json"
LLM1_SUITE_OUTPUT = OUTPUT_DIR / "ge_suite_llm.py"
PATTERNS_OUTPUT = OUTPUT_DIR / "patterns_found.json"
LLM2_OUTPUT = OUTPUT_DIR / "analysis_report.txt"
DASHBOARD_OUTPUT = OUTPUT_DIR / "pipeline_report.html"
ALLOWED_EXTENSIONS = {".csv", ".log"}

MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log Quality Studio</title>
  <style>
    :root {
      --bg: #f7fbff;
      --bg-soft: #eef6fc;
      --panel: #ffffff;
      --panel-soft: #f9fcff;
      --text: #284358;
      --muted: #607b91;
      --line: #d7e6f1;
      --accent: #72b7dd;
      --accent-strong: #4d98ca;
      --accent-soft: #ebf7ff;
      --success: #2f855a;
      --success-soft: #eefaf3;
      --warn: #b7791f;
      --warn-soft: #fff8ea;
      --danger: #c05621;
      --danger-soft: #fff1eb;
      --shadow: 0 24px 60px rgba(84, 118, 148, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(157, 215, 244, 0.28), transparent 24%),
        radial-gradient(circle at top right, rgba(203, 232, 249, 0.5), transparent 24%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-soft) 100%);
    }
    .page {
      width: min(1380px, calc(100% - 28px));
      margin: 18px auto 36px;
    }
    .hero {
      padding: 30px;
      border-radius: 30px;
      border: 1px solid rgba(215, 230, 241, 0.88);
      background: rgba(255, 255, 255, 0.84);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 0.82rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    h1 {
      margin: 14px 0 10px;
      font-size: clamp(2.3rem, 4vw, 4.2rem);
      line-height: 0.92;
      letter-spacing: -0.05em;
    }
    .lead {
      max-width: 920px;
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 1.03rem;
    }
    .layout {
      display: grid;
      grid-template-columns: 480px minmax(0, 1fr);
      gap: 20px;
      margin-top: 20px;
      align-items: start;
    }
    .panel {
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid rgba(215, 230, 241, 0.92);
      border-radius: 26px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-body {
      padding: 22px;
    }
    .stack {
      display: grid;
      gap: 16px;
    }
    h2, h3 {
      margin: 0 0 12px;
      letter-spacing: -0.03em;
    }
    .muted {
      color: var(--muted);
      line-height: 1.6;
      font-size: 0.94rem;
    }
    .field { margin-bottom: 16px; }
    label {
      display: block;
      margin-bottom: 8px;
      font-size: 0.95rem;
      font-weight: 800;
    }
    input[type=file], textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel-soft);
      color: var(--text);
      padding: 14px;
      font: inherit;
    }
    textarea {
      min-height: 180px;
      resize: vertical;
      line-height: 1.5;
    }
    .hint {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.55;
    }
    .toggle-card {
      margin: 12px 0;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, #fbfeff 0%, #f2f8fd 100%);
    }
    .toggle {
      display: flex;
      gap: 10px;
      align-items: start;
      margin: 0;
      font-weight: 800;
    }
    .toggle small {
      display: block;
      margin-top: 2px;
      font-weight: 500;
      color: var(--muted);
      line-height: 1.45;
    }
    button {
      width: 100%;
      margin-top: 8px;
      padding: 14px 18px;
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #fff;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 16px 30px rgba(77, 152, 202, 0.24);
    }
    .status {
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      line-height: 1.5;
      font-weight: 600;
    }
    .status.ok { background: var(--success-soft); color: var(--success); border-color: rgba(47, 133, 90, 0.16); }
    .status.warn { background: var(--warn-soft); color: var(--warn); border-color: rgba(183, 121, 31, 0.16); }
    .status.err { background: var(--danger-soft); color: var(--danger); border-color: rgba(192, 86, 33, 0.16); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }
    .metric {
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, #fff 0%, #f6fbff 100%);
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.82rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 1.5rem;
      line-height: 1.12;
      word-break: break-word;
    }
    .stage-list {
      display: grid;
      gap: 12px;
    }
    .stage-card {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fcff 100%);
      overflow: hidden;
    }
    .stage-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 18px 10px;
    }
    .stage-name {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 800;
    }
    .stage-num {
      display: inline-flex;
      width: 30px;
      height: 30px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 0.85rem;
    }
    .badge {
      padding: 8px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }
    .badge.ok { background: var(--success-soft); color: var(--success); }
    .badge.warn { background: var(--warn-soft); color: var(--warn); }
    .badge.err { background: var(--danger-soft); color: var(--danger); }
    .stage-body {
      padding: 0 18px 18px;
    }
    .arrowbar {
      position: relative;
      height: 14px;
      margin: 14px 0 18px;
      border-radius: 999px;
      background: linear-gradient(90deg, #eaf5fd 0%, #dff1fc 100%);
      overflow: hidden;
    }
    .arrowfill {
      position: absolute;
      top: 0;
      left: -25%;
      width: 35%;
      height: 100%;
      background: linear-gradient(90deg, transparent 0%, var(--accent) 18%, var(--accent-strong) 50%, transparent 100%);
      clip-path: polygon(0 0, 82% 0, 100% 50%, 82% 100%, 0 100%, 14% 50%);
      animation: sweep 1.3s linear infinite;
      opacity: 0.95;
    }
    @keyframes sweep {
      0% { left: -28%; }
      100% { left: 102%; }
    }
    .mini-metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .pill {
      padding: 8px 11px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 0.83rem;
      font-weight: 800;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fff;
      padding: 0 14px;
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      list-style: none;
      padding: 14px 0;
      font-weight: 800;
    }
    summary::-webkit-details-marker { display: none; }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: #fff;
    }
    table {
      width: 100%;
      min-width: 720px;
      border-collapse: collapse;
    }
    th, td {
      padding: 11px 12px;
      text-align: left;
      border-bottom: 1px solid #e9f1f7;
      vertical-align: top;
      font-size: 0.93rem;
    }
    th {
      background: #f4fbff;
      color: #486883;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    pre {
      margin: 0 0 14px;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #f5fbff;
      color: #39546c;
      overflow: auto;
      line-height: 1.55;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.88rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .dashboard-frame {
      width: 100%;
      min-height: 760px;
      border: 0;
      border-radius: 20px;
      background: #fff;
    }
    .link-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 14px;
      border-radius: 999px;
      text-decoration: none;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-weight: 800;
    }
    .loading-overlay {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(247, 251, 255, 0.84);
      backdrop-filter: blur(6px);
      z-index: 9999;
      padding: 20px;
    }
    .loading-overlay.show { display: flex; }
    .loading-card {
      width: min(760px, 100%);
      padding: 26px;
      border-radius: 28px;
      background: rgba(255,255,255,0.95);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .loading-steps {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .loading-step {
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #f9fcff;
      color: var(--muted);
      font-weight: 700;
    }
    .loading-step.active {
      background: var(--accent-soft);
      color: var(--accent-strong);
      border-color: rgba(77, 152, 202, 0.25);
    }
    .hidden { display: none; }
    @media (max-width: 1080px) {
      .layout { grid-template-columns: 1fr; }
      .dashboard-frame { min-height: 560px; }
    }
  </style>
</head>
<body>
  <div id="loadingOverlay" class="loading-overlay">
    <div class="loading-card">
      <div class="eyebrow">Pipeline Running</div>
      <h2 style="margin-top:12px;">The system is processing your file in the background.</h2>
      <p class="muted">Each stage runs in sequence: parser, manual validator, LLM1 rule writer, pattern engine, LLM2 root-cause analyzer, then the dashboard renderer.</p>
      <div class="arrowbar"><div class="arrowfill"></div></div>
      <div class="loading-steps">
        <div class="loading-step active">Stage 1. Parsing uploaded file into structured data</div>
        <div class="loading-step">Stage 2. Running manual Great Expectations rules</div>
        <div class="loading-step">Stage 3. Asking Qwen on Hugging Face to generate validation rules</div>
        <div class="loading-step">Stage 4. Detecting patterns from parsed logs</div>
        <div class="loading-step">Stage 5. Running LLM2 root-cause analysis from prior outputs</div>
        <div class="loading-step">Stage 6. Rendering pipeline dashboard</div>
      </div>
    </div>
  </div>

  <main class="page">
    <section class="hero">
      <div class="eyebrow">Log Quality Pipeline</div>
      <h1>Upload once, inspect every stage.</h1>
      <p class="lead">The app accepts a raw <code>.log</code> or any <code>.csv</code>, parses it, runs manual validation rules, lets Qwen on Hugging Face write additional GE rules, detects structural patterns, asks a second LLM to connect all earlier outputs into a root-cause story, and then shows the dashboard from <code>pipeline_report.html</code>.</p>
    </section>

    <section class="layout">
      <aside class="panel">
        <div class="panel-body">
          <h2>Run Setup</h2>
          <form id="runForm" method="post" enctype="multipart/form-data">
            <div class="field">
              <label for="log_file">Upload file</label>
              <input id="log_file" name="log_file" type="file" accept=".csv,.log" required>
              <p class="hint">The parser now accepts arbitrary CSV columns and tries to infer content, timestamp, level, component, and event id before enrichment.</p>
            </div>

            <div class="toggle-card">
              <label class="toggle" for="use_custom_rules">
                <input id="use_custom_rules" name="use_custom_rules" type="checkbox" {% if use_custom_rules %}checked{% endif %}>
                <span>Use custom manual validation rules
                  <small>If this stays off, the system uses the existing built-in `ge_validator` rules.</small>
                </span>
              </label>
            </div>

            <div id="rules_panel" class="{% if not use_custom_rules %}hidden{% endif %}">
              <div class="field">
                <label for="custom_rules">Rules JSON</label>
                <textarea id="custom_rules" name="custom_rules" spellcheck="false">{{ custom_rules }}</textarea>
                <p class="hint">Supported types: <code>not_null</code>, <code>unique</code>, <code>in_set</code>, <code>regex</code>, <code>between</code>, <code>mean_between</code>.</p>
              </div>
            </div>

            <div class="toggle-card">
              <label class="toggle" for="run_llm1">
                <input id="run_llm1" name="run_llm1" type="checkbox" {% if run_llm1 %}checked{% endif %}>
                <span>Run LLM1 on Hugging Face
                  <small>This uses the parsed uploaded file and writes `ge_suite_llm.py` plus `validation_results_llm.json`.</small>
                </span>
              </label>
            </div>

            <div class="toggle-card">
              <label class="toggle" for="run_llm2">
                <input id="run_llm2" name="run_llm2" type="checkbox" {% if run_llm2 %}checked{% endif %}>
                <span>Run LLM2 root-cause analysis
                  <small>LLM2 uses the parsed logs, manual GE output, LLM1 output, and pattern engine output together.</small>
                </span>
              </label>
            </div>

            <button type="submit">Run full pipeline</button>
          </form>
        </div>
      </aside>

      <section class="stack">
        {% if message %}
          <div class="status {{ status_kind }}">{{ message }}</div>
        {% endif %}

        {% if summary %}
          <div class="panel">
            <div class="panel-body">
              <h2>Run Summary</h2>
              <div class="metrics">
                <div class="metric"><span>Uploaded file</span><strong>{{ summary.filename }}</strong></div>
                <div class="metric"><span>Rows parsed</span><strong>{{ summary.rows }}</strong></div>
                <div class="metric"><span>Manual rules</span><strong>{{ summary.manual_total }}</strong></div>
                <div class="metric"><span>Manual passed</span><strong>{{ summary.manual_passed }}</strong></div>
                <div class="metric"><span>LLM rules</span><strong>{{ summary.llm_total }}</strong></div>
                <div class="metric"><span>LLM passed</span><strong>{{ summary.llm_passed }}</strong></div>
                <div class="metric"><span>Dashboard</span><strong>{{ 'Ready' if dashboard_ready else 'Pending' }}</strong></div>
              </div>
            </div>
          </div>
        {% endif %}

        {% if stages %}
          <div class="panel">
            <div class="panel-body">
              <h2>Pipeline Stages</h2>
              <div class="stage-list">
                {% for stage in stages %}
                  <div class="stage-card">
                    <div class="stage-head">
                      <div class="stage-name">
                        <span class="stage-num">{{ loop.index }}</span>
                        <span>{{ stage.title }}</span>
                      </div>
                      <span class="badge {{ stage.status_kind }}">{{ stage.status }}</span>
                    </div>
                    <div class="stage-body">
                      <div class="arrowbar"><div class="arrowfill"></div></div>
                      <p class="muted">{{ stage.message }}</p>

                      {% if stage.metrics %}
                        <div class="mini-metrics">
                          {% for metric in stage.metrics %}
                            <span class="pill">{{ metric }}</span>
                          {% endfor %}
                        </div>
                      {% endif %}

                      {% if stage.preview_table %}
                        <details>
                          <summary>{{ stage.preview_title }}</summary>
                          <div class="table-wrap" style="margin-bottom:14px;">
                            <table>
                              <thead>
                                <tr>
                                  {% for column in stage.preview_table.columns %}
                                    <th>{{ column }}</th>
                                  {% endfor %}
                                </tr>
                              </thead>
                              <tbody>
                                {% for row in stage.preview_table.rows %}
                                  <tr>
                                    {% for value in row %}
                                      <td>{{ value }}</td>
                                    {% endfor %}
                                  </tr>
                                {% endfor %}
                              </tbody>
                            </table>
                          </div>
                        </details>
                      {% endif %}

                      {% if stage.preview_text %}
                        <details>
                          <summary>{{ stage.preview_title }}</summary>
                          <pre>{{ stage.preview_text }}</pre>
                        </details>
                      {% endif %}
                    </div>
                  </div>
                {% endfor %}
              </div>
            </div>
          </div>
        {% endif %}

        {% if dashboard_ready %}
          <div class="panel">
            <div class="panel-body">
              <h2>Dashboard</h2>
              <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px;">
                <span class="pill">Source: pipeline_report.html</span>
                <a class="link-button" href="/dashboard" target="_blank">Open dashboard in new tab</a>
              </div>
              <iframe class="dashboard-frame" src="/dashboard"></iframe>
            </div>
          </div>
        {% endif %}
      </section>
    </section>
  </main>

  <script>
    const toggle = document.getElementById("use_custom_rules");
    const panel = document.getElementById("rules_panel");
    const form = document.getElementById("runForm");
    const overlay = document.getElementById("loadingOverlay");
    const loadingSteps = Array.from(document.querySelectorAll(".loading-step"));

    toggle.addEventListener("change", () => {
      panel.classList.toggle("hidden", !toggle.checked);
    });

    form.addEventListener("submit", () => {
      overlay.classList.add("show");
      let current = 0;
      loadingSteps[current].classList.add("active");
      const interval = setInterval(() => {
        loadingSteps.forEach((step, index) => {
          step.classList.toggle("active", index === current);
        });
        current = (current + 1) % loadingSteps.length;
      }, 1300);
      window.__pipelineLoadingInterval = interval;
    });
  </script>
</body>
</html>
"""


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _status_kind(success: bool, *, warning: bool = False) -> str:
    if warning:
        return "warn"
    return "ok" if success else "err"


def _table_from_dataframe(df, limit_rows: int = 8, limit_cols: int = 10) -> dict:
    subset = df.iloc[:limit_rows, :limit_cols].fillna("")
    return {
        "columns": list(subset.columns),
        "rows": subset.astype(str).values.tolist(),
    }


def _manual_validation_rows(validation: dict) -> list[dict]:
    rows = []
    for item in validation.get("results", []):
        cfg = item.get("expectation_config", {})
        kwargs = cfg.get("kwargs", {})
        result = item.get("result", {})
        rows.append(
            {
                "rule_type": cfg.get("type", "?"),
                "column": kwargs.get("column", "—"),
                "success": "pass" if item.get("success") else "fail",
                "unexpected_count": result.get("unexpected_count", 0),
            }
        )
    return rows


def _llm1_rule_lines() -> list[str]:
    if not LLM1_SUITE_OUTPUT.exists():
        return []
    return [
        line.strip()
        for line in LLM1_SUITE_OUTPUT.read_text(encoding="utf-8").splitlines()
        if "suite.add_expectation(" in line
    ]


def _llm_stats(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("statistics", {})


def _clear_outputs(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def run_pipeline(
    *,
    input_path: Path,
    custom_rules: str | list[dict] | None = None,
    run_llm1: bool = True,
    run_llm2: bool = True,
    strict: bool = False,
) -> dict:
    parsed_df = parse_logs(input_path=input_path, output_path=PARSED_OUTPUT)
    validation = run_validation(
        input_path=PARSED_OUTPUT,
        output_path=VALIDATION_OUTPUT,
        custom_rules=custom_rules,
    )

    llm1_error = None
    llm1_result = None
    if run_llm1:
        _clear_outputs(LLM1_RESULTS_OUTPUT, LLM1_SUITE_OUTPUT)
        try:
            llm1_result = run_llm_ge_writer(
                parsed_path=PARSED_OUTPUT,
                suite_path=LLM1_SUITE_OUTPUT,
                results_path=LLM1_RESULTS_OUTPUT,
            )
        except Exception as exc:
            _clear_outputs(LLM1_RESULTS_OUTPUT, LLM1_SUITE_OUTPUT)
            llm1_error = str(exc)
            if strict:
                raise

    patterns = run_pattern_engine(input_path=PARSED_OUTPUT, output_path=PATTERNS_OUTPUT)

    llm2_text = None
    llm2_error = None
    if run_llm2:
        _clear_outputs(LLM2_OUTPUT)
        try:
            llm2_text = run_analysis(
                parsed_path=PARSED_OUTPUT,
                patterns_path=PATTERNS_OUTPUT,
                validation_path=VALIDATION_OUTPUT,
                llm1_path=LLM1_RESULTS_OUTPUT,
                output_path=LLM2_OUTPUT,
                provider_override="huggingface",
            )
        except Exception as exc:
            _clear_outputs(LLM2_OUTPUT)
            llm2_error = str(exc)
            if strict:
                raise
    elif LLM2_OUTPUT.exists():
        llm2_text = LLM2_OUTPUT.read_text(encoding="utf-8")

    report_path = run_visualizer(
        parsed_path=PARSED_OUTPUT,
        patterns_path=PATTERNS_OUTPUT,
        output_path=DASHBOARD_OUTPUT,
    )

    return {
        "parsed_df": parsed_df,
        "validation": validation,
        "llm1_result": llm1_result,
        "llm1_error": llm1_error,
        "patterns": patterns,
        "llm2_text": llm2_text,
        "llm2_error": llm2_error,
        "report_path": report_path,
    }


def _pattern_rows(patterns: dict) -> list[dict]:
    rows = []
    for item in patterns.get("patterns", []):
        compact = {k: v for k, v in item.items() if k not in {"description", "pattern"}}
        rows.append(
            {
                "pattern": item.get("pattern", "?"),
                "description": item.get("description", ""),
                "details": json.dumps(compact, indent=2, default=str),
            }
        )
    return rows


def _build_stage_cards(
    *,
    parsed_df,
    validation: dict,
    llm1_error: str | None,
    patterns: dict,
    llm2_text: str | None,
    llm2_error: str | None,
) -> list[dict]:
    manual_rows = _manual_validation_rows(validation)
    llm1_stats = _llm_stats(LLM1_RESULTS_OUTPUT)
    llm1_lines = _llm1_rule_lines()
    pattern_rows = _pattern_rows(patterns)

    stages = [
        {
            "title": "Parser",
            "status": "Complete",
            "status_kind": "ok",
            "message": "The uploaded file was normalized and enriched by parser.py into the parsed log dataset used by the rest of the pipeline.",
            "metrics": [
                f"Rows: {len(parsed_df)}",
                f"Columns: {len(parsed_df.columns)}",
            ],
            "preview_title": "Parsed rows preview",
            "preview_table": _table_from_dataframe(parsed_df),
            "preview_text": None,
        },
        {
            "title": "Manual GE Validator",
            "status": "Complete",
            "status_kind": "ok" if validation.get("success") else "warn",
            "message": "These are the active manual system rules from ge_validator and the result of checking them against the parsed upload.",
            "metrics": [
                f"Evaluated: {validation.get('statistics', {}).get('evaluated_expectations', 0)}",
                f"Passed: {validation.get('statistics', {}).get('successful_expectations', 0)}",
                f"Failed: {validation.get('statistics', {}).get('unsuccessful_expectations', 0)}",
            ],
            "preview_title": "Manual rules and results",
            "preview_table": {
                "columns": ["rule_type", "column", "success", "unexpected_count"],
                "rows": [[row["rule_type"], row["column"], row["success"], row["unexpected_count"]] for row in manual_rows],
            },
            "preview_text": None,
        },
    ]

    llm1_success = llm1_stats is not None and not llm1_error
    stages.append(
        {
            "title": "LLM1 Rule Writer",
            "status": "Complete" if llm1_success else "Warning",
            "status_kind": "ok" if llm1_success else "warn",
            "message": (
                "Qwen on Hugging Face generated Great Expectations rules from the parsed upload."
                if llm1_success
                else f"LLM1 was attempted, but it did not complete cleanly: {llm1_error or 'No generated output was found.'}"
            ),
            "metrics": (
                [
                    f"Generated rules: {len(llm1_lines)}",
                    f"Passed: {llm1_stats.get('successful_expectations', 0)}" if llm1_stats else "Passed: 0",
                    f"Evaluated: {llm1_stats.get('evaluated_expectations', 0)}" if llm1_stats else "Evaluated: 0",
                ]
            ),
            "preview_title": "Generated LLM1 rules",
            "preview_table": None,
            "preview_text": "\n".join(llm1_lines[:60]) if llm1_lines else "No LLM1 suite lines available.",
        }
    )

    stages.append(
        {
            "title": "Pattern Engine",
            "status": "Complete",
            "status_kind": "ok",
            "message": "The pattern engine searched the parsed logs for structural and behavioral signals that can later support the root-cause explanation.",
            "metrics": [
                f"Patterns found: {len(pattern_rows)}",
                f"Rows analyzed: {patterns.get('total_rows', len(parsed_df))}",
            ],
            "preview_title": "Pattern findings",
            "preview_table": {
                "columns": ["pattern", "description", "details"],
                "rows": [[row["pattern"], row["description"], row["details"]] for row in pattern_rows],
            },
            "preview_text": None,
        }
    )

    stages.append(
        {
            "title": "LLM2 Root-Cause Analyzer",
            "status": "Complete" if llm2_text and not llm2_error else "Warning",
            "status_kind": "ok" if llm2_text and not llm2_error else "warn",
            "message": (
                "LLM2 used the parsed logs, manual validator output, LLM1 output, and pattern findings together to write the root-cause analysis."
                if llm2_text and not llm2_error
                else f"LLM2 did not complete cleanly: {llm2_error or 'No analysis text was produced.'}"
            ),
            "metrics": [
                "Inputs: parsed logs + manual GE + LLM1 + patterns",
            ],
            "preview_title": "Root-cause analysis",
            "preview_table": None,
            "preview_text": llm2_text or "No LLM2 analysis available.",
        }
    )

    stages.append(
        {
            "title": "Dashboard Renderer",
            "status": "Complete" if DASHBOARD_OUTPUT.exists() else "Warning",
            "status_kind": "ok" if DASHBOARD_OUTPUT.exists() else "warn",
            "message": "The final dashboard is generated from pipeline_report.html after the earlier structured stages finish.",
            "metrics": [
                f"HTML output: {DASHBOARD_OUTPUT.name}",
                f"Ready: {'yes' if DASHBOARD_OUTPUT.exists() else 'no'}",
            ],
            "preview_title": None,
            "preview_table": None,
            "preview_text": None,
        }
    )
    return stages


def _render(**context):
    defaults = {
        "message": "",
        "status_kind": "ok",
        "summary": None,
        "stages": [],
        "dashboard_ready": False,
        "custom_rules": "",
        "use_custom_rules": False,
        "run_llm1": True,
        "run_llm2": True,
    }
    defaults.update(context)
    return render_template_string(PAGE, **defaults)


@app.route("/dashboard")
def dashboard():
    if not DASHBOARD_OUTPUT.exists():
        return "Dashboard not generated yet.", 404
    return send_file(DASHBOARD_OUTPUT)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return _render()

    uploaded = request.files.get("log_file")
    use_custom_rules = request.form.get("use_custom_rules") == "on"
    custom_rules = request.form.get("custom_rules", "").strip()
    run_llm1 = request.form.get("run_llm1") == "on"
    run_llm2 = request.form.get("run_llm2") == "on"

    if not uploaded or not uploaded.filename:
        return _render(
            message="Please choose a .csv or .log file before submitting.",
            status_kind="err",
            use_custom_rules=use_custom_rules,
            custom_rules=custom_rules,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
        )

    if not _allowed_file(uploaded.filename):
        return _render(
            message="Unsupported file type. Please upload a .csv or .log file.",
            status_kind="err",
            use_custom_rules=use_custom_rules,
            custom_rules=custom_rules,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
        )

    if use_custom_rules and not custom_rules:
        return _render(
            message="Custom rules are enabled, but the rules JSON box is empty.",
            status_kind="err",
            use_custom_rules=use_custom_rules,
            custom_rules=custom_rules,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
        )

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(uploaded.filename) or "uploaded_file"
        upload_path = UPLOAD_DIR / filename
        uploaded.save(upload_path)

        pipeline_result = run_pipeline(
            input_path=upload_path,
            custom_rules=custom_rules if use_custom_rules else None,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
            strict=False,
        )
        parsed_df = pipeline_result["parsed_df"]
        validation = pipeline_result["validation"]
        llm1_error = pipeline_result["llm1_error"]
        patterns = pipeline_result["patterns"]
        llm2_text = pipeline_result["llm2_text"]
        llm2_error = pipeline_result["llm2_error"]

        if not llm2_text and LLM2_OUTPUT.exists():
            llm2_text = LLM2_OUTPUT.read_text(encoding="utf-8")

        stats = validation.get("statistics", {})
        summary = {
            "filename": filename,
            "rows": len(parsed_df),
            "manual_total": stats.get("evaluated_expectations", 0),
            "manual_passed": stats.get("successful_expectations", 0),
            "llm_total": 0,
            "llm_passed": 0,
        }
        llm1_stats = _llm_stats(LLM1_RESULTS_OUTPUT) if run_llm1 and not llm1_error else None
        if llm1_stats:
            summary["llm_total"] = llm1_stats.get("evaluated_expectations", 0)
            summary["llm_passed"] = llm1_stats.get("successful_expectations", 0)
        stages = _build_stage_cards(
            parsed_df=parsed_df,
            validation=validation,
            llm1_error=llm1_error,
            patterns=patterns,
            llm2_text=llm2_text,
            llm2_error=llm2_error,
        )

        top_message = "Pipeline completed. Each stage below shows what the system did with your uploaded file."
        top_status = "ok"
        if llm1_error or llm2_error:
            top_message = (
                "Pipeline completed with partial warnings. The manual pipeline and dashboard ran, "
                "but one or more LLM stages need attention."
            )
            top_status = "warn"

        return _render(
            message=top_message,
            status_kind=top_status,
            summary=summary,
            stages=stages,
            dashboard_ready=DASHBOARD_OUTPUT.exists(),
            custom_rules=custom_rules,
            use_custom_rules=use_custom_rules,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
        )
    except Exception as exc:
        return _render(
            message=f"Pipeline failed: {exc}",
            status_kind="err",
            custom_rules=custom_rules,
            use_custom_rules=use_custom_rules,
            run_llm1=run_llm1,
            run_llm2=run_llm2,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
