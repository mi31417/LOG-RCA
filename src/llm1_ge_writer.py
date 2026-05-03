"""
Stage 0 – LLM-Driven GE Suite Writer

Reads outputs/parsed_logs.csv, builds a compact profile, asks Hugging Face
to generate a GX 1.0 validation suite, executes it, and runs local diagnostics.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent

PARSED_CSV = ROOT / "outputs" / "parsed_logs.csv"
GE_SUITE_FILE = ROOT / "outputs" / "ge_suite_llm.py"
RESULTS_FILE = ROOT / "outputs" / "validation_results_llm.json"
DIAGNOSTICS_FILE = ROOT / "outputs" / "llm_diagnostics.json"
RAW_LLM_OUTPUT_FILE = ROOT / "outputs" / "raw_llm1_response.txt"

DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


HF_CODE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "gx_python_script",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "python_code": {
                    "type": "string",
                    "description": "A complete executable Python script.",
                }
            },
            "required": ["python_code"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Compact profile
# ---------------------------------------------------------------------------
def build_profile(df: pd.DataFrame) -> str:
    lines = []

    lines.append(f"ROWS: {len(df)}")
    lines.append(f"COLUMNS: {len(df.columns)}")
    lines.append("")

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        nulls = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        lines.append(f"COLUMN: {col}")
        lines.append(f"  dtype: {dtype}")
        lines.append(f"  nulls: {nulls}")
        lines.append(f"  unique: {unique}")

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric):
                lines.append(
                    f"  numeric_summary: min={numeric.min()}, max={numeric.max()}, "
                    f"mean={numeric.mean():.3f}, median={numeric.median():.3f}"
                )

        else:
            non_null = series.dropna().astype(str)
            if len(non_null):
                top_values = non_null.value_counts().head(5)
                top_str = ", ".join([f"{repr(k)}:{int(v)}" for k, v in top_values.items()])
                lines.append(f"  top_values: {top_str}")

        lines.append("")

    lines.append("SAMPLE ROWS:")
    lines.append(df.head(3).to_string(index=False))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shorter prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(
    r"""\
You are a senior data quality analyst and Great Expectations GX 1.0 engineer.

Your task is to generate ONE complete executable Python script that validates:
outputs/parsed_logs.csv

Return ONLY Python code. No markdown. No explanation.

---------------------------------------------------------------------
GOAL
---------------------------------------------------------------------
Create 15–25 meaningful validation expectations.

Do NOT optimize for 100% pass.
Some failures are GOOD because they reveal data quality or parsing issues.

Think like a data analyst defining expected contracts — not just describing data.

---------------------------------------------------------------------
CRITICAL DATA CONTRACTS (VERY IMPORTANT)
---------------------------------------------------------------------
These must be STRICT (no mostly):

- LineId must not be null and must be unique
- Content must not be null
- EventId must not be null and should follow a code pattern if present
- Date, Time, or timestamp must not be null if present
- EventTemplate must not be null if present
- param_count must be consistent with EventTemplate if both exist

---------------------------------------------------------------------
SEMANTIC (MULTI-COLUMN) RULES
---------------------------------------------------------------------
You MUST create derived dq_* columns for semantic checks.

Examples of REQUIRED logic:

1. Failure flag vs content:
   if has_failure is true → Content must contain failure keywords

2. Warning flag vs content:
   if has_warning is true → Content must contain warning keywords

3. HRESULT logic:
   if has_hresult is true → hresult must not be empty

4. Template consistency:
   param_count must equal number of <*> in EventTemplate

---------------------------------------------------------------------
IMPORTANT RULE FOR dq_* COLUMNS
---------------------------------------------------------------------
DO NOT use "mostly" for dq_* columns.

Use STRICT validation:

ExpectColumnValuesToBeInSet(
    column="dq_*",
    value_set=[True]
)

Even ONE False should FAIL the expectation.

---------------------------------------------------------------------
FORMAT RULES
---------------------------------------------------------------------
Add regex checks where meaningful:

- Date → YYYY-MM-DD
- Time → HH:MM:SS
- timestamp → YYYY-MM-DD HH:MM:SS
- EventId → pattern like E123
- hresult → 0x[0-9A-Fa-f]+
- kb_package → KB123

Do NOT use weak regex like .*

---------------------------------------------------------------------
NUMERIC / DISTRIBUTION RULES
---------------------------------------------------------------------
- Add range checks for numeric columns
- Add mean checks for boolean columns
- DO NOT use exact observed min/max as boundaries

---------------------------------------------------------------------
AVOID THESE MISTAKES
---------------------------------------------------------------------
- Do NOT create rules that always pass
- Do NOT use observed values directly for value_set blindly
- Do NOT duplicate null checks
- Do NOT mark frequently repeating columns as unique
- Do NOT use mostly=0.95 to hide failures
- Do NOT invent columns

---------------------------------------------------------------------
REQUIRED GX STRUCTURE (MANDATORY)
---------------------------------------------------------------------

import json
import pandas as pd
import great_expectations as gx
from great_expectations.expectations import ...

df = pd.read_csv("outputs/parsed_logs.csv")

# Create dq_* columns BEFORE GX setup

ctx = gx.get_context(mode="ephemeral")
datasource = ctx.data_sources.add_pandas("log_datasource")
asset = datasource.add_dataframe_asset("parsed_logs")
batch_def = asset.add_batch_definition_whole_dataframe("full_batch")

suite = ctx.suites.add(gx.ExpectationSuite(name="log_quality_suite"))

suite.add_expectation(...)

vd = ctx.validation_definitions.add(
    gx.ValidationDefinition(
        name="log_quality_validation",
        data=batch_def,
        suite=suite
    )
)

result = vd.run(batch_parameters={"dataframe": df})

raw = result.to_json_dict()

with open("outputs/validation_results_llm.json", "w") as f:
    json.dump(raw, f, indent=2, default=str)

print(f"EXPECTATIONS_WRITTEN: {len(suite.expectations)}")
print(f"EXPECTATIONS_PASSED: {sum(1 for r in result.results if r.success)}")

---------------------------------------------------------------------
FINAL CHECK BEFORE OUTPUT
---------------------------------------------------------------------
Ensure:
- Valid Python code
- At least 15 expectations
- dq_* rules are STRICT (no mostly)
- All referenced columns exist or are inside if checks
- Output JSON is written correctly

Return ONLY the Python script.
"""
)
# ---------------------------------------------------------------------------
# LLM response handling
# ---------------------------------------------------------------------------
def _save_raw_output(text: str) -> None:
    RAW_LLM_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_LLM_OUTPUT_FILE.write_text(str(text), encoding="utf-8")


def _strip_code_fences(code: str) -> str:
    code = str(code).strip()

    if code.startswith("```"):
        lines = code.splitlines()
        start = 1 if lines and lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
        code = "\n".join(lines[start:end])

    return code.strip()


def _extract_code_block(text: str) -> str:
    _save_raw_output(text)

    code = _strip_code_fences(text)

    if code.startswith("{"):
        try:
            parsed = json.loads(code)
            if isinstance(parsed, dict) and "python_code" in parsed:
                code = _strip_code_fences(parsed["python_code"])
        except json.JSONDecodeError:
            pass

    if not code:
        raise RuntimeError("LLM returned an empty response.")

    try:
        ast.parse(code)
    except SyntaxError as exc:
        preview = "\n".join(code.splitlines()[:25])
        raise RuntimeError(
            "LLM did not return valid executable Python code.\n"
            f"Raw output saved to: {RAW_LLM_OUTPUT_FILE}\n\n"
            f"Preview:\n{preview}"
        ) from exc

    return code


def _parse_hf_code_response(payload: dict) -> str:
    _save_raw_output(json.dumps(payload, indent=2, default=str))

    try:
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected Hugging Face response format. Raw response saved to {RAW_LLM_OUTPUT_FILE}"
        ) from exc

    if isinstance(content, str):
        return _extract_code_block(content)

    if isinstance(content, dict):
        if "python_code" in content:
            return _extract_code_block(content["python_code"])
        if "content" in content:
            return _extract_code_block(content["content"])

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item.get("text", ""))

        joined = "".join(text_parts).strip()
        return _extract_code_block(joined)

    raise RuntimeError(
        f"Unsupported HF response content type: {type(content)}. "
        f"Raw response saved to {RAW_LLM_OUTPUT_FILE}"
    )


def get_llm_config(
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> tuple[str, str]:
    model = model_override or os.getenv("LLM1_MODEL", DEFAULT_MODEL)
    hf_base_url = (
        hf_base_url_override or os.getenv("HF_BASE_URL", DEFAULT_HF_BASE_URL)
    ).rstrip("/")
    return model, hf_base_url


def ask_huggingface(
    profile: str,
    *,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> str:
    model, hf_base_url = get_llm_config(
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN or HUGGINGFACE_API_TOKEN is not set. Add one of them to your .env file."
        )

    user_msg = (
        "Data profile for outputs/parsed_logs.csv:\n\n"
        f"{profile}\n\n"
        "Generate the complete GX 1.0 Python script now."
    )

    print(f"Sending profile to Hugging Face model {model}...")
    print(f"Prompt size: {len(SYSTEM_PROMPT) + len(user_msg):,} characters")

    try:
        response = requests.post(
            f"{hf_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "max_tokens": 8192,
                "temperature": 0.1,
                "response_format": HF_CODE_SCHEMA,
            },
            timeout=600,
        )

        print(f"Hugging Face status: {response.status_code}")

        if response.status_code != 200:
            _save_raw_output(response.text)
            raise RuntimeError(
                f"Hugging Face request failed with status {response.status_code}. "
                f"Raw response saved to {RAW_LLM_OUTPUT_FILE}. "
                f"Details: {response.text[-1200:]}"
            )

        payload = response.json()

    except requests.RequestException as exc:
        raise RuntimeError(f"Hugging Face request failed: {exc}") from exc

    code = _parse_hf_code_response(payload)
    print("LLM code parsed successfully.")
    return code


# ---------------------------------------------------------------------------
# Runtime repair
# ---------------------------------------------------------------------------
def repair_huggingface_runtime_code(
    broken_code: str,
    error_text: str,
    *,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> str:
    model, hf_base_url = get_llm_config(
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

    if not hf_token:
        raise RuntimeError("HF token is required for runtime repair.")

    repair_prompt = textwrap.dedent(
        f"""\
        Repair this generated GX 1.0 script.

        Return only executable Python code.

        Rules:
        - Keep reading outputs/parsed_logs.csv
        - Keep writing outputs/validation_results_llm.json
        - Do not use row_condition or condition_parser
        - Use derived dq_* columns for multi-column checks
        - Use suite.add_expectation(...)

        Runtime error:
        {error_text}

        Broken script:
        {broken_code}
        """
    )

    response = requests.post(
        f"{hf_base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": repair_prompt},
            ],
            "stream": False,
            "max_tokens": 8192,
            "temperature": 0.0,
            "response_format": HF_CODE_SCHEMA,
        },
        timeout=600,
    )

    print(f"Hugging Face repair status: {response.status_code}")

    if response.status_code != 200:
        _save_raw_output(response.text)
        raise RuntimeError(
            f"Hugging Face repair failed with status {response.status_code}. "
            f"Raw response saved to {RAW_LLM_OUTPUT_FILE}. "
            f"Details: {response.text[-1200:]}"
        )

    repaired = _parse_hf_code_response(response.json())
    ast.parse(repaired)

    return repaired


# ---------------------------------------------------------------------------
# Execute generated suite
# ---------------------------------------------------------------------------
def execute_suite(suite_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [sys.executable, str(suite_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    output = result.stdout + result.stderr

    print("\n--- Generated suite output ---")
    for line in output.splitlines():
        if "Calculating Metrics" not in line:
            print(line)
    print("------------------------------")

    if result.returncode != 0:
        raise RuntimeError(
            f"Generated suite exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )

    written = 0
    passed = 0

    for line in output.splitlines():
        if line.startswith("EXPECTATIONS_WRITTEN:"):
            written = int(line.split(":", 1)[1].strip())
        elif line.startswith("EXPECTATIONS_PASSED:"):
            passed = int(line.split(":", 1)[1].strip())

    if written == 0:
        raise RuntimeError(
            "Generated suite ran, but EXPECTATIONS_WRITTEN was 0. "
            f"Check raw output at {RAW_LLM_OUTPUT_FILE}"
        )

    return written, passed


# ---------------------------------------------------------------------------
# Local diagnostics
# ---------------------------------------------------------------------------
def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.lower().str.strip().isin(
        ["true", "1", "yes", "y"]
    )


def run_local_diagnostics(
    parsed_path: Path = PARSED_CSV,
    output_path: Path = DIAGNOSTICS_FILE,
) -> dict:
    df = pd.read_csv(parsed_path)

    diagnostics: dict[str, object] = {
        "diagnostic_source": "local_deterministic_checks",
        "purpose": "Catch hidden semantic mismatches if the LLM suite is too conservative.",
        "total_rows": int(len(df)),
        "checks": {},
    }

    checks = diagnostics["checks"]

    if {"has_failure", "Content"}.issubset(df.columns):
        flag = _bool_series(df["has_failure"])
        content_match = df["Content"].fillna("").astype(str).str.contains(
            r"fail|failed|failure|error|exception|fatal|critical",
            case=False,
            na=False,
            regex=True,
        )
        valid = (~flag) | content_match
        checks["dq_failure_flag_matches_content"] = {
            "pass_rate": round(float(valid.mean()), 4),
            "failed_rows": int((~valid).sum()),
        }

    if {"has_warning", "Content"}.issubset(df.columns):
        flag = _bool_series(df["has_warning"])
        content_match = df["Content"].fillna("").astype(str).str.contains(
            r"warn|warning",
            case=False,
            na=False,
            regex=True,
        )
        valid = (~flag) | content_match
        checks["dq_warning_flag_matches_content"] = {
            "pass_rate": round(float(valid.mean()), 4),
            "failed_rows": int((~valid).sum()),
        }

    if {"has_hresult", "hresult"}.issubset(df.columns):
        flag = _bool_series(df["has_hresult"])
        has_value = df["hresult"].fillna("").astype(str).str.strip() != ""
        valid = (~flag) | has_value
        checks["dq_hresult_flag_has_value"] = {
            "pass_rate": round(float(valid.mean()), 4),
            "failed_rows": int((~valid).sum()),
        }

    if {"param_count", "EventTemplate"}.issubset(df.columns):
        expected_count = df["EventTemplate"].fillna("").astype(str).str.count(r"<\*>")
        actual_count = pd.to_numeric(df["param_count"], errors="coerce").fillna(-1)
        valid = actual_count.eq(expected_count)
        checks["dq_param_count_matches_event_template"] = {
            "pass_rate": round(float(valid.mean()), 4),
            "failed_rows": int((~valid).sum()),
        }

    if {"has_failure", "event_class"}.issubset(df.columns):
        flag = _bool_series(df["has_failure"])
        event_class = df["event_class"].fillna("").astype(str).str.upper()
        valid = (~flag) | event_class.eq("FAILURE")
        checks["dq_failure_flag_matches_event_class"] = {
            "pass_rate": round(float(valid.mean()), 4),
            "failed_rows": int((~valid).sum()),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    print("\n--- Local deterministic diagnostics ---")
    if checks:
        for name, payload in checks.items():
            print(
                f"{name}: pass_rate={payload['pass_rate']:.2%}, "
                f"failed_rows={payload['failed_rows']}"
            )
    else:
        print("No applicable local diagnostics found for this schema.")
    print(f"Diagnostics saved to {output_path}")
    print("--------------------------------------")

    return diagnostics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_llm_ge_writer(
    parsed_path: Path = PARSED_CSV,
    suite_path: Path = GE_SUITE_FILE,
    results_path: Path = RESULTS_FILE,
    diagnostics_path: Path = DIAGNOSTICS_FILE,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> tuple[int, int]:
    load_dotenv(ROOT / ".env")

    parsed_path = Path(parsed_path)
    suite_path = Path(suite_path)
    diagnostics_path = Path(diagnostics_path)

    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed CSV not found: {parsed_path}")

    print("Building compact data profile...")
    df = pd.read_csv(parsed_path)
    profile = build_profile(df)
    print(f"  Profile built ({len(profile):,} chars)")

    code = ask_huggingface(
        profile,
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )

    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(code, encoding="utf-8")

    print(f"\nGenerated suite saved to {suite_path}")

    line_count = code.count("\n") + 1
    exp_count = code.count("add_expectation(")

    print(f"  ~{line_count} lines of code")
    print(f"  ~{exp_count} add_expectation() calls")

    if exp_count == 0:
        raise RuntimeError(
            "Generated code contains 0 add_expectation() calls. "
            f"Check raw output at {RAW_LLM_OUTPUT_FILE}"
        )

    print("\nExecuting generated suite...")

    current_code = code
    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            written, passed = execute_suite(suite_path)
            break

        except RuntimeError as exc:
            if attempt >= max_attempts:
                raise

            print(f"\nAttempt {attempt} failed. Asking Hugging Face for one runtime repair...")

            repaired_code = repair_huggingface_runtime_code(
                current_code,
                str(exc),
                model_override=model_override,
                hf_base_url_override=hf_base_url_override,
            )

            suite_path.write_text(repaired_code, encoding="utf-8")
            current_code = repaired_code

            print("Repaired suite saved. Re-running...")

    diagnostics = run_local_diagnostics(
        parsed_path=parsed_path,
        output_path=diagnostics_path,
    )

    if written >= 12 and passed == written:
        print(
            "\nWARNING: LLM-generated suite passed 100%. "
            "This may mean the rules are conservative."
        )
        print(f"Check diagnostics here: {diagnostics_path}")

    failed = written - passed
    pct = round(passed / written * 100, 1) if written else 0

    print(f"\n{'=' * 50}")
    print(f"  LLM wrote     : {written} expectations")
    print(f"  Passed        : {passed} ({pct}%)")
    print(f"  Failed        : {failed}")
    print(f"  GE results    : {RESULTS_FILE}")
    print(f"  Diagnostics   : {diagnostics_path}")
    print(f"{'=' * 50}")

    return written, passed


if __name__ == "__main__":
    run_llm_ge_writer()
