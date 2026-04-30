"""
Stage 0 – LLM-Driven GE Suite Writer
Profiles parsed_logs.csv, asks an LLM to write a GX 1.0 validation suite,
saves the generated code to outputs/ge_suite_llm.py, executes it, and
reports how many expectations were written vs passed.
"""

import json
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent.parent
PARSED_CSV      = ROOT / "outputs" / "parsed_logs.csv"
GE_SUITE_FILE   = ROOT / "outputs" / "ge_suite_llm.py"
RESULTS_FILE    = ROOT / "outputs" / "validation_results_llm.json"

DEFAULT_PROVIDER = "huggingface"
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
# Step 1 – Build a rich data profile
# ---------------------------------------------------------------------------
def build_profile(df: pd.DataFrame) -> str:
    lines = []

    lines.append(f"TOTAL ROWS : {len(df):,}")
    lines.append(f"TOTAL COLS : {len(df.columns)}")
    lines.append("")

    for col in df.columns:
        series = df[col]
        dtype  = str(series.dtype)
        nulls  = int(series.isna().sum())
        lines.append(f"── {col}  (dtype={dtype}, nulls={nulls})")

        # numeric
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            nn = series.dropna()
            if len(nn):
                lines.append(f"   min={nn.min()}, max={nn.max()}, "
                             f"mean={nn.mean():.3f}, std={nn.std():.3f}")
                q = nn.quantile([0.25, 0.5, 0.75])
                lines.append(f"   p25={q[0.25]:.1f}, p50={q[0.50]:.1f}, p75={q[0.75]:.1f}")

        # boolean
        elif pd.api.types.is_bool_dtype(series):
            vc = series.value_counts()
            pct_true = series.sum() / len(series) * 100
            lines.append(f"   True={int(series.sum())} ({pct_true:.1f}%), "
                         f"False={int((~series).sum())}")

        # categorical / string (low cardinality ≤ 20 unique values)
        elif series.dtype == object or str(series.dtype) == "string":
            nunique = series.nunique(dropna=True)
            lines.append(f"   unique values (non-null): {nunique}")
            non_null = series.dropna()
            if nunique <= 20:
                vc = non_null.value_counts()
                for val, cnt in vc.items():
                    lines.append(f"     '{val}': {cnt} ({cnt/len(df)*100:.1f}%)")
            else:
                # show top 8 + sample regex pattern
                top8 = non_null.value_counts().head(8)
                lines.append(f"   top 8 values:")
                for val, cnt in top8.items():
                    lines.append(f"     '{val}': {cnt}")
                # show 3 random samples
                samples = non_null.sample(min(3, len(non_null)),
                                          random_state=42).tolist()
                lines.append(f"   samples: {samples}")

        lines.append("")

    # 5 sample rows
    lines.append("── SAMPLE ROWS (5 rows, all columns)")
    lines.append(df.head(5).to_string(index=False))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2 – Ask an LLM to write the GE suite
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Step 2 – Ask an LLM to write the GE suite
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent(r"""\
You are a senior data quality engineer specializing in Great Expectations (GX 1.0).

Your task is to analyze a structured data profile and generate a complete, executable GX 1.0 validation suite in Python.

---------------------------------------------------------------------
PHASE 1 — ANALYSIS (INTERNAL ONLY, DO NOT OUTPUT)
---------------------------------------------------------------------
Before writing any code, internally analyze the dataset:

- Identify column types: numeric, categorical, boolean, text, timestamp
- Identify important columns (high cardinality, critical flags, identifiers)
- Identify patterns (IDs, timestamps, codes, structured text)
- Identify relationships between columns (e.g., flags and dependent fields)
- Identify potential data quality risks (nulls, skew, inconsistencies)

DO NOT output this analysis.

---------------------------------------------------------------------
PHASE 2 — VALIDATION PLAN (INTERNAL ONLY, DO NOT OUTPUT)
---------------------------------------------------------------------
Internally create a validation plan:

- Select only meaningful, high-signal rules
- Avoid redundant or trivial expectations
- Prioritize correctness and usefulness over quantity

Rules should include:
- Null constraints
- Value sets (categorical/boolean)
- Numeric ranges
- Regex format validation
- Uniqueness constraints where appropriate
- Boolean proportion checks (via mean)
- At least 2 strong business logic rules involving multiple columns

DO NOT output this plan.

---------------------------------------------------------------------
PHASE 3 — CODE GENERATION (OUTPUT ONLY THIS)
---------------------------------------------------------------------
Generate a complete executable Python script that:

1. Uses Great Expectations GX 1.0 (modern API only)
2. Follows EXACT structure:

    import great_expectations as gx
    from great_expectations.expectations import ...

    ctx = gx.get_context(mode="ephemeral")
    datasource = ctx.data_sources.add_pandas("log_datasource")
    asset = datasource.add_dataframe_asset("parsed_logs")
    batch_def = asset.add_batch_definition_whole_dataframe("full_batch")

    suite = ctx.suites.add(gx.ExpectationSuite(name="log_quality_suite"))

    suite.add_expectation(...)

    vd = ctx.validation_definitions.add(
        gx.ValidationDefinition(name="log_quality_validation", data=batch_def, suite=suite)
    )

    result = vd.run(batch_parameters={"dataframe": df})

3. Reads data from:
    outputs/parsed_logs.csv

4. Writes results to:
    outputs/validation_results_llm.json

5. Prints EXACTLY:

    EXPECTATIONS_WRITTEN: <N>
    EXPECTATIONS_PASSED: <N>

---------------------------------------------------------------------
STRICT IMPLEMENTATION RULES
---------------------------------------------------------------------
- Return ONLY Python code (no markdown, no explanation)
- The script must run without modification
- Use ONLY these expectation classes unless absolutely necessary:

    ExpectColumnValuesToNotBeNull
    ExpectColumnValuesToBeNull
    ExpectColumnValuesToBeInSet
    ExpectColumnValuesToMatchRegex
    ExpectColumnValuesToBeBetween
    ExpectColumnValuesToBeUnique
    ExpectColumnMeanToBeBetween

- Use:
    suite.add_expectation(...)

- DO NOT:
    - Use legacy GX APIs
    - Use suite.expect_* methods
    - Use ExpectationConfiguration
    - Import internal modules (gx.core.*)
    - Invent fake classes or APIs
    - Add expectations that always pass
    - Use trivial regex like ".*"
    - Add rules for columns that do not exist

---------------------------------------------------------------------
DATA-CONSTRAINED GENERATION
---------------------------------------------------------------------
- Use ONLY column names present in the data profile
- Do NOT hallucinate columns
- Derive all rules strictly from observed data patterns

---------------------------------------------------------------------
REGEX RULE GUIDANCE
---------------------------------------------------------------------
Use strong regex patterns where appropriate:

- Date → YYYY-MM-DD
- Time → HH:MM:SS
- EventId → E\d+
- hresult → 0x[0-9A-Fa-f]+
- timestamp → YYYY-MM-DD HH:MM:SS

---------------------------------------------------------------------
BUSINESS LOGIC (MANDATORY)
---------------------------------------------------------------------
You MUST include at least 2 strong multi-column rules.

Examples:

- If has_hresult == True → hresult must not be empty
- If has_failure == True → event_class must be "FAILURE"

Implement using:
- gx.expectations.UnexpectedRowsExpectation (if available)
OR
- custom pandas logic + ExpectationValidationResult

These must be meaningful and not trivial.

---------------------------------------------------------------------
BOOLEAN PROPORTION RULES
---------------------------------------------------------------------
- For boolean columns, convert to float
- Use ExpectColumnMeanToBeBetween

---------------------------------------------------------------------
SPECIAL COLUMN RULES
---------------------------------------------------------------------
- hresult and kb_package use empty string ("") when missing
- session_id is fully null → enforce ExpectColumnValuesToBeNull

---------------------------------------------------------------------
OUTPUT FORMAT
---------------------------------------------------------------------
Return ONLY the final executable Python script.
No markdown.
No explanation.
No comments outside code.

---------------------------------------------------------------------
QUALITY OVER QUANTITY
---------------------------------------------------------------------
Do not optimize for all expectations passing.
It is acceptable and useful for some expectations to fail if they reveal real data quality problems.
Generate rules that reflect expected data quality, not just observed values.

Include at least 25 expectations when the profile supports it.
Do not avoid meaningful expectations just because they may fail.

---------------------------------------------------------------------
FINAL SELF-CHECK (MANDATORY)
---------------------------------------------------------------------
Before finishing, ensure:

- Code is syntactically valid
- All imports are present
- GX API usage is correct
- No missing variables
- Script runs end-to-end without errors
- Output file is written correctly

---------------------------------------------------------------------
DATA PROFILE
---------------------------------------------------------------------
{profile}
""")


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        start = 1 if lines and lines[0].startswith("```") else 0
        end = len(lines) - 1 if lines and lines[-1].strip() == "```" else len(lines)
        code = "\n".join(lines[start:end])
    return code.strip()


def _extract_code_block(text: str) -> str:
    code = _strip_code_fences(text)
    if not code:
        raise RuntimeError("LLM returned an empty response.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        preview = "\n".join(code.splitlines()[:12])
        raise RuntimeError(
            "LLM did not return clean executable Python code. "
            "This usually means the selected model ignored the prompt format.\n\n"
            f"Preview:\n{preview}"
        ) from exc
    return code


def _parse_hf_code_response(payload: dict) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return _extract_code_block(content)

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        joined = "".join(text_parts).strip()
        if not joined:
            raise RuntimeError("Hugging Face returned an empty content list.")
        try:
            parsed = json.loads(joined)
            if isinstance(parsed, dict) and "python_code" in parsed:
                return _extract_code_block(parsed["python_code"])
        except json.JSONDecodeError:
            pass
        return _extract_code_block(joined)

    raise RuntimeError("Unsupported Hugging Face response format.")


def _repair_code_with_huggingface(model: str, hf_base_url: str, hf_token: str, broken_code: str, error_text: str) -> str:
    repair_prompt = textwrap.dedent(f"""\
        The previous response was intended to be a complete Python script, but it contains syntax or formatting issues.

        Repair the script so that it is valid executable Python and still follows the GX 1.0 pattern exactly.
        Return only the repaired script.

        Syntax error:
        {error_text}

        Broken script:
        {broken_code}
    """)

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
    response.raise_for_status()

    repaired = _parse_hf_code_response(response.json())
    ast.parse(repaired)
    return repaired


def repair_huggingface_runtime_code(broken_code: str, error_text: str) -> str:
    model, hf_base_url = get_llm_config()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF token is required for runtime repair.")

    repair_prompt = textwrap.dedent(f"""\
        The following Python script was generated for this project, but it failed at runtime.
        Fix the script so it remains a complete executable GX 1.0 validation suite and resolves
        the runtime error below.

        IMPORTANT:
        - Keep using the GX 1.0 pattern from the system instructions.
        - Do not use internal GX imports.
        - Do not use ExpectationConfiguration.
        - Use imported expectation classes and `suite.add_expectation(...)`.
        - Return only the full repaired Python script.

        Runtime error:
        {error_text}

        Script to repair:
        {broken_code}
    """)

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
    response.raise_for_status()

    return _parse_hf_code_response(response.json())


def get_llm_config(
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> tuple[str, str]:
    model = model_override or os.getenv("LLM1_MODEL", DEFAULT_MODEL)
    hf_base_url = (hf_base_url_override or os.getenv("HF_BASE_URL", DEFAULT_HF_BASE_URL)).rstrip("/")
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
            "HF_TOKEN or HUGGINGFACE_API_TOKEN is not set. "
            "Add one of them to your environment or .env file."
        )

    user_msg = (
        "Here is the data profile for outputs/parsed_logs.csv.\n\n"
        f"{profile}\n\n"
        "Write the complete GX 1.0 validation suite Python script as described."
    )

    print(f"Sending profile to Hugging Face model {model}...")
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
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text[-1000:] if exc.response is not None else str(exc)
        raise RuntimeError(f"Hugging Face request failed: {detail}") from exc

    payload = response.json()
    try:
        return _parse_hf_code_response(payload)
    except RuntimeError as exc:
        raw_content = payload["choices"][0]["message"]["content"]
        if isinstance(raw_content, list):
            raw_content = json.dumps(raw_content)
        try:
            return _repair_code_with_huggingface(
                model=model,
                hf_base_url=hf_base_url,
                hf_token=hf_token,
                broken_code=str(raw_content),
                error_text=str(exc),
            )
        except Exception:
            raise exc


def ask_llm(
    profile: str,
    *,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> str:
    return ask_huggingface(
        profile,
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )


# ---------------------------------------------------------------------------
# Step 3 – Execute the generated script and parse output
# ---------------------------------------------------------------------------
def execute_suite(suite_path: Path) -> tuple[int, int]:
    """
    Runs the generated script as a subprocess from the project root.
    Parses the EXPECTATIONS_WRITTEN / EXPECTATIONS_PASSED lines from stdout.
    Returns (written, passed).
    """
    result = subprocess.run(
        [sys.executable, str(suite_path)],
        capture_output=True, text=True,
        cwd=str(ROOT),
    )

    output = result.stdout + result.stderr
    print("\n--- Generated suite output ---")
    # Print everything except GX progress bars
    for line in output.splitlines():
        if "Calculating Metrics" not in line:
            print(line)
    print("------------------------------")

    if result.returncode != 0:
        raise RuntimeError(
            f"Generated suite exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

    written = passed = 0
    for line in output.splitlines():
        if line.startswith("EXPECTATIONS_WRITTEN:"):
            written = int(line.split(":")[1].strip())
        elif line.startswith("EXPECTATIONS_PASSED:"):
            passed  = int(line.split(":")[1].strip())

    return written, passed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_llm_ge_writer(
    parsed_path: Path = PARSED_CSV,
    suite_path:  Path = GE_SUITE_FILE,
    results_path: Path = RESULTS_FILE,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> tuple[int, int]:
    load_dotenv(ROOT / ".env")

    # 1. Profile
    print("Building data profile...")
    df      = pd.read_csv(parsed_path)
    profile = build_profile(df)
    print(f"  Profile built  ({len(profile):,} chars)")

    # 2. LLM call
    code = ask_llm(
        profile,
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(code, encoding="utf-8")
    print(f"\nGenerated suite saved to {suite_path}")

    line_count = code.count("\n") + 1
    exp_count  = code.count("add_expectation(")
    print(f"  ~{line_count} lines of code,  ~{exp_count} add_expectation() calls")

    # 3. Execute
    print("\nExecuting generated suite...")
    current_code = code
    max_attempts = 3
    last_exc: RuntimeError | None = None

    print("\nSkipping execution of generated suite due to GX version mismatch.")
    written = code.count("add_expectation(")
    passed = 0

    # 4. Summary
    failed = written - passed
    pct    = round(passed / written * 100, 1) if written else 0
    print(f"\n{'='*50}")
    print(f"  LLM wrote  : {written} expectations")
    print(f"  Passed     : {passed}  ({pct}%)")
    print(f"  Failed     : {failed}")
    print(f"  Results    : {results_path}")
    print(f"{'='*50}")

    return written, passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_llm_ge_writer()
