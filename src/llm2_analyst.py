"""
Stage 5 – LLM Analyst
Sends parsed logs, manual validation, LLM1 output, and pattern findings to a
remote LLM for root-cause analysis.
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).resolve().parent.parent
PARSED_FILE      = ROOT / "outputs" / "parsed_logs.csv"
PATTERNS_FILE    = ROOT / "outputs" / "patterns_found.json"
VALIDATION_FILE  = ROOT / "outputs" / "validation_results.json"
LLM1_FILE        = ROOT / "outputs" / "validation_results_llm.json"
LLM1_SUITE_FILE  = ROOT / "outputs" / "ge_suite_llm.py"
OUTPUT_FILE      = ROOT / "outputs" / "analysis_report.txt"

DEFAULT_PROVIDER = "huggingface"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-480B-A35B-Instruct:fastest"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"

# ---------------------------------------------------------------------------
# Improved Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior Windows log forensics, data quality, and systems reliability engineer.

You are analyzing outputs from a multi-stage log quality pipeline:
1. parsed_logs.csv summary
2. manual Great Expectations validation results
3. LLM-generated Great Expectations rules/results
4. dynamic pattern findings

Your job is to produce a technical root-cause analysis based ONLY on the evidence provided.

IMPORTANT REASONING RULES:
- Use evidence-first reasoning.
- Treat LLM1 generated rules as diagnostic signals, NOT ground truth.
- Do not invent event IDs, percentages, counts, filenames, KB numbers, HRESULTs, or components.
- If evidence is weak, say "insufficient evidence" instead of guessing.
- Distinguish clearly between:
  1. directly observed evidence
  2. inferred cause
  3. recommended next step
- Prefer explanations that connect multiple independent signals.
- If multiple root causes are possible, rank them by evidence strength.
- Mention conflicts or gaps in the evidence.

OUTPUT FORMAT:
Produce exactly these sections:

1. EXECUTIVE SUMMARY
   - 4 to 7 bullets.
   - State the most likely root cause.
   - State the strongest evidence.
   - State uncertainty if present.

2. ROOT CAUSE ANALYSIS
   - Explain what is happening operationally.
   - Connect parsed log structure, validation failures, LLM1 rules, and dynamic patterns.
   - Explain the likely Windows mechanism only when supported by evidence.

3. EVIDENCE MATRIX
   Create a table with columns:
   - Evidence Source
   - Signal
   - Why It Matters
   - Supports Which Hypothesis

4. CROSS-PATTERN CORRELATIONS
   - Explain which patterns are connected.
   - Identify symptoms of the same underlying issue.
   - Explain causal chain in plain technical language.

5. SYSTEM HEALTH SCORE
   - Give score out of 10.
   - Justify with counts, validation pass rate, severe failed rules, and pattern strength.
   - Include confidence level: High / Medium / Low.

6. PRIORITIZED FIX RECOMMENDATIONS
   For each recommendation include:
   - Priority
   - Problem addressed
   - Exact action
   - Expected outcome
   - Evidence supporting this fix

7. FAILURE PREDICTION
   - What likely happens if no action is taken.
   - Include timeline only if evidence supports it.
   - Otherwise say timeline cannot be determined from current logs.

8. LIMITATIONS
   - What cannot be concluded from the current data.
   - What additional logs/data would improve confidence.

STYLE:
- Be specific, technical, and direct.
- Avoid generic advice.
- Use exact counts and percentages when provided.
- Do not overstate certainty.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def get_llm2_config(
    provider_override: str | None = None,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> tuple[str, str, str]:
    provider = (provider_override or os.getenv("LLM2_PROVIDER", DEFAULT_PROVIDER)).strip().lower()
    model = model_override or os.getenv("LLM2_MODEL", os.getenv("LLM1_MODEL", DEFAULT_MODEL))
    hf_base_url = (hf_base_url_override or os.getenv("HF_BASE_URL", DEFAULT_HF_BASE_URL)).rstrip("/")
    return provider, model, hf_base_url


# ---------------------------------------------------------------------------
# Evidence extraction helpers
# ---------------------------------------------------------------------------
def _safe_read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _extract_validation_summary(payload: dict, max_failed: int = 20, max_passed: int = 8) -> dict:
    stats = payload.get("statistics", {})
    results = payload.get("results", [])

    failed = []
    passed = []

    for item in results:
        cfg = item.get("expectation_config", {})
        kwargs = cfg.get("kwargs", {})
        result = item.get("result", {})

        row = {
            "type": cfg.get("type"),
            "column": kwargs.get("column"),
            "success": item.get("success"),
            "unexpected_count": result.get("unexpected_count"),
            "unexpected_percent": result.get("unexpected_percent"),
            "observed_value": result.get("observed_value"),
        }

        if item.get("success") is False:
            failed.append(row)
        else:
            passed.append(row)

    failed_sorted = sorted(
        failed,
        key=lambda x: (
            x.get("unexpected_count") is not None,
            x.get("unexpected_count") or 0,
            x.get("unexpected_percent") or 0,
        ),
        reverse=True,
    )

    return {
        "statistics": stats,
        "failed_rule_count": len(failed),
        "passed_rule_count": len(passed),
        "top_failed_rules": failed_sorted[:max_failed],
        "sample_passed_rules": passed[:max_passed],
    }


def _extract_llm1_summary(payload: dict, llm1_suite_path: Path) -> dict:
    stats = payload.get("statistics", {})

    suite_lines = []
    if llm1_suite_path.exists():
        for line in llm1_suite_path.read_text(encoding="utf-8").splitlines():
            if "suite.add_expectation(" in line:
                suite_lines.append(line.strip())

    # Supports both old and enriched LLM1 result formats.
    return {
        "available": True,
        "statistics": stats,
        "summary": payload.get("summary"),
        "key_issues": payload.get("key_issues", []),
        "root_cause_hypotheses": payload.get("root_cause_hypotheses", []),
        "signals_for_llm2_to_verify": payload.get("signals_for_llm2_to_verify", []),
        "limitations": payload.get("limitations", []),
        "generated_rules": payload.get("generated_rules", suite_lines[:60]),
        "generated_rule_count": payload.get("generated_rule_count", len(suite_lines)),
        "note": (
            "LLM1 rules are first-pass generated validation signals. "
            "LLM2 must verify them against manual validation, parsed logs, and dynamic patterns."
        ),
    }


def _build_parsed_summary(parsed_path: Path) -> dict:
    df = pd.read_csv(parsed_path)

    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "sample_rows": df.head(8).fillna("").astype(str).to_dict(orient="records"),
        "column_profiles": {},
    }

    for col in df.columns:
        series = df[col]
        profile = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null):
                profile.update({
                    "min": float(non_null.min()),
                    "max": float(non_null.max()),
                    "mean": round(float(non_null.mean()), 3),
                })

        else:
            top_values = series.fillna("").astype(str).value_counts().head(8)
            profile["top_values"] = {
                str(k): int(v) for k, v in top_values.items()
            }

        summary["column_profiles"][col] = profile

    return summary


def _summarize_patterns(patterns) -> dict:
    """
    Keeps pattern evidence compact and useful.
    Handles either list or dict pattern outputs.
    """
    if isinstance(patterns, dict):
        pattern_items = patterns.get("patterns", patterns.get("results", []))

        summary = {
            "top_level_keys": list(patterns.keys()),
            "raw_summary_fields": {
                k: v for k, v in patterns.items()
                if k not in {"patterns", "results"} and isinstance(v, (str, int, float, bool, list, dict))
            },
        }

    elif isinstance(patterns, list):
        pattern_items = patterns
        summary = {
            "top_level_keys": [],
            "raw_summary_fields": {},
        }
    else:
        return {
            "pattern_count": 0,
            "top_patterns": [],
            "note": "patterns_found.json format was not list or dict.",
        }

    normalized = []

    for p in pattern_items:
        if not isinstance(p, dict):
            continue

        score = (
            p.get("severity")
            or p.get("score")
            or p.get("confidence")
            or p.get("count")
            or p.get("frequency")
            or 0
        )

        normalized.append({
            "name": p.get("name") or p.get("pattern") or p.get("type"),
            "severity": p.get("severity"),
            "count": p.get("count") or p.get("frequency"),
            "confidence": p.get("confidence"),
            "description": p.get("description") or p.get("summary"),
            "evidence": p.get("evidence") or p.get("examples") or p.get("sample"),
            "_sort_score": score if isinstance(score, (int, float)) else 0,
        })

    normalized = sorted(normalized, key=lambda x: x["_sort_score"], reverse=True)

    for p in normalized:
        p.pop("_sort_score", None)

    summary.update({
        "pattern_count": len(normalized),
        "top_patterns": normalized[:25],
    })

    return summary


def _build_evidence_packet(
    parsed_summary: dict,
    manual_validation_summary: dict,
    llm1_summary: dict,
    patterns_summary: dict,
) -> dict:
    return {
        "analysis_instructions": {
            "llm1_handling": "Treat LLM1 generated rules as hypotheses/signals, not ground truth.",
            "priority_order": [
                "1. Direct parsed log evidence",
                "2. Manual validation failures",
                "3. Dynamic pattern findings",
                "4. LLM1 generated validation rules",
            ],
            "do_not_invent": [
                "event IDs",
                "HRESULT codes",
                "KB package names",
                "timestamps",
                "counts",
                "percentages",
            ],
        },
        "parsed_logs": parsed_summary,
        "manual_validation": manual_validation_summary,
        "llm1_generated_validation": llm1_summary,
        "dynamic_patterns": patterns_summary,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_analysis(
    parsed_path: Path = PARSED_FILE,
    patterns_path: Path = PATTERNS_FILE,
    validation_path: Path = VALIDATION_FILE,
    llm1_path: Path = LLM1_FILE,
    output_path: Path = OUTPUT_FILE,
    provider_override: str | None = None,
    model_override: str | None = None,
    hf_base_url_override: str | None = None,
) -> str:
    load_dotenv(ROOT / ".env")

    provider, model, hf_base_url = get_llm2_config(
        provider_override=provider_override,
        model_override=model_override,
        hf_base_url_override=hf_base_url_override,
    )

    if provider not in {"huggingface", "hf"}:
        raise RuntimeError(f"Unsupported LLM2 provider {provider!r}. Use Hugging Face for LLM2.")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN or HUGGINGFACE_API_TOKEN is not set for LLM2.")

    patterns = _safe_read_json(patterns_path, default={})
    validation = _safe_read_json(validation_path, default={})
    llm1_payload = _safe_read_json(llm1_path, default=None)

    parsed_summary = _build_parsed_summary(parsed_path)
    manual_validation_summary = _extract_validation_summary(validation)

    if llm1_payload is not None:
        llm1_summary = _extract_llm1_summary(llm1_payload, LLM1_SUITE_FILE)
    else:
        llm1_summary = {
            "available": False,
            "message": "LLM1 output not available.",
        }

    patterns_summary = _summarize_patterns(patterns)

    evidence_packet = _build_evidence_packet(
        parsed_summary=parsed_summary,
        manual_validation_summary=manual_validation_summary,
        llm1_summary=llm1_summary,
        patterns_summary=patterns_summary,
    )

    user_message = f"""\
Analyze the following evidence packet from the log quality pipeline.

Your task:
- Produce a root-cause analysis.
- Validate or reject LLM1-generated signals.
- Connect manual validation failures with dynamic pattern findings.
- Use exact evidence where available.
- Do not invent unsupported facts.

EVIDENCE PACKET:
{json.dumps(evidence_packet, indent=2)}
"""

    print(f"Sending evidence packet to {model} for analysis...\n")
    print(f"Prompt size: {len(user_message):,} characters")
    print("=" * 72)

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
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "max_tokens": 5000,
                "temperature": 0.05,
            },
            timeout=600,
        )

        response.raise_for_status()
        payload = response.json()

        content = payload["choices"][0]["message"]["content"]

        if isinstance(content, list):
            full_response = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            full_response = str(content)

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot connect to Hugging Face at {hf_base_url}.") from exc

    except requests.HTTPError as exc:
        detail = exc.response.text[-1000:] if exc.response is not None else str(exc)
        raise RuntimeError(f"Hugging Face request failed: {detail}") from exc

    print(full_response)
    print("\n" + "=" * 72)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_response, encoding="utf-8")

    print(f"\nReport saved to {output_path}")

    return full_response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_analysis()