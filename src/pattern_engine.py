"""
Stage 3 – Dynamic Pattern Engine

Reads outputs/parsed_logs.csv and detects useful patterns from any schema.
Designed to support root-cause analysis without hardcoding one dataset.
"""

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "outputs" / "parsed_logs.csv"
OUTPUT_FILE = ROOT / "outputs" / "patterns_found.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_pct(part, total) -> float:
    try:
        return round(float(part) / float(total) * 100, 2) if total else 0
    except Exception:
        return 0


def safe_to_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def value_counts_json(series: pd.Series, limit: int = 10) -> dict:
    counts = safe_to_string(series).replace("", pd.NA).dropna().value_counts().head(limit)
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def detect_datetime_series(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    timestamp_col = find_column(
        df,
        ["timestamp", "datetime", "date_time", "created_at", "logged_at", "time_stamp"],
    )

    if timestamp_col:
        parsed = pd.to_datetime(df[timestamp_col], errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed, timestamp_col

    date_col = find_column(df, ["date", "log_date"])
    time_col = find_column(df, ["time", "log_time"])

    if date_col and time_col:
        parsed = pd.to_datetime(
            safe_to_string(df[date_col]) + " " + safe_to_string(df[time_col]),
            errors="coerce",
        )
        if parsed.notna().sum() > 0:
            return parsed, f"{date_col}+{time_col}"

    # Last fallback: try every column and choose the one with most valid datetimes
    best_col = None
    best_series = None
    best_score = 0

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        score = parsed.notna().mean()

        if score > best_score:
            best_col = col
            best_score = score

    if best_series is not None and best_score >= 0.5:
        return best_series, str(best_col)

    return None, None


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------
def pattern_dataset_overview(df: pd.DataFrame) -> dict:
    missing_counts = df.isna().sum()
    missing_summary = {
        str(col): {
            "missing_count": int(count),
            "missing_pct": safe_pct(count, len(df)),
        }
        for col, count in missing_counts.items()
        if count > 0
    }

    return {
        "pattern": "dataset_overview",
        "description": "Basic dataset size, schema, and missing-value summary",
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "columns": [str(col) for col in df.columns],
        "columns_with_missing_values": missing_summary,
    }


def pattern_column_type_summary(df: pd.DataFrame) -> dict:
    type_summary = {}

    for col in df.columns:
        series = df[col]
        type_summary[str(col)] = {
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "unique_pct": safe_pct(series.nunique(dropna=True), len(series)),
        }

    return {
        "pattern": "column_type_summary",
        "description": "Data type, non-null count, and cardinality for each column",
        "columns": type_summary,
    }


def pattern_categorical_dominance(df: pd.DataFrame) -> dict:
    findings = []

    for col in df.columns:
        series = safe_to_string(df[col]).replace("", pd.NA).dropna()

        if series.empty:
            continue

        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)

        # Treat low-cardinality columns as categorical
        if unique_count <= 30 or unique_ratio <= 0.2:
            counts = series.value_counts()
            top_value = str(counts.index[0])
            top_count = int(counts.iloc[0])

            findings.append({
                "column": str(col),
                "unique_values": int(unique_count),
                "top_value": top_value,
                "top_count": top_count,
                "top_pct": safe_pct(top_count, len(series)),
                "top_values": {str(k): int(v) for k, v in counts.head(10).to_dict().items()},
            })

    return {
        "pattern": "categorical_dominance",
        "description": "Low-cardinality columns and their dominant values",
        "columns_analyzed": int(len(findings)),
        "dominant_columns": findings,
    }


def pattern_numeric_outliers(df: pd.DataFrame) -> dict:
    findings = []

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce").dropna()

        # Skip columns that are barely numeric
        if len(numeric) < max(5, len(df) * 0.1):
            continue

        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_count = 0
            lower = float(q1)
            upper = float(q3)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = numeric[(numeric < lower) | (numeric > upper)]
            outlier_count = len(outliers)

        findings.append({
            "column": str(col),
            "numeric_count": int(len(numeric)),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "mean": round(float(numeric.mean()), 4),
            "median": round(float(numeric.median()), 4),
            "std": round(float(numeric.std()), 4) if len(numeric) > 1 else 0,
            "outlier_count": int(outlier_count),
            "outlier_pct": safe_pct(outlier_count, len(numeric)),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        })

    return {
        "pattern": "numeric_outliers",
        "description": "Numeric columns summarized with IQR-based outlier detection",
        "numeric_columns_analyzed": int(len(findings)),
        "outlier_findings": findings,
    }


def pattern_boolean_flags(df: pd.DataFrame) -> dict:
    findings = []

    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n"}

    for col in df.columns:
        series = safe_to_string(df[col]).str.lower().str.strip()
        non_empty = series[series != ""]

        if non_empty.empty:
            continue

        values = set(non_empty.unique())

        if values.issubset(true_values | false_values):
            bool_series = non_empty.map(lambda x: x in true_values)
            true_count = int(bool_series.sum())
            false_count = int((~bool_series).sum())

            findings.append({
                "column": str(col),
                "true_count": true_count,
                "false_count": false_count,
                "true_pct": safe_pct(true_count, len(bool_series)),
                "false_pct": safe_pct(false_count, len(bool_series)),
            })

    return {
        "pattern": "boolean_flag_distribution",
        "description": "Boolean and boolean-like column distributions",
        "boolean_columns_found": int(len(findings)),
        "flags": findings,
    }


def pattern_timestamp_bursts(df: pd.DataFrame) -> dict:
    ts, source = detect_datetime_series(df)

    if ts is None:
        return {
            "pattern": "timestamp_bursts",
            "description": "Repeated timestamp burst detection",
            "status": "skipped",
            "reason": "No usable timestamp-like column found",
        }

    ts = ts.dropna()

    if ts.empty:
        return {
            "pattern": "timestamp_bursts",
            "description": "Repeated timestamp burst detection",
            "status": "skipped",
            "reason": "Timestamp values could not be parsed",
        }

    counts = ts.value_counts()
    bursts = counts[counts > 1]
    rows_in_bursts = int(bursts.sum()) if len(bursts) else 0

    return {
        "pattern": "timestamp_bursts",
        "description": "Rows sharing identical timestamp values",
        "timestamp_source": str(source),
        "valid_timestamps": int(len(ts)),
        "unique_timestamps": int(counts.shape[0]),
        "burst_timestamps": int(bursts.shape[0]),
        "rows_in_bursts": rows_in_bursts,
        "pct_rows_in_bursts": safe_pct(rows_in_bursts, len(df)),
        "largest_burst_count": int(counts.max()),
        "largest_burst_timestamp": str(counts.idxmax()),
    }


def pattern_text_signals(df: pd.DataFrame) -> dict:
    content_col = find_column(
        df,
        ["Content", "message", "msg", "log", "text", "description", "details"],
    )

    if content_col is None:
        # Fallback: choose longest average text column
        best_col = None
        best_len = 0

        for col in df.columns:
            series = safe_to_string(df[col])
            avg_len = series.str.len().mean()

            if avg_len > best_len:
                best_col = col
                best_len = avg_len

        content_col = best_col if best_len >= 10 else None

    if content_col is None:
        return {
            "pattern": "text_signal_summary",
            "description": "Keyword signals from text-like columns",
            "status": "skipped",
            "reason": "No useful text-like column found",
        }

    text = safe_to_string(df[content_col])

    signals = {
        "error_failure": r"error|exception|fail|failed|failure",
        "warning": r"warn|warning",
        "timeout": r"timeout|timed out",
        "reboot_restart_shutdown": r"reboot|restart|shutdown",
        "success_complete": r"success|successful|complete|completed",
        "login_auth": r"login|auth|authentication|authorized|unauthorized",
    }

    results = {}

    for name, pattern in signals.items():
        mask = text.str.contains(pattern, case=False, na=False, regex=True)
        results[name] = {
            "count": int(mask.sum()),
            "pct": safe_pct(mask.sum(), len(df)),
        }

    return {
        "pattern": "text_signal_summary",
        "description": "Keyword-based signal summary from the most useful text column",
        "text_column_used": str(content_col),
        "signals": results,
    }


def pattern_event_sequences(df: pd.DataFrame) -> dict:
    event_col = find_column(
        df,
        ["EventId", "event_id", "event", "code", "event_code", "id"],
    )

    if event_col is None:
        return {
            "pattern": "event_sequences",
            "description": "Most common adjacent event sequences",
            "status": "skipped",
            "reason": "No event-like column found",
        }

    events = safe_to_string(df[event_col]).replace("", pd.NA).dropna().tolist()

    if len(events) < 3:
        return {
            "pattern": "event_sequences",
            "description": "Most common adjacent event sequences",
            "status": "skipped",
            "reason": "Not enough event values",
        }

    pairs = {}
    triples = {}

    for i in range(len(events) - 1):
        pair = f"{events[i]} -> {events[i + 1]}"
        pairs[pair] = pairs.get(pair, 0) + 1

    for i in range(len(events) - 2):
        triple = f"{events[i]} -> {events[i + 1]} -> {events[i + 2]}"
        triples[triple] = triples.get(triple, 0) + 1

    top_pairs = dict(sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:10])
    top_triples = dict(sorted(triples.items(), key=lambda x: x[1], reverse=True)[:10])

    return {
        "pattern": "event_sequences",
        "description": "Most common adjacent 2-event and 3-event sequences",
        "event_column_used": str(event_col),
        "total_events": int(len(events)),
        "top_2_event_sequences": {str(k): int(v) for k, v in top_pairs.items()},
        "top_3_event_sequences": {str(k): int(v) for k, v in top_triples.items()},
    }


def pattern_code_repetition(df: pd.DataFrame) -> dict:
    candidate_cols = []

    for col in df.columns:
        name = str(col).lower()

        if any(token in name for token in ["id", "code", "event", "package", "template", "error"]):
            candidate_cols.append(col)

    findings = []

    for col in candidate_cols:
        series = safe_to_string(df[col]).replace("", pd.NA).dropna()

        if series.empty:
            continue

        counts = series.value_counts()
        top_value = str(counts.index[0])
        top_count = int(counts.iloc[0])

        findings.append({
            "column": str(col),
            "non_empty_rows": int(len(series)),
            "unique_values": int(series.nunique()),
            "top_value": top_value,
            "top_count": top_count,
            "top_pct_of_non_empty": safe_pct(top_count, len(series)),
            "top_values": {str(k): int(v) for k, v in counts.head(10).to_dict().items()},
        })

    return {
        "pattern": "code_repetition",
        "description": "Repeated IDs, codes, packages, templates, and code-like fields",
        "columns_analyzed": int(len(findings)),
        "repetition_findings": findings,
    }


def pattern_template_complexity(df: pd.DataFrame) -> dict:
    param_col = find_column(df, ["param_count"])
    template_col = find_column(df, ["EventTemplate", "event_template", "template", "pattern"])

    if param_col:
        pc = pd.to_numeric(df[param_col], errors="coerce").dropna()
        source = param_col
    elif template_col:
        pc = safe_to_string(df[template_col]).str.count(r"<\*>")
        source = template_col
    else:
        return {
            "pattern": "template_complexity",
            "description": "Template complexity based on variable placeholders",
            "status": "skipped",
            "reason": "No param_count or template-like column found",
        }

    if pc.empty:
        return {
            "pattern": "template_complexity",
            "description": "Template complexity based on variable placeholders",
            "status": "skipped",
            "reason": "No valid complexity values found",
        }

    dist = pc.value_counts().sort_index().to_dict()

    return {
        "pattern": "template_complexity",
        "description": "Statistics on variable placeholders in templates",
        "source_column": str(source),
        "mean": round(float(pc.mean()), 4),
        "min": int(pc.min()),
        "max": int(pc.max()),
        "median": float(pc.median()),
        "std": round(float(pc.std()), 4) if len(pc) > 1 else 0,
        "pct_zero_params": safe_pct(int((pc == 0).sum()), len(pc)),
        "pct_high_params": safe_pct(int((pc >= 5).sum()), len(pc)),
        "distribution": {str(k): int(v) for k, v in dist.items()},
    }


def pattern_correlation_candidates(df: pd.DataFrame) -> dict:
    numeric_df = pd.DataFrame()

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            numeric_df[str(col)] = numeric

    if numeric_df.shape[1] < 2:
        return {
            "pattern": "correlation_candidates",
            "description": "Strong numeric column correlations",
            "status": "skipped",
            "reason": "Fewer than two numeric-like columns found",
        }

    corr = numeric_df.corr(numeric_only=True)
    pairs = []

    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.loc[cols[i], cols[j]]

            if pd.notna(value) and abs(value) >= 0.7:
                pairs.append({
                    "column_a": cols[i],
                    "column_b": cols[j],
                    "correlation": round(float(value), 4),
                })

    pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "pattern": "correlation_candidates",
        "description": "Highly correlated numeric-like column pairs",
        "strong_pairs_found": int(len(pairs)),
        "strong_pairs": pairs[:20],
    }


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------
DETECTORS = [
    pattern_dataset_overview,
    pattern_column_type_summary,
    pattern_categorical_dominance,
    pattern_numeric_outliers,
    pattern_boolean_flags,
    pattern_timestamp_bursts,
    pattern_text_signals,
    pattern_event_sequences,
    pattern_code_repetition,
    pattern_template_complexity,
    pattern_correlation_candidates,
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_pattern_engine(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
) -> dict:
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input file is empty. Pattern engine cannot run.")

    findings = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "patterns": [],
    }

    print(f"Running dynamic pattern engine on {len(df):,} rows and {len(df.columns):,} columns...\n")

    for detector in DETECTORS:
        try:
            result = detector(df)
            findings["patterns"].append(result)

            print(f"[{result.get('pattern')}]")
            print(f"  {result.get('description')}")

            if result.get("status") == "skipped":
                print(f"  Skipped: {result.get('reason')}")
            else:
                shown = 0
                for key, value in result.items():
                    if key in {"pattern", "description"}:
                        continue

                    if shown >= 6:
                        break

                    if isinstance(value, list):
                        print(f"  {key}: {len(value)} item(s)")
                    elif isinstance(value, dict):
                        print(f"  {key}: {len(value)} item(s)")
                    else:
                        print(f"  {key}: {value}")

                    shown += 1

            print()

        except Exception as exc:
            error_result = {
                "pattern": detector.__name__,
                "description": "This detector failed but the engine continued",
                "status": "error",
                "error": str(exc),
            }

            findings["patterns"].append(error_result)

            print(f"[{detector.__name__}]")
            print(f"  Error: {exc}")
            print("  Continuing with next detector.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"All findings saved to {output_path}")

    return findings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pattern_engine()
