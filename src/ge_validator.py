"""
Stage 2 – Great Expectations Validator
Reads outputs/parsed_logs.csv, loads validation rules, runs validation,
and saves results.
"""

import json
from pathlib import Path

import pandas as pd
import great_expectations as gx


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "outputs" / "parsed_logs.csv"
OUTPUT_FILE = ROOT / "outputs" / "validation_results.json"
CONFIG_FILE = ROOT / "validation_rules.json"


def load_rules() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("No validation_rules.json found. Using generic validation only.")
    return {}


def run_validation(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    custom_rules=None,
) -> dict:

    df = pd.read_csv(input_path)

    if "has_failure" in df.columns:
        df["has_failure"] = pd.to_numeric(df["has_failure"], errors="coerce")

    # Choose custom rules if user provided them, otherwise use default rules
    if custom_rules:
        try:
            if isinstance(custom_rules, str):
                rules = json.loads(custom_rules)
            else:
                rules = custom_rules
        except Exception as e:
            raise ValueError(f"Invalid custom rules JSON: {e}")
    else:
        rules = load_rules()

    ctx = gx.get_context(mode="ephemeral")
    validator = ctx.sources.pandas_default.read_dataframe(df)

    # ------------------------------------------------------------
    # 1. Generic rules for ANY dataset
    # ------------------------------------------------------------
    for column in df.columns:
        validator.expect_column_values_to_not_be_null(
            column=column,
            mostly=0.50
        )

    for column in df.columns:
        lower_col = column.lower()
        if lower_col in ["id", "lineid", "line_id", "eventid", "event_id"]:
            validator.expect_column_values_to_be_unique(column=column)

    # ------------------------------------------------------------
    # 2. Optional custom/default rules
    # ------------------------------------------------------------
    for column in rules.get("not_null", []):
        if column in df.columns:
            validator.expect_column_values_to_not_be_null(column=column)

    for column in rules.get("unique", []):
        if column in df.columns:
            validator.expect_column_values_to_be_unique(column=column)

    for column, value_set in rules.get("in_set", {}).items():
        if column in df.columns:
            validator.expect_column_values_to_be_in_set(
                column=column,
                value_set=list(value_set)
            )

    for column, regex in rules.get("regex", {}).items():
        if column in df.columns:
            validator.expect_column_values_to_match_regex(
                column=column,
                regex=regex
            )

    for column, bounds in rules.get("between", {}).items():
        if column in df.columns:
            validator.expect_column_values_to_be_between(
                column=column,
                min_value=bounds["min"],
                max_value=bounds["max"]
            )

    for column, bounds in rules.get("mean_between", {}).items():
        if column in df.columns:
            validator.expect_column_mean_to_be_between(
                column=column,
                min_value=bounds["min"],
                max_value=bounds["max"]
            )

    result = validator.validate()

    raw_candidate = (
        result.to_json_dict()
        if hasattr(result, "to_json_dict")
        else result.json()
    )
    raw = raw_candidate if isinstance(raw_candidate, dict) else json.loads(raw_candidate)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, default=str)

    stats = raw.get("statistics", {})
    total = stats.get("evaluated_expectations", 0)
    passed = stats.get("successful_expectations", 0)
    failed = total - passed

    print(f"\nValidation complete — results saved to {output_path}")
    print(f"  Dataset rows       : {len(df)}")
    print(f"  Dataset columns    : {len(df.columns)}")
    print(f"  Total expectations : {total}")
    print(f"  Passed             : {passed}")
    print(f"  Failed             : {failed}")

    return raw


if __name__ == "__main__":
    run_validation()
