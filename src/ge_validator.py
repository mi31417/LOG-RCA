"""
Stage 2 – Great Expectations Validator (GX 1.0 API)
Reads outputs/parsed_logs.csv, runs an expectation suite, and saves results.
"""

import json
from pathlib import Path

import pandas as pd
import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnValuesToNotBeNull,
    ExpectColumnValuesToBeUnique,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToMatchRegex,
    ExpectColumnValuesToBeBetween,
    ExpectColumnMeanToBeBetween,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent
INPUT_FILE  = ROOT / "outputs" / "parsed_logs.csv"
OUTPUT_FILE = ROOT / "outputs" / "validation_results.json"


# ---------------------------------------------------------------------------
# Expectation definitions
# ---------------------------------------------------------------------------
def _default_rule_specs(df: pd.DataFrame) -> list[dict]:
    rules = [
        {"type": "not_null", "column": "LineId"},
        {"type": "not_null", "column": "Date"},
        {"type": "not_null", "column": "Time"},
        {"type": "not_null", "column": "Component"},
        {"type": "not_null", "column": "Content"},
        {"type": "not_null", "column": "EventId"},
        {"type": "unique", "column": "LineId"},
        {"type": "in_set", "column": "Level", "value_set": ["Info", "Warning", "Error", "Critical"]},
        {
            "type": "in_set",
            "column": "event_class",
            "value_set": ["FAILURE", "WARNING", "SESSION", "PACKAGE", "INIT", "REBOOT", "INFO"],
        },
        {"type": "regex", "column": "EventId", "regex": r"E\d+"},
        {"type": "regex", "column": "Date", "regex": r"\d{4}-\d{2}-\d{2}"},
        {"type": "between", "column": "param_count", "min_value": 0, "max_value": 20},
        {"type": "between", "column": "LineId", "min_value": 1, "max_value": 100_000},
        {"type": "mean_between", "column": "has_failure", "min_value": 0.00, "max_value": 0.5},
    ]

    component_values = set(df["Component"].dropna().astype(str).str.strip()) if "Component" in df.columns else set()
    component_values.discard("")
    if component_values and component_values.issubset({"CBS", "CSI"}):
        rules.append({"type": "in_set", "column": "Component", "value_set": ["CBS", "CSI"]})

    return rules


EXPECTATION_BUILDERS = {
    "not_null": lambda rule: ExpectColumnValuesToNotBeNull(column=rule["column"]),
    "unique": lambda rule: ExpectColumnValuesToBeUnique(column=rule["column"]),
    "in_set": lambda rule: ExpectColumnValuesToBeInSet(
        column=rule["column"],
        value_set=set(rule["value_set"]),
    ),
    "regex": lambda rule: ExpectColumnValuesToMatchRegex(
        column=rule["column"],
        regex=rule["regex"],
    ),
    "between": lambda rule: ExpectColumnValuesToBeBetween(
        column=rule["column"],
        min_value=rule.get("min_value"),
        max_value=rule.get("max_value"),
    ),
    "mean_between": lambda rule: ExpectColumnMeanToBeBetween(
        column=rule["column"],
        min_value=rule.get("min_value"),
        max_value=rule.get("max_value"),
    ),
}


def _load_custom_rules(custom_rules: str | list[dict] | None, df: pd.DataFrame) -> list:
    if not custom_rules:
        payload = _default_rule_specs(df)
    elif isinstance(custom_rules, str):
        try:
            payload = json.loads(custom_rules)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Custom rules JSON is invalid: {exc}") from exc
    else:
        payload = custom_rules

    if not isinstance(payload, list):
        raise ValueError("Custom rules must be a JSON array of rule objects.")

    expectations = []
    missing_columns = set()

    for idx, rule in enumerate(payload, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Custom rule #{idx} must be an object.")

        rule_type = rule.get("type")
        column = rule.get("column")

        if rule_type not in EXPECTATION_BUILDERS:
            supported = ", ".join(sorted(EXPECTATION_BUILDERS))
            raise ValueError(f"Unsupported custom rule type {rule_type!r}. Supported types: {supported}.")

        if not column:
            raise ValueError(f"Custom rule #{idx} is missing 'column'.")

        if column not in df.columns:
            missing_columns.add(column)
            continue

        if rule_type == "mean_between":
            df[column] = pd.to_numeric(df[column], errors="coerce")

        expectations.append(EXPECTATION_BUILDERS[rule_type](rule))

    if missing_columns:
        cols = ", ".join(sorted(missing_columns))
        raise ValueError(f"Custom rules reference columns not found in parsed data: {cols}")

    if not expectations:
        raise ValueError("No valid custom rules were provided.")

    return expectations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_validation(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    custom_rules: str | list[dict] | None = None,
) -> dict:

    df = pd.read_csv(input_path)

    # Cast boolean column to float so the mean expectation works correctly
    if "has_failure" in df.columns:
        df["has_failure"] = pd.to_numeric(df["has_failure"], errors="coerce")

    expectations = _load_custom_rules(custom_rules, df)

    # -- GX ephemeral context (no filesystem project needed) --
    ctx = gx.get_context(mode="ephemeral")

    datasource  = ctx.data_sources.add_pandas("log_datasource")
    asset       = datasource.add_dataframe_asset("parsed_logs")
    batch_def   = asset.add_batch_definition_whole_dataframe("full_batch")

    suite = ctx.suites.add(gx.ExpectationSuite(name="log_quality_suite"))
    for exp in expectations:
        suite.add_expectation(exp)

    vd = ctx.validation_definitions.add(
        gx.ValidationDefinition(
            name="log_quality_validation",
            data=batch_def,
            suite=suite,
        )
    )

    result = vd.run(batch_parameters={"dataframe": df})

    # -- Serialize to JSON-safe dict --
    raw_candidate = (result.to_json_dict()
                     if hasattr(result, "to_json_dict")
                     else result.json())
    # to_json_dict() already returns a dict; .json() returns a string
    raw = raw_candidate if isinstance(raw_candidate, dict) else json.loads(raw_candidate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)

    # -- Summary --
    stats      = raw.get("statistics", {})
    total      = stats.get("evaluated_expectations", len(expectations))
    passed     = stats.get("successful_expectations", 0)
    failed     = total - passed

    print(f"\nValidation complete — results saved to {output_path}")
    print(f"  Total expectations : {total}")
    print(f"  Passed             : {passed}")
    print(f"  Failed             : {failed}")

    if failed:
        print("\nFailed expectations:")
        for exp_result in raw.get("expectations", []):
            if not exp_result.get("success"):
                etype  = exp_result.get("expectation_type", "?")
                kwargs = exp_result.get("kwargs", {})
                col    = kwargs.get("column", "—")
                print(f"  ✗ {etype}  [column={col}]")

    return raw


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_validation()
