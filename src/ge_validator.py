"""
Stage 2 – Great Expectations Validator
Reads outputs/parsed_logs.csv, loads validation rules from validation_rules.json,
runs an expectation suite, and saves results.
"""

import json #for reading json config file
from pathlib import Path  #for handling fiel paths

import pandas as pd #data manipulation
import great_expectations as gx #main GX framework
import great_expectations.expectations as gxe #Expectation classes


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#Root directory of the project (it goes up one level from /src)
ROOT = Path(__file__).resolve().parents[1]
#input datset (outputs from stage1)
INPUT_FILE = ROOT / "outputs" / "parsed_logs.csv"
#output file from validation results
OUTPUT_FILE = ROOT / "outputs" / "validation_results.json"
#JSON file containing validation rules
CONFIG_FILE = ROOT / "validation_rules.json"


# ---------------------------------------------------------------------
#Load validation rules from JSON
# ---------------------------------------------------------------------
def load_rules() -> dict:
    #read validation rules from json file and returns it as a dictionary
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------
#Build expectations dynamically from rules 
# ---------------------------------------------------------------------

def build_expectations(df: pd.DataFrame, rules: dict) -> list:
    expectations = []

    for column in rules.get("not_null", []):
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnValuesToNotBeNull(column=column)
            )

    for column in rules.get("unique", []):
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnValuesToBeUnique(column=column)
            )

    for column, value_set in rules.get("in_set", {}).items():
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnValuesToBeInSet(
                    column=column,
                    value_set=list(value_set),
                )
            )

    for column, regex in rules.get("regex", {}).items():
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnValuesToMatchRegex(
                    column=column,
                    regex=regex,
                )
            )

    for column, bounds in rules.get("between", {}).items():
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnValuesToBeBetween(
                    column=column,
                    min_value=bounds["min"],
                    max_value=bounds["max"],
                )
            )

    for column, bounds in rules.get("mean_between", {}).items():
        if column in df.columns:
            expectations.append(
                gxe.ExpectColumnMeanToBeBetween(
                    column=column,
                    min_value=bounds["min"],
                    max_value=bounds["max"],
                )
            )

    return expectations


def run_validation(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
) -> dict:
    df = pd.read_csv(input_path)

    if "has_failure" in df.columns:
        df["has_failure"] = pd.to_numeric(df["has_failure"], errors="coerce")

    rules = load_rules()
    expectations = build_expectations(df, rules)

    ctx = gx.get_context(mode="ephemeral")

    datasource = ctx.data_sources.add_pandas("log_datasource")
    asset = datasource.add_dataframe_asset("parsed_logs")
    batch_def = asset.add_batch_definition_whole_dataframe("full_batch")

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
    total = stats.get("evaluated_expectations", len(expectations))
    passed = stats.get("successful_expectations", 0)
    failed = total - passed

    print(f"\nValidation complete — results saved to {output_path}")
    print(f"  Total expectations : {total}")
    print(f"  Passed             : {passed}")
    print(f"  Failed             : {failed}")

    return raw


if __name__ == "__main__":
    run_validation()
