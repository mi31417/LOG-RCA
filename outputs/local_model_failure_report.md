# Local Model Failure Report

## Stage 3 Goal

Generate a valid Great Expectations GX 1.0 validation suite from `outputs/parsed_logs.csv`.

Expected behavior:
- Return only executable Python code
- Use the GX 1.0 fluent API
- Save a runnable suite to `outputs/ge_suite_llm.py`

---

## Model 1: `mistral:7b`

### Verdict

Failed.

### Failure Type

- Returned prose before code
- Wrapped output in markdown fences
- Used a non-GX / incorrect library

### Output Snippet

```python
Here's a sample GX 1.0 validation suite Python script based on the provided data. I've added comments to explain each validation rule.

```python
import pandas as pd
from gx_validation import ValidationSuite

# Load the data
df = pd.read_csv('yourfile.csv')

# Create a validation suite
suite = ValidationSuite()
```

### Why It Was Rejected

The model ignored the prompt format and did not generate valid Great Expectations GX 1.0 code.

---

## Model 2: `codellama:7b`

### Verdict

Failed.

### Failure Type

- Returned prose before code
- Wrapped output in markdown fences
- Invented non-GX validation classes

### Output Snippet

```python
Here's a Python script for creating a GX 1.0 validation suite based on the provided data:

```python
import pandas as pd
from gx import ValidationSuite, ValidationRule

df = pd.read_csv('your_data.csv')
validation_suite = ValidationSuite("My Validation Suite")
```

### Why It Was Rejected

The model did not follow the required GX 1.0 API and produced non-executable output for this project.

---

## Model 3: `qwen2.5-coder:7b`

### Verdict

Failed during local test window.

### Failure Type

- Model downloaded successfully in Ollama
- Stage 3 run did not produce a fresh valid output in a reasonable time
- Small direct local smoke test also stalled instead of returning a quick answer

### Observed Result

```text
qwen2.5-coder:7b downloaded successfully, but local inference did not return a usable
Stage 3 result during the test run. No valid GX 1.0 suite was produced for documentation.
```

### Why It Was Rejected

The local run was too slow/unreliable for this workflow, so it did not provide a usable result for Stage 3 validation-suite generation.

---

## Summary

| Model | Result | Main Issue |
|---|---|---|
| `mistral:7b` | Failed | Prose + wrong library |
| `codellama:7b` | Failed | Prose + invented API |
| `qwen2.5-coder:7b` | Failed in test window | Stalled / no usable output |
