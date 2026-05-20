# LLM-Driven Log Analysis and Pattern Discovery Using Great Expectations

A Flask-based application that analyzes logs using LLM-driven rule generation, Great Expectations validation, pattern discovery, and evidence-grounded root-cause analysis.

## Project Structure

```text
main.py                  Flask app and end-to-end pipeline runner
config.py                Minimal default dataset configuration
src/parser.py            Stage 1: parsing and enrichment
src/ge_validator.py      Stage 2: Great Expectations validation
src/llm1_ge_writer.py    Stage 3: LLM-generated GE suite
src/pattern_engine.py    Stage 4: pattern detection
src/llm2_analyst.py      Stage 5: root-cause analysis
src/visualizer.py        Stage 6: HTML dashboard generation
datasets/                Sample datasets
uploads/                 Uploaded input files
outputs/                 Generated artifacts and reports
```

## Quick Start

### 1. Extract and Open Project
```bash
# Extract the zip file and open in VS Code
unzip LLM_Log_RCA_Project.zip
cd LLM_Log_RCA_Project
code .
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python main.py
```

### 5. Open in Browser
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

### 6. Upload and Analyze
1. Click upload button
2. Select `Windows.log` or `Apache.log` from the provided sample files
3. Click the run button
4. Wait for the pipeline to complete
5. Scroll through results on screen (parsed logs, validation results, patterns, root-cause analysis)
6. Click "Open Dashboard" button to view the visual dashboard in a new page

## Requirements

- Python 3.10 or above
- pip
- Internet connection
- Hugging Face API token (add to `.env` file as `HF_TOKEN=your_token_here`)
- Web browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

**Virtual environment not activating?**
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

**Missing packages?**
```bash
pip install -r requirements.txt
```

**Flask not running?**
- Check that port 5000 is not in use
- Try: `python main.py`

**Hugging Face API error?**
- Create `.env` file in project root
- Add: `HF_TOKEN=your_valid_hugging_face_token`
- Ensure token is valid and has available credits

## Contributors

Naba Waseem, Pavithra Gottipati, Eshwar Prasad Bingi  
University of Maryland - Baltimore County (UMBC)
