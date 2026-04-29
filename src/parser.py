"""
Stage 1 – Log Parser
Accepts a structured CSV or raw .log file and enriches it with 9 derived columns.
"""

import re
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASET

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT /DATASET
OUTPUT_FILE = ROOT / "outputs" / "parsed_logs.csv"

REQUIRED_COLUMNS = [
    "LineId",
    "Date",
    "Time",
    "Level",
    "Component",
    "Content",
    "EventId",
    "EventTemplate",
]

COLUMN_ALIASES = {
    "LineId": ["lineid", "line_id", "row_id", "id", "index"],
    "Date": ["date", "log_date"],
    "Time": ["time", "log_time"],
    "Level": ["level", "severity", "log_level", "status"],
    "Component": ["component", "source", "module", "service", "logger", "host"],
    "Content": ["content", "message", "log", "text", "description", "details"],
    "EventId": ["eventid", "event_id", "event", "code"],
    "EventTemplate": ["eventtemplate", "event_template", "template", "pattern"],
    "timestamp": ["timestamp", "datetime", "date_time", "ts"],
}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
RE_HRESULT  = re.compile(r'0x8[0-9A-Fa-f]+')
RE_KB       = re.compile(r'KB\d+', re.IGNORECASE)
RE_SESSION  = re.compile(r'[Ss]ession\s+([\w\-]+)')
RE_STAR     = re.compile(r'<\*>')
RE_APACHE   = re.compile(
    r"^\[(?P<stamp>[^\]]+)\]\s+\[(?P<level>[^\]]+)\]\s+(?P<content>.*)$"
)


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------
_FAIL_KW    = re.compile(r'fail', re.IGNORECASE)
_WARN_KW    = re.compile(r'warning', re.IGNORECASE)
_SESSION_KW = re.compile(r'session', re.IGNORECASE)
_PKG_KW     = re.compile(r'KB\d+|package', re.IGNORECASE)
_INIT_KW    = re.compile(r'init|initializ|load', re.IGNORECASE)
_REBOOT_KW  = re.compile(r'reboot|restart|shutdown', re.IGNORECASE)


def _classify(content: str, has_failure: bool, has_warning: bool, level: str = "") -> str:
    level_normalized = str(level).strip().lower()
    if level_normalized in {"error", "critical", "crit"}:
        return "FAILURE"
    if level_normalized in {"warning", "warn"}:
        return "WARNING"
    if has_failure:
        return "FAILURE"
    if has_warning:
        return "WARNING"
    if _SESSION_KW.search(content):
        return "SESSION"
    if _PKG_KW.search(content):
        return "PACKAGE"
    if _REBOOT_KW.search(content):
        return "REBOOT"
    if _INIT_KW.search(content):
        return "INIT"
    return "INFO"


def _normalize_level(level: str) -> str:
    level = str(level).strip().lower()
    mapping = {
        "notice": "Info",
        "info": "Info",
        "warning": "Warning",
        "warn": "Warning",
        "error": "Error",
        "critical": "Critical",
        "crit": "Critical",
    }
    return mapping.get(level, level.title() if level else "Info")


def _infer_component(content: str) -> str:
    token = re.split(r"[\s:(]", content.strip(), maxsplit=1)[0]
    token = token.strip("[]")
    if token and any(ch.isalpha() for ch in token):
        return token[:32]
    return "UNKNOWN"


def _build_event_template(content: str) -> str:
    template = re.sub(r"0x[0-9A-Fa-f]+", "<*>", content)
    template = re.sub(r"\b\d+\b", "<*>", template)
    template = re.sub(r"\b[A-Z]:\\[^\s,]+", "<*>", template)
    template = re.sub(r"\s+", " ", template).strip()
    return template or "<*>"


def _find_matching_column(df: pd.DataFrame, target: str) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for alias in COLUMN_ALIASES.get(target, []):
        if alias in normalized:
            return normalized[alias]
    return None


def _split_timestamp(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(series, errors="coerce")
    return (
        parsed.dt.strftime("%Y-%m-%d").fillna(""),
        parsed.dt.strftime("%H:%M:%S").fillna(""),
    )


def _build_content_from_row(row: pd.Series) -> str:
    parts = []
    for key, value in row.items():
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            parts.append(f"{key}={text}")
    return " | ".join(parts)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")

    if all(col in df.columns for col in REQUIRED_COLUMNS):
        return df

    normalized = df.copy()
    timestamp_col = _find_matching_column(df, "timestamp")
    date_col = _find_matching_column(df, "Date")
    time_col = _find_matching_column(df, "Time")
    content_col = _find_matching_column(df, "Content")
    component_col = _find_matching_column(df, "Component")
    level_col = _find_matching_column(df, "Level")
    event_id_col = _find_matching_column(df, "EventId")
    template_col = _find_matching_column(df, "EventTemplate")
    line_id_col = _find_matching_column(df, "LineId")

    if timestamp_col and not date_col:
        normalized["Date"], normalized["Time"] = _split_timestamp(df[timestamp_col])
    else:
        normalized["Date"] = df[date_col].astype(str).fillna("") if date_col else ""
        if time_col:
            normalized["Time"] = df[time_col].astype(str).fillna("")
        elif timestamp_col:
            _, normalized["Time"] = _split_timestamp(df[timestamp_col])
        else:
            normalized["Time"] = ""

    if "Date" not in normalized:
        normalized["Date"] = ""
    if "Time" not in normalized:
        normalized["Time"] = ""

    normalized["LineId"] = (
        pd.to_numeric(df[line_id_col], errors="coerce").fillna(range(1, len(df) + 1)).astype(int)
        if line_id_col else pd.Series(range(1, len(df) + 1), index=df.index)
    )
    normalized["Level"] = (
        df[level_col].astype(str).map(_normalize_level).fillna("Info")
        if level_col else "Info"
    )
    if component_col:
        normalized["Component"] = df[component_col].astype(str).fillna("").replace("", "UNKNOWN")
    else:
        source_series = df.apply(_build_content_from_row, axis=1)
        normalized["Component"] = source_series.map(_infer_component)

    if content_col:
        normalized["Content"] = df[content_col].astype(str).fillna("")
    else:
        normalized["Content"] = df.apply(_build_content_from_row, axis=1)

    if event_id_col:
        raw_ids = df[event_id_col].astype(str).fillna("")
        normalized["EventId"] = raw_ids.where(raw_ids.str.strip() != "", other="")
        empty_mask = normalized["EventId"].eq("")
        normalized.loc[empty_mask, "EventId"] = [
            f"E{i}" for i in normalized.loc[empty_mask, "LineId"]
        ]
    else:
        normalized["EventId"] = normalized["LineId"].map(lambda v: f"E{v}")

    if template_col:
        normalized["EventTemplate"] = df[template_col].astype(str).fillna("")
        empty_mask = normalized["EventTemplate"].str.strip().eq("")
        normalized.loc[empty_mask, "EventTemplate"] = normalized.loc[empty_mask, "Content"].map(_build_event_template)
    else:
        normalized["EventTemplate"] = normalized["Content"].map(_build_event_template)

    for col in REQUIRED_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""

    ordered = REQUIRED_COLUMNS + [col for col in normalized.columns if col not in REQUIRED_COLUMNS]
    return normalized[ordered]


def _parse_raw_log_file(input_path: Path) -> pd.DataFrame:
    rows = []
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = RE_APACHE.match(line)
            if match:
                stamp = pd.to_datetime(match.group("stamp"), errors="coerce")
                content = match.group("content").strip()
                level = _normalize_level(match.group("level"))
                date = stamp.strftime("%Y-%m-%d") if not pd.isna(stamp) else ""
                time = stamp.strftime("%H:%M:%S") if not pd.isna(stamp) else ""
            else:
                content = line
                level = "Info"
                date = ""
                time = ""

            rows.append(
                {
                    "LineId": index,
                    "Date": date,
                    "Time": time,
                    "Level": level,
                    "Component": _infer_component(content),
                    "Content": content,
                    "EventId": f"E{index}",
                    "EventTemplate": _build_event_template(content),
                }
            )

    if not rows:
        raise ValueError("The uploaded log file is empty.")

    return _normalize_dataframe(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))


def load_logs(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(input_path)
        return _normalize_dataframe(df)
    if suffix == ".log":
        return _parse_raw_log_file(input_path)
    raise ValueError("Unsupported file type. Please upload a .csv or .log file.")


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------
def parse_logs(input_path: Path = INPUT_FILE,
               output_path: Path = OUTPUT_FILE) -> pd.DataFrame:

    print(f"Looking for file at: {input_path}")

    df = load_logs(Path(input_path))

    content = df["Content"].fillna("")
    template = df["EventTemplate"].fillna("")

    # 1. timestamp – combine Date + Time into a single datetime column
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )

    # 2. hresult – first HRESULT code found (e.g. 0x80070002), else empty str
    df["hresult"] = content.apply(
        lambda c: m.group() if (m := RE_HRESULT.search(c)) else ""
    )

    # 3. kb_package – first KB number found (e.g. KB3145739), else empty str
    df["kb_package"] = content.apply(
        lambda c: m.group().upper() if (m := RE_KB.search(c)) else ""
    )

    # 4. session_id – first session identifier captured after "Session", else empty str
    df["session_id"] = content.apply(
        lambda c: m.group(1) if (m := RE_SESSION.search(c)) else ""
    )

    # 5. param_count – number of <*> tokens in the EventTemplate
    df["param_count"] = template.apply(lambda t: len(RE_STAR.findall(t)))

    level_series = df["Level"].fillna("").astype(str)

    # 6. has_failure – infer from level plus error/failure-like content
    df["has_failure"] = (
        level_series.str.lower().isin(["error", "critical", "crit"])
        | content.str.contains(r'fail|error|exception|forbidden|denied', case=False, na=False)
    )

    # 7. has_warning – infer from level plus warning-like content
    df["has_warning"] = (
        level_series.str.lower().isin(["warning", "warn"])
        | content.str.contains(r'warning|warn', case=False, na=False)
    )

    # 8. has_hresult – True if an HRESULT code was found
    df["has_hresult"] = df["hresult"] != ""

    # 9. event_class – high-level classification
    df["event_class"] = [
        _classify(c, f, w, level)
        for c, f, w, level in zip(content, df["has_failure"], df["has_warning"], level_series)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Shape : {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  {col}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parse_logs()
