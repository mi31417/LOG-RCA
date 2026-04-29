"""
Stage 5 – HTML Report Generator
Reads parsed_logs.csv and patterns_found.json, produces a single self-contained
outputs/pipeline_report.html with all charts embedded as base64 PNG images.
"""

import base64
import io
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
PARSED_CSV    = ROOT / "outputs" / "parsed_logs.csv"
PATTERNS_JSON = ROOT / "outputs" / "patterns_found.json"
OUTPUT_HTML   = ROOT / "outputs" / "pipeline_report.html"

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "FAILURE": "#d62728", "WARNING": "#ff7f0e", "SESSION": "#1f77b4",
    "PACKAGE": "#2ca02c", "INFO": "#9467bd", "INIT":    "#8c564b",
    "REBOOT":  "#e377c2",
}
ACCENT  = "#1f77b4"
DANGER  = "#d62728"
NEUTRAL = "#7f7f7f"
FIG_DPI = 110


def _infer_report_profile(df: pd.DataFrame) -> dict:
    components = set(df.get("Component", pd.Series(dtype="object")).fillna("").astype(str))
    components.discard("")
    level_values = set(df.get("Level", pd.Series(dtype="object")).fillna("").astype(str))
    level_values.discard("")

    source_label = "Structured log dataset"
    source_context = "Multi-stage quality and pattern analysis"

    if {"CBS", "CSI"} & components:
        source_label = "Windows servicing log dataset"
        source_context = "Windows update and component servicing activity"
    elif {"mod_jk", "workerEnv.init", "jk2_init"} & components:
        source_label = "Apache web server log dataset"
        source_context = "HTTP server worker and connector activity"
    elif {"Error", "Warning"} & level_values:
        source_context = "Operational application or infrastructure logs"

    return {
        "source_label": source_label,
        "source_context": source_context,
    }


# ---------------------------------------------------------------------------
# Helper: render a matplotlib figure to a base64 PNG <img> tag
# ---------------------------------------------------------------------------
def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;">'


# ---------------------------------------------------------------------------
# Chart 1 – Event class distribution (bar)
# ---------------------------------------------------------------------------
def chart_event_class(df: pd.DataFrame) -> str:
    counts = df["event_class"].value_counts().sort_values(ascending=True)
    colors = [CLASS_COLORS.get(c, NEUTRAL) for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, counts.values):
        ax.text(v + 6, bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({v/len(df)*100:.1f}%)", va="center", fontsize=9)
    ax.set_xlabel("Log entries")
    ax.set_title("Event Class Distribution", fontsize=13, pad=10)
    ax.set_xlim(0, counts.values.max() * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 2 – Top components
# ---------------------------------------------------------------------------
def chart_component_skew(df: pd.DataFrame) -> str:
    counts = df["Component"].fillna("UNKNOWN").astype(str).value_counts().head(6).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(counts.index, counts.values, color="#4d98ca", edgecolor="white", height=0.65)
    for bar, value in zip(bars, counts.values):
        ax.text(
            value + max(counts.values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,} ({value / len(df) * 100:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Log entries")
    ax.set_title("Top Components", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, counts.values.max() * 1.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 3 – Top code/signal values
# ---------------------------------------------------------------------------
def chart_hresult(df: pd.DataFrame) -> str:
    hresult_rows = df[df["hresult"] != ""]
    if hresult_rows.empty:
        template_counts = df["EventTemplate"].fillna("").astype(str).value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        bars = ax.bar(range(len(template_counts)), template_counts.values, color=ACCENT, edgecolor="white", width=0.6)
        ax.set_xticks(range(len(template_counts)))
        ax.set_xticklabels([textwrap.fill(t[:48], 18) for t in template_counts.index], fontsize=8.5)
        for bar, v in zip(bars, template_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:,}", ha="center", fontsize=9)
        ax.set_ylabel("Occurrences")
        ax.set_title("Top Event Templates", fontsize=13, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(template_counts.values.max() * 1.25, 1))
        fig.tight_layout()
        return _fig_to_b64(fig)

    top5 = hresult_rows["hresult"].value_counts().head(5)
    total = len(hresult_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(top5.index, top5.values,
                  color=[DANGER] + [NEUTRAL] * (len(top5) - 1),
                  edgecolor="white", width=0.55)
    for bar, v in zip(bars, top5.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:,}\n({v/total*100:.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Occurrences")
    ax.set_title("Top Error Codes", fontsize=13, pad=10)
    ax.tick_params(axis="x", rotation=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, top5.values.max() * 1.25)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 4 – Timeline: log entries per minute
# ---------------------------------------------------------------------------
def chart_timeline(df: pd.DataFrame) -> str:
    df2 = df.copy()
    df2["ts"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2["minute"] = df2["ts"].dt.floor("min")
    per_min = (
        df2.dropna(subset=["minute"])
           .groupby("minute")
           .size()
           .reset_index(name="count")
    )

    if per_min.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.text(0.5, 0.5, "No valid timestamps found for timeline analysis",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color=NEUTRAL)
        ax.axis("off")
        return _fig_to_b64(fig)

    p90 = per_min["count"].quantile(0.90)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(per_min["minute"], per_min["count"], alpha=0.20, color=ACCENT)
    ax.plot(per_min["minute"], per_min["count"], color=ACCENT, linewidth=1.3)
    burst = per_min[per_min["count"] >= p90]
    ax.scatter(burst["minute"], burst["count"],
               color=DANGER, zorder=5, s=25,
               label=f"Top-10% burst  (>={int(p90)} entries/min)")

    ax.set_ylabel("Entries / minute")
    ax.set_title("Log Activity Timeline", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M\n%b %d"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Chart 5 – Pattern key counts (bar)
# ---------------------------------------------------------------------------
def _pattern_summary(patterns: dict) -> tuple[list[str], list[int]]:
    """Extract one representative count per pattern."""
    mapping = {
        "event_sequences":            ("total_events",            "event sequence rows"),
        "boolean_flag_distribution":  ("boolean_columns_found",   "boolean flag columns"),
        "timestamp_bursts":           ("burst_timestamps",        "burst timestamps"),
        "code_repetition":            ("columns_analyzed",        "code-like columns"),
        "template_complexity":        ("mean",                    "mean param_count"),
        "categorical_dominance":      ("columns_analyzed",        "categorical columns"),
        "text_signal_summary":        ("signal_total",            "text signal hits"),
        "dataset_overview":           ("total_rows",              "rows analyzed"),
    }
    names, vals = [], []
    for p in patterns["patterns"]:
        key, label = mapping.get(p["pattern"], ("", p["pattern"]))
        if p["pattern"] == "text_signal_summary":
            val = sum(item.get("count", 0) for item in p.get("signals", {}).values())
        else:
            val = p.get(key, 0)
        names.append(label)
        vals.append(float(val))
    return names, vals


def chart_patterns(patterns: dict) -> str:
    names, vals = _pattern_summary(patterns)
    if not vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No pattern summary available", ha="center", va="center",
                transform=ax.transAxes, fontsize=13, color=NEUTRAL)
        ax.axis("off")
        return _fig_to_b64(fig)
    colors = [DANGER if "chain" in n or "HRESULT" in n or "reboot" in n
              else ACCENT for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        label = f"{v:.2f}" if v < 10 else f"{int(v):,}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.012,
                label, ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [textwrap.fill(n, 14) for n in names],
        fontsize=8.5,
    )
    ax.set_ylabel("Count / value")
    ax.set_title("Pattern Engine — Key Count per Pattern", fontsize=13, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Summary statistics table (HTML)
# ---------------------------------------------------------------------------
def build_stats_table(df: pd.DataFrame, patterns: dict) -> str:
    pattern_map = {
        item.get("pattern"): item
        for item in patterns.get("patterns", [])
        if isinstance(item, dict)
    }

    tdata = pattern_map.get("timestamp_bursts", {})
    pcdata = pattern_map.get("template_complexity", {})
    code_data = pattern_map.get("code_repetition", {})
    sequence_data = pattern_map.get("event_sequences", {})
    flag_data = pattern_map.get("boolean_flag_distribution", {})

    hresult_series = df.get("hresult", pd.Series(dtype="object")).fillna("").astype(str)
    hresult_non_empty = hresult_series[hresult_series != ""]
    hresult_counts = hresult_non_empty.value_counts()
    dominant_hresult = hresult_counts.index[0] if not hresult_counts.empty else "None"
    dominant_hresult_count = int(hresult_counts.iloc[0]) if not hresult_counts.empty else 0
    dominant_hresult_pct = round(dominant_hresult_count / len(hresult_non_empty) * 100, 2) if len(hresult_non_empty) else 0

    event_sequences = sequence_data.get("top_3_event_sequences", {})
    top_sequence = next(iter(event_sequences.items()), ("No dominant sequence found", 0))

    kb_non_empty = df.get("kb_package", pd.Series(dtype="object")).fillna("").astype(str)
    kb_non_empty = kb_non_empty[kb_non_empty != ""]
    unique_kb_count = int(kb_non_empty.nunique()) if not kb_non_empty.empty else 0
    total_kb_refs = int(len(kb_non_empty))
    avg_refs_per_kb = round(total_kb_refs / unique_kb_count, 1) if unique_kb_count else 0

    session_non_empty = df.get("session_id", pd.Series(dtype="object")).fillna("").astype(str)
    session_non_empty = session_non_empty[session_non_empty != ""]
    session_events = int((df.get("event_class", pd.Series(dtype="object")) == "SESSION").sum())
    session_pct = round(session_events / len(df) * 100, 2) if len(df) else 0

    validation_text = "Validation results available"
    validation_path = ROOT / "outputs" / "validation_results.json"
    if validation_path.exists():
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
        stats = payload.get("statistics", {})
        validation_text = (
            f"{stats.get('successful_expectations', 0)} / "
            f"{stats.get('evaluated_expectations', 0)} expectations passed"
        )

    ts_series = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts_series.empty:
        date_range = "No valid timestamps available"
    else:
        date_range = f"{ts_series.min().strftime('%Y-%m-%d')}  →  {ts_series.max().strftime('%Y-%m-%d')}"

    rows = [
        ("Total log entries",           f"{len(df):,}"),
        ("Date range",                  date_range),
        ("Unique event IDs",            f"{df['EventId'].nunique()}"),
        ("Unique components",           f"{df['Component'].nunique()}  ({', '.join(df['Component'].unique())})"),
        ("Top 3-event sequence",        f"{top_sequence[0]}  ({top_sequence[1]} occurrences)"),
        ("Burst timestamps (>1 row)",   f"{tdata.get('burst_timestamps', 0)} / {tdata.get('unique_timestamps', 0)}  "
                                        f"({tdata.get('pct_rows_in_bursts', 0)}% of rows in bursts)"),
        ("Largest burst",               f"{tdata.get('largest_burst_count', 0)} rows at {tdata.get('largest_burst_timestamp', 'n/a')}"),
        ("param_count (mean / max)",    f"{pcdata.get('mean', 'n/a')} / {pcdata.get('max', 'n/a')}"),
        ("Boolean flag columns",        str(flag_data.get("boolean_columns_found", 0))),
        ("Code-like columns analyzed",  str(code_data.get("columns_analyzed", 0))),
        ("GE validation result",        validation_text),
    ]

    if len(hresult_non_empty):
        rows.extend([
            ("Rows with error codes",       f"{len(hresult_non_empty):,}  ({(len(hresult_non_empty)/len(df)*100 if len(df) else 0):.2f}%)"),
            ("Dominant error code",         f"{dominant_hresult}  ({dominant_hresult_count} occurrences, "
                                            f"{dominant_hresult_pct}% of coded rows)"),
            ("Unique error codes",          str(int(hresult_non_empty.nunique()))),
        ])

    if session_events or len(session_non_empty):
        rows.extend([
            ("SESSION events",              f"{session_events:,}  ({session_pct}%)"),
            ("Unique session IDs found",    str(int(session_non_empty.nunique()))),
        ])

    if unique_kb_count:
        rows.append(
            ("KB packages referenced",      f"{unique_kb_count} unique  "
                                            f"({total_kb_refs} total refs, avg {avg_refs_per_kb} per KB)")
        )

    html_rows = ""
    for i, (k, v) in enumerate(rows):
        bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        danger = "color:#d62728;font-weight:600;" if any(
            kw in k for kw in ("Failure", "HRESULT", "Orphaned", "health")
        ) else ""
        html_rows += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:8px 14px;font-weight:500;width:42%;">{k}</td>'
            f'<td style="padding:8px 14px;{danger}">{v}</td>'
            f"</tr>\n"
        )

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:14px;
              border:1px solid #dee2e6;border-radius:6px;overflow:hidden;">
  <thead>
    <tr style="background:#343a40;color:#fff;">
      <th style="padding:10px 14px;text-align:left;">Metric</th>
      <th style="padding:10px 14px;text-align:left;">Value</th>
    </tr>
  </thead>
  <tbody>
{html_rows}  </tbody>
</table>"""


# ---------------------------------------------------------------------------
# Assemble the HTML report
# ---------------------------------------------------------------------------
def _build_badges(df: pd.DataFrame, validation_text: str) -> list[tuple[str, str]]:
    failure_pct = float(df.get("has_failure", pd.Series(dtype="bool")).mean() * 100) if "has_failure" in df.columns else 0.0
    warning_pct = float(df.get("has_warning", pd.Series(dtype="bool")).mean() * 100) if "has_warning" in df.columns else 0.0
    badges = [(validation_text, "ok")]
    if failure_pct:
        badges.insert(0, (f"Failure-like rows: {failure_pct:.1f}%", "danger"))
    elif warning_pct:
        badges.insert(0, (f"Warning-like rows: {warning_pct:.1f}%", "warn"))
    return badges


def build_html(charts: dict[str, str], stats_table: str, report_meta: dict) -> str:
    card = (
        'style="background:#fff;border-radius:8px;padding:24px 28px;'
        'box-shadow:0 1px 4px rgba(0,0,0,.10);margin-bottom:28px;"'
    )
    badge_html = " ".join(
        f'<span class="badge badge-{kind}">{label}</span>'
        for label, kind in report_meta["badges"]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Log Quality Pipeline Report</title>
<style>
  body {{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;
        color:#212529;margin:0;padding:0;}}
  .wrap {{max-width:1100px;margin:0 auto;padding:32px 20px;}}
  h1 {{font-size:1.9rem;font-weight:700;margin:0 0 4px;}}
  .subtitle {{color:#6c757d;margin:0 0 32px;font-size:.95rem;}}
  h2 {{font-size:1.15rem;font-weight:600;margin:0 0 16px;color:#343a40;
       border-left:4px solid #1f77b4;padding-left:10px;}}
  .grid2 {{display:grid;grid-template-columns:1fr 1fr;gap:28px;}}
  .badge {{display:inline-block;padding:3px 10px;border-radius:20px;
           font-size:.8rem;font-weight:600;}}
  .badge-danger {{background:#fde;color:#d62728;}}
  .badge-ok    {{background:#dfd;color:#2ca02c;}}
  .badge-warn  {{background:#fff2cc;color:#8a5a00;}}
  @media(max-width:720px){{.grid2{{grid-template-columns:1fr;}}}}
</style>
</head>
<body>
<div class="wrap">

  <h1>Log Quality Pipeline &mdash; Analysis Report</h1>
  <p class="subtitle">
    Source: <code>{report_meta['source_label']}</code> &nbsp;|&nbsp;
    {report_meta['row_count']:,} log entries &nbsp;|&nbsp;
    {report_meta['source_context']} &nbsp;|&nbsp;
    {badge_html}
  </p>

  <!-- Row 1: event class + component skew -->
  <div class="grid2">
    <div {card}>
      <h2>Event Class Distribution</h2>
      {charts['event_class']}
    </div>
    <div {card}>
      <h2>Top Components</h2>
      {charts['component_skew']}
    </div>
  </div>

  <!-- Row 2: HRESULT + timeline -->
  <div {card}>
    <h2>{report_meta['code_chart_title']}</h2>
    {charts['hresult']}
  </div>

  <div {card}>
    <h2>Log Activity Timeline</h2>
    {charts['timeline']}
  </div>

  <!-- Row 3: pattern counts -->
  <div {card}>
    <h2>Pattern Engine &mdash; Key Counts per Pattern</h2>
    {charts['patterns']}
  </div>

  <!-- Row 4: summary table -->
  <div {card}>
    <h2>Summary Statistics</h2>
    {stats_table}
  </div>

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_visualizer(
    parsed_path: Path   = PARSED_CSV,
    patterns_path: Path = PATTERNS_JSON,
    output_path: Path   = OUTPUT_HTML,
) -> Path:
    df       = pd.read_csv(parsed_path)
    df["hresult"]    = df["hresult"].fillna("")
    df["kb_package"] = df["kb_package"].fillna("")
    patterns = json.loads(patterns_path.read_text(encoding="utf-8"))

    print("Generating charts...")
    charts = {
        "event_class":    chart_event_class(df),
        "component_skew": chart_component_skew(df),
        "hresult":        chart_hresult(df),
        "timeline":       chart_timeline(df),
        "patterns":       chart_patterns(patterns),
    }
    print("  5/5 charts rendered")

    stats_table = build_stats_table(df, patterns)
    validation_payload = json.loads((ROOT / "outputs" / "validation_results.json").read_text(encoding="utf-8"))
    validation_stats = validation_payload.get("statistics", {})
    validation_text = (
        f"GE Validation: {validation_stats.get('successful_expectations', 0)}/"
        f"{validation_stats.get('evaluated_expectations', 0)} passed"
    )
    profile = _infer_report_profile(df)
    report_meta = {
        **profile,
        "row_count": len(df),
        "badges": _build_badges(df, validation_text),
        "code_chart_title": "Top Error Codes" if (df["hresult"] != "").any() else "Top Event Templates",
    }
    html = build_html(charts, stats_table, report_meta)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    size_kb = output_path.stat().st_size / 1024
    print(f"Report saved to {output_path}  ({size_kb:.0f} KB)")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_visualizer()
