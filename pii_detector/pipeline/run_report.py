"""
run_report.py — Generate all output artefacts from a completed detection run.

Outputs:
  1. detailed_report.csv   — full 28-column technical output + timing
  2. summary_report.csv    — simple column → PII / SAFE / EXCLUDED table
  3. timing_report.txt     — ASCII box-art per-module runtime breakdown
  4. masked_data.csv       — original CSV with PII columns AES-masked
                             (only produced when password is supplied)
"""

from __future__ import annotations

import os
import textwrap
from typing import List, Optional

import pandas as pd

from pii_detector.config.settings import SCRIPT_VERSION, OUTPUT_SCHEMA_VERSION


# ─── 1. Detailed Report ───────────────────────────────────────────────────────

def save_detailed_report(results: List[dict], path: str) -> None:
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


# ─── 2. Summary Report ───────────────────────────────────────────────────────

def save_summary_report(results: List[dict], path: str) -> pd.DataFrame:
    """
    Produces a 3-row transposed table:

        Column_Name  |  person_name  |  email_contact  |  row_number  | ...
        PII_Status   |  ✅ PII       |  ✅ PII         |  ✅ SAFE     | ...
        Entity_Type  |  Person Name  |  Email Address  |  —           | ...
    """
    rows = {
        "Column_Name": [],
        "PII_Status":  [],
        "Entity_Type": [],
        "Policy_Action": [],
    }
    for r in results:
        col    = r.get("Column_Name", "")
        action = r.get("Policy_Action", "NONE")
        entity = r.get("Final_Entity_Type") or r.get("Primary_Entity") or "—"

        if action == "PROTECT":
            status = "🔴 PII DETECTED"
        elif action == "DETECTED_EXCLUDED":
            status = "🟡 DETECTED (excluded by policy)"
        else:
            status = "🟢 SAFE"

        rows["Column_Name"].append(col)
        rows["PII_Status"].append(status)
        rows["Entity_Type"].append(entity if action != "NONE" else "—")
        rows["Policy_Action"].append(action)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ─── 3. Timing Report ────────────────────────────────────────────────────────

_BAR_WIDTH = 20   # characters for the bar chart inside the box


def _bar(value: float, max_val: float, width: int = _BAR_WIDTH) -> str:
    if max_val == 0:
        return "░" * width
    filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def save_timing_report(results: List[dict], path: str) -> str:
    """Write a plain-text ASCII timing report and return it as a string."""
    lines: List[str] = []

    BOX = 62  # total box inner width

    def line(text: str = "") -> str:
        return f"║ {text:<{BOX}} ║"

    def div() -> str:
        return "╠" + "═" * (BOX + 2) + "╣"

    def header() -> str:
        return "╔" + "═" * (BOX + 2) + "╗"

    def footer() -> str:
        return "╚" + "═" * (BOX + 2) + "╝"

    # Gather global totals
    total_presidio = sum(r.get("Time_Presidio_sec",   0.0) for r in results)
    total_regex    = sum(r.get("Time_Regex_sec",       0.0) for r in results)
    total_gliner   = sum(r.get("Time_GLiNER_sec",      0.0) for r in results)
    total_agg      = sum(r.get("Time_Aggregation_sec", 0.0) for r in results)
    grand_total    = total_presidio + total_regex + total_gliner + total_agg

    pii_cols  = sum(1 for r in results if r.get("Policy_Action") == "PROTECT")
    excl_cols = sum(1 for r in results if r.get("Policy_Action") == "DETECTED_EXCLUDED")
    safe_cols = sum(1 for r in results if r.get("Policy_Action", "NONE") == "NONE")
    total_cols = len(results)

    lines.append(header())
    lines.append(line("  ENTERPRISE PII DETECTOR — ENGINE RUNTIME REPORT"))
    lines.append(line(f"  Detector v{SCRIPT_VERSION}   Schema v{OUTPUT_SCHEMA_VERSION}"))
    lines.append(div())
    lines.append(line("  SUMMARY"))
    lines.append(line(f"  Total columns scanned : {total_cols}"))
    lines.append(line(f"  🔴 PII detected       : {pii_cols}"))
    lines.append(line(f"  🟡 Detected/excluded  : {excl_cols}"))
    lines.append(line(f"  🟢 Safe               : {safe_cols}"))
    lines.append(div())
    lines.append(line("  TOTAL ENGINE RUNTIMES"))

    max_t = max(total_presidio, total_regex, total_gliner, total_agg, 0.001)
    stages = [
        ("🔍 Presidio NLP scan   ", total_presidio),
        ("🔎 Regex pattern scan  ", total_regex),
        ("🤖 GLiNER AI inference ", total_gliner),
        ("⚙  Aggregation logic   ", total_agg),
    ]
    for label, t in stages:
        bar  = _bar(t, max_t)
        lines.append(line(f"  {label}  {t:6.2f}s  {bar}"))

    lines.append(line(f"  {'─'*54}"))
    lines.append(line(f"  ✅ Grand Total          {grand_total:6.2f}s"))
    lines.append(div())
    lines.append(line("  PER-COLUMN BREAKDOWN"))
    lines.append(line())

    max_col_t = max(
        (r.get("Time_Presidio_sec",0)+r.get("Time_Regex_sec",0)+
         r.get("Time_GLiNER_sec",0)+r.get("Time_Aggregation_sec",0))
        for r in results
    ) if results else 0.001

    for r in results:
        col    = r.get("Column_Name", "?")
        action = r.get("Policy_Action", "NONE")
        entity = r.get("Final_Entity_Type") or r.get("Primary_Entity") or "—"
        t_p    = r.get("Time_Presidio_sec",   0.0)
        t_r    = r.get("Time_Regex_sec",       0.0)
        t_g    = r.get("Time_GLiNER_sec",      0.0)
        t_a    = r.get("Time_Aggregation_sec", 0.0)
        t_tot  = t_p + t_r + t_g + t_a

        icon = "🔴" if action == "PROTECT" else ("🟡" if action == "DETECTED_EXCLUDED" else "🟢")
        header_txt = f"  {icon}  {col[:30]:<30}  {entity[:18]:<18}"
        lines.append(line(header_txt))

        max_stage = max(t_p, t_r, t_g, t_a, 0.001)
        sub_stages = [
            ("   🔍 Presidio  ", t_p),
            ("   🔎 Regex     ", t_r),
            ("   🤖 GLiNER   ", t_g),
            ("   ⚙  Aggregate", t_a),
        ]
        for slabel, st in sub_stages:
            bar = _bar(st, max_stage, width=12)
            lines.append(line(f"{slabel}  {st:5.2f}s  {bar}"))
        lines.append(line(f"   {'─'*54}"))
        lines.append(line(f"   Total  {t_tot:5.2f}s   {_bar(t_tot, max_col_t)}"))
        lines.append(line())

    lines.append(footer())
    report = "\n".join(lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ─── 4. Masked Data ───────────────────────────────────────────────────────────

def save_masked_data(
    original_df: pd.DataFrame,
    results: List[dict],
    path: str,
    password: str,
) -> pd.DataFrame:
    """
    Return (and save) a copy of original_df where every PII column is
    AES-256-GCM masked. A manifest header row is NOT inserted (would break
    downstream CSV parsers); instead the masked columns are annotated in
    the column header as  <colname> [AES256-MASKED].
    """
    from pii_detector.masking.aes_masker import mask_series

    masked_df = original_df.copy()
    rename_map = {}

    for r in results:
        if r.get("Policy_Action") == "PROTECT":
            col = r["Column_Name"]
            if col in masked_df.columns:
                masked_df[col] = mask_series(masked_df[col], password)
                rename_map[col] = f"{col} [AES256-MASKED]"

    masked_df.rename(columns=rename_map, inplace=True)
    masked_df.to_csv(path, index=False)
    return masked_df


# ─── Master Runner ────────────────────────────────────────────────────────────

def generate_all_reports(
    results:     List[dict],
    original_df: pd.DataFrame,
    output_dir:  str,
    mask_password: Optional[str] = None,
) -> dict:
    """
    Write all report files to output_dir.

    Returns a dict mapping report_name → file_path.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = {}

    # 1. Detailed
    p = os.path.join(output_dir, "detailed_report.csv")
    save_detailed_report(results, p)
    paths["detailed_report"] = p

    # 2. Summary
    p = os.path.join(output_dir, "summary_report.csv")
    save_summary_report(results, p)
    paths["summary_report"] = p

    # 3. Timing
    p = os.path.join(output_dir, "timing_report.txt")
    save_timing_report(results, p)
    paths["timing_report"] = p

    # 4. Masked (only if password provided)
    if mask_password:
        p = os.path.join(output_dir, "masked_data.csv")
        save_masked_data(original_df, results, p, mask_password)
        paths["masked_data"] = p

    return paths
