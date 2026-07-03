"""
MODUL 7 - Policy Recommendation & EWS Dashboard
=================================================
Mensintesis hasil Modul 5 (efektivitas intervensi) dan Modul 6
(robustness/confidence) menjadi rekomendasi operasional EWS.

Fungsi utama:
    run_module7_policy_recommendation(
        intervention_results_path, robustness_results_path,
        output_dir, config
    ) -> dict

Output:
- data/processed/module_07/policy_recommendation_results.json
- data/processed/module_07/policy_recommendation_summary.csv
- data/processed/module_07/policy_recommendation_summary.md
- data/processed/module_07/ews_dashboard.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict = {
    # EWS trigger thresholds (impact index)
    "red_impact_threshold": 4.0,       # cumulative_impact >= threshold -> RED
    "yellow_impact_threshold": 2.0,    # cumulative_impact >= threshold -> YELLOW
    # Vulnerability concentration: % of total impact absorbed by top-1 node
    "red_vulnerability_threshold": 0.50,
    "yellow_vulnerability_threshold": 0.30,
    # Minimum impact reduction % to qualify as a recommended intervention
    "min_impact_reduction_pct": 5.0,
    # Rule version for audit trail
    "rule_version": "v1.0",
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Format {label} tidak valid (harus dict).")
    return data


def _save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# EWS trigger rules
# ---------------------------------------------------------------------------


def _determine_trigger_level(
    cumulative_impact: float,
    vulnerability_concentration: float,
    confidence: str,
    cfg: Dict,
) -> str:
    """
    Determine EWS trigger level: GREEN / YELLOW / RED.

    Rules (impact index + vulnerability concentration):
      RED    : cumulative_impact >= red_impact_threshold
               OR vulnerability_concentration >= red_vulnerability_threshold
               (downgrade to YELLOW if confidence == 'LOW')
      YELLOW : cumulative_impact >= yellow_impact_threshold
               OR vulnerability_concentration >= yellow_vulnerability_threshold
      GREEN  : otherwise
    """
    red_impact = cfg.get("red_impact_threshold", DEFAULT_CONFIG["red_impact_threshold"])
    yellow_impact = cfg.get("yellow_impact_threshold", DEFAULT_CONFIG["yellow_impact_threshold"])
    red_vuln = cfg.get("red_vulnerability_threshold", DEFAULT_CONFIG["red_vulnerability_threshold"])
    yellow_vuln = cfg.get("yellow_vulnerability_threshold", DEFAULT_CONFIG["yellow_vulnerability_threshold"])

    is_red = (cumulative_impact >= red_impact) or (vulnerability_concentration >= red_vuln)
    is_yellow = (cumulative_impact >= yellow_impact) or (vulnerability_concentration >= yellow_vuln)

    if is_red:
        # Downgrade if confidence is LOW (not enough evidence to sound full alarm)
        level = "YELLOW" if confidence == "LOW" else "RED"
    elif is_yellow:
        level = "YELLOW"
    else:
        level = "GREEN"

    return level


# ---------------------------------------------------------------------------
# Per-grade synthesis
# ---------------------------------------------------------------------------


def _synthesize_grade(
    grade: str,
    m5_grade: Dict,
    m6_grade: Dict,
    cfg: Dict,
) -> Dict:
    """Synthesize policy recommendation for a single grade."""

    # --- Extract key metrics from Modul 5 ---
    baseline = m5_grade.get("baseline", {})
    cumulative_impact = float(baseline.get("cumulative_impact_total", 0.0))

    most_vulnerable: List[Dict] = baseline.get("most_vulnerable_nodes", [])
    most_influential: List[Dict] = baseline.get("most_influential_sources", [])

    # Vulnerability concentration: top-1 node's share of total received impact
    if most_vulnerable:
        top1_impact = float(most_vulnerable[0].get("cumulative_received_impact", 0.0))
        vuln_concentration = (top1_impact / cumulative_impact) if cumulative_impact > 0 else 0.0
    else:
        vuln_concentration = 0.0

    priority_markets: List[str] = [n["node"] for n in most_vulnerable[:5] if "node" in n]

    # Best interventions from ranking
    ranking: List[Dict] = m5_grade.get("ranking", [])
    min_reduction = cfg.get("min_impact_reduction_pct", DEFAULT_CONFIG["min_impact_reduction_pct"])
    priority_interventions: List[str] = [
        r["scenario"]
        for r in ranking
        if float(r.get("impact_reduction_pct", 0.0)) >= min_reduction
    ]
    if not priority_interventions and ranking:
        # Fall back to best available even if below threshold
        best = max(ranking, key=lambda r: float(r.get("impact_reduction_pct", 0.0)))
        priority_interventions = [best["scenario"]]

    best_reduction_pct = max(
        (float(r.get("impact_reduction_pct", 0.0)) for r in ranking), default=0.0
    )
    expected_impact = f"{best_reduction_pct:.1f}% pengurangan dampak kumulatif"

    # --- Extract confidence from Modul 6 ---
    confidence_level: str = m6_grade.get("confidence_level", "LOW")
    stability: Dict = m6_grade.get("stability_metrics", {})
    stability_score: float = float(stability.get("stability_score", 0.0))

    # --- EWS trigger ---
    trigger_level = _determine_trigger_level(
        cumulative_impact=cumulative_impact,
        vulnerability_concentration=vuln_concentration,
        confidence=confidence_level,
        cfg=cfg,
    )

    # --- Operational recommendations ---
    short_term_actions, medium_term_actions = _build_actions(
        trigger_level=trigger_level,
        priority_markets=priority_markets,
        priority_interventions=priority_interventions,
        confidence_level=confidence_level,
    )

    weekly_monitoring = _build_monitoring_indicators(
        most_vulnerable=most_vulnerable,
        most_influential=most_influential,
    )

    return {
        "grade": grade,
        "trigger_level": trigger_level,
        "priority_markets": priority_markets,
        "priority_interventions": priority_interventions,
        "expected_impact": expected_impact,
        "confidence_level": confidence_level,
        "stability_score": round(stability_score, 4),
        "cumulative_impact_index": round(cumulative_impact, 4),
        "vulnerability_concentration": round(vuln_concentration, 4),
        "short_term_actions": short_term_actions,
        "medium_term_actions": medium_term_actions,
        "weekly_monitoring_indicators": weekly_monitoring,
    }


def _build_actions(
    trigger_level: str,
    priority_markets: List[str],
    priority_interventions: List[str],
    confidence_level: str,
) -> tuple[List[str], List[str]]:
    top = priority_markets[0] if priority_markets else "pasar utama"
    interventions_str = ", ".join(priority_interventions) if priority_interventions else "node_attenuation"

    if trigger_level == "RED":
        short = [
            f"Aktivasi protokol darurat stok cadangan di {top}",
            f"Terapkan intervensi segera: {interventions_str}",
            "Koordinasi lintas pasar untuk mencegah contagion lebih lanjut",
            "Lapor ke pemangku kebijakan dalam 24 jam",
        ]
        medium = [
            "Evaluasi rantai pasok dan identifikasi bottleneck struktural",
            "Perkuat buffer stock di pasar-pasar paling rentan",
            "Tinjau ulang perjanjian distribusi antar pasar",
            "Implementasi sistem pemantauan harga real-time",
        ]
    elif trigger_level == "YELLOW":
        short = [
            f"Pantau ketat pergerakan harga di {top}",
            f"Siapkan skenario intervensi: {interventions_str}",
            "Tingkatkan frekuensi pengumpulan data pasar menjadi harian",
        ]
        medium = [
            "Perkuat koordinasi antar agen distribusi",
            "Lakukan pre-positioning stok di pasar rentan",
            "Kembangkan early warning checklist untuk petugas lapangan",
        ]
    else:  # GREEN
        short = [
            "Lanjutkan pemantauan rutin mingguan",
            f"Verifikasi stabilitas jaringan di {top}",
        ]
        medium = [
            "Dokumentasi praktik terbaik distribusi saat ini",
            "Persiapkan skenario kontingensi sebagai cadangan",
        ]

    if confidence_level == "LOW":
        short.append("⚠ Confidence rendah — perkuat validasi data sebelum aksi besar")

    return short, medium


def _build_monitoring_indicators(
    most_vulnerable: List[Dict],
    most_influential: List[Dict],
) -> List[str]:
    indicators = []
    for entry in most_vulnerable[:3]:
        node = entry.get("node", "")
        if node:
            indicators.append(f"Indeks harga mingguan: {node}")
    for entry in most_influential[:2]:
        node = entry.get("node", "")
        if node:
            indicators.append(f"Volume transaksi: {node}")
    indicators.append("Rasio stok/permintaan pasar utama")
    indicators.append("Deviasi harga terhadap rata-rata 4 minggu")
    return indicators


# ---------------------------------------------------------------------------
# Outputs: JSON, CSV, Markdown
# ---------------------------------------------------------------------------


def _build_summary_df(per_grade_recs: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for grade, rec in per_grade_recs.items():
        rows.append(
            {
                "grade": grade,
                "trigger_level": rec["trigger_level"],
                "confidence_level": rec["confidence_level"],
                "stability_score": rec["stability_score"],
                "cumulative_impact_index": rec["cumulative_impact_index"],
                "vulnerability_concentration": rec["vulnerability_concentration"],
                "priority_markets": "; ".join(rec["priority_markets"]),
                "priority_interventions": "; ".join(rec["priority_interventions"]),
                "expected_impact": rec["expected_impact"],
            }
        )
    return pd.DataFrame(rows)


def _save_markdown(df: pd.DataFrame, path: Path, generated_at: str, rule_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Modul 7 – Policy Recommendation Summary\n",
        f"**Generated at:** {generated_at}  ",
        f"**Rule version:** {rule_version}  ",
        f"**Grade count:** {len(df)}  \n",
        df.to_markdown(index=False),
        "\n",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# EWS Dashboard HTML
# ---------------------------------------------------------------------------

_TRIGGER_COLOR = {
    "RED": ("#e74c3c", "#fff"),
    "YELLOW": ("#f39c12", "#000"),
    "GREEN": ("#27ae60", "#fff"),
}


def _grade_card_html(rec: Dict) -> str:
    level = rec["trigger_level"]
    bg, fg = _TRIGGER_COLOR.get(level, ("#95a5a6", "#fff"))
    short_actions = "".join(f"<li>{a}</li>" for a in rec["short_term_actions"])
    medium_actions = "".join(f"<li>{a}</li>" for a in rec["medium_term_actions"])
    monitoring = "".join(f"<li>{m}</li>" for m in rec["weekly_monitoring_indicators"])
    interventions = ", ".join(rec["priority_interventions"]) or "—"
    markets = ", ".join(rec["priority_markets"]) or "—"

    return f"""
<div class="grade-card" id="grade-{rec['grade']}">
  <div class="card-header" style="background:{bg};color:{fg};">
    <span class="grade-label">Grade: {rec['grade']}</span>
    <span class="trigger-badge">{level}</span>
  </div>
  <div class="card-body">
    <table class="metrics-table">
      <tr><th>Confidence</th><td>{rec['confidence_level']}</td></tr>
      <tr><th>Stability Score</th><td>{rec['stability_score']:.4f}</td></tr>
      <tr><th>Cumulative Impact Index</th><td>{rec['cumulative_impact_index']:.4f}</td></tr>
      <tr><th>Vulnerability Concentration</th><td>{rec['vulnerability_concentration']:.4f}</td></tr>
      <tr><th>Priority Markets</th><td>{markets}</td></tr>
      <tr><th>Priority Interventions</th><td>{interventions}</td></tr>
      <tr><th>Expected Impact</th><td>{rec['expected_impact']}</td></tr>
    </table>
    <div class="section-title">Aksi Jangka Pendek</div>
    <ul>{short_actions}</ul>
    <div class="section-title">Aksi Jangka Menengah</div>
    <ul>{medium_actions}</ul>
    <div class="section-title">Indikator Monitoring Mingguan</div>
    <ul>{monitoring}</ul>
  </div>
</div>
"""


def _build_dashboard_html(
    per_grade_recs: Dict[str, Dict],
    generated_at: str,
    grade_count: int,
    rule_version: str,
) -> str:
    cards = "\n".join(_grade_card_html(rec) for rec in per_grade_recs.values())

    # Build summary bar
    level_counts: Dict[str, int] = {"RED": 0, "YELLOW": 0, "GREEN": 0}
    for rec in per_grade_recs.values():
        lv = rec.get("trigger_level", "GREEN")
        level_counts[lv] = level_counts.get(lv, 0) + 1

    summary_pills = "".join(
        f'<span class="pill" style="background:{_TRIGGER_COLOR[lv][0]};color:{_TRIGGER_COLOR[lv][1]};">'
        f"{lv}: {cnt}</span>"
        for lv, cnt in level_counts.items()
    )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>RiceEWS – Policy Recommendation Dashboard</title>
  <style>
    body {{font-family:Arial,sans-serif;margin:0;padding:0;background:#f5f6fa;color:#2c3e50;}}
    .topbar {{background:#2c3e50;color:#fff;padding:14px 24px;display:flex;
              justify-content:space-between;align-items:center;}}
    .topbar h1 {{margin:0;font-size:1.3rem;}}
    .meta {{font-size:0.8rem;opacity:0.75;}}
    .summary-bar {{background:#ecf0f1;padding:12px 24px;display:flex;gap:10px;align-items:center;}}
    .summary-bar span {{font-size:0.9rem;margin-right:8px;}}
    .pill {{border-radius:12px;padding:4px 14px;font-weight:bold;font-size:0.85rem;}}
    .grid {{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));
            gap:20px;padding:24px;}}
    .grade-card {{border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.12);
                  overflow:hidden;background:#fff;}}
    .card-header {{padding:14px 20px;display:flex;justify-content:space-between;align-items:center;}}
    .grade-label {{font-size:1.05rem;font-weight:bold;}}
    .trigger-badge {{border-radius:20px;padding:4px 14px;font-weight:bold;
                     border:2px solid rgba(255,255,255,.5);font-size:0.9rem;}}
    .card-body {{padding:16px 20px;}}
    .metrics-table {{width:100%;border-collapse:collapse;font-size:0.88rem;margin-bottom:12px;}}
    .metrics-table th {{text-align:left;width:55%;padding:4px 0;color:#7f8c8d;font-weight:normal;}}
    .metrics-table td {{padding:4px 0;font-weight:bold;}}
    .section-title {{font-size:0.8rem;text-transform:uppercase;letter-spacing:.06em;
                     color:#95a5a6;margin:10px 0 4px;}}
    ul {{margin:0 0 8px 0;padding-left:18px;font-size:0.87rem;line-height:1.6;}}
    .debug-meta {{background:#ecf0f1;padding:12px 24px;font-size:0.8rem;color:#7f8c8d;}}
    .rule-legend {{background:#fff;margin:0 24px 24px;border-radius:8px;
                   padding:16px 20px;box-shadow:0 2px 6px rgba(0,0,0,.07);}}
    .rule-legend h3 {{margin-top:0;font-size:1rem;}}
    .rule-legend table {{border-collapse:collapse;width:100%;font-size:0.87rem;}}
    .rule-legend th,td {{padding:6px 10px;border:1px solid #dde;text-align:left;}}
    .rule-legend th {{background:#ecf0f1;}}
  </style>
</head>
<body>
<div class="topbar">
  <h1>🌾 RiceEWS – Policy Recommendation Dashboard</h1>
  <span class="meta">Modul 7 | Rule: {rule_version}</span>
</div>
<div class="summary-bar">
  <span>Ringkasan Trigger:</span>
  {summary_pills}
</div>
<div class="rule-legend">
  <h3>EWS Trigger Rules</h3>
  <table>
    <tr><th>Level</th><th>Kondisi</th><th>Tindakan Utama</th></tr>
    <tr style="background:#fdecea;">
      <td><strong>🔴 RED</strong></td>
      <td>Impact Index ≥ 4.0 ATAU Vulnerability Concentration ≥ 50%<br/>
          (jika confidence LOW → downgrade ke YELLOW)</td>
      <td>Aktivasi darurat, intervensi segera, eskalasi ke pemangku kebijakan</td>
    </tr>
    <tr style="background:#fff8e1;">
      <td><strong>🟡 YELLOW</strong></td>
      <td>Impact Index ≥ 2.0 ATAU Vulnerability Concentration ≥ 30%</td>
      <td>Pemantauan ketat, siapkan skenario intervensi</td>
    </tr>
    <tr style="background:#e8f8f0;">
      <td><strong>🟢 GREEN</strong></td>
      <td>Selain kondisi di atas</td>
      <td>Pemantauan rutin mingguan</td>
    </tr>
  </table>
</div>
<div class="grid">
{cards}
</div>
<div class="debug-meta">
  <strong>Debug metadata:</strong>
  generated_at={generated_at} |
  grade_count={grade_count} |
  rule_version={rule_version}
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_module7_policy_recommendation(
    intervention_results_path: str,
    robustness_results_path: str,
    output_dir: str,
    config: Optional[Dict] = None,
    shock_propagation_results_path: Optional[str] = None,
) -> Dict:
    """
    Synthesise Module 5 & 6 outputs into operational EWS policy recommendations.

    Parameters
    ----------
    intervention_results_path : str
        Path to data/processed/module_05/intervention_results.json
    robustness_results_path : str
        Path to data/processed/module_06/robustness_results.json
    output_dir : str
        Directory for module 7 outputs.
    config : dict | None
        Override default thresholds (see DEFAULT_CONFIG).
    shock_propagation_results_path : str | None
        Optional path to module_04 output (currently reserved for future use).

    Returns
    -------
    dict
        Full results including per_grade recommendations and file paths.
    """
    cfg: Dict = {**DEFAULT_CONFIG, **(config or {})}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rule_version: str = cfg.get("rule_version", DEFAULT_CONFIG["rule_version"])

    # Load inputs
    m5 = _load_json(Path(intervention_results_path), "intervention_results.json")
    m6 = _load_json(Path(robustness_results_path), "robustness_results.json")

    grades: List[str] = m5.get("grades", list(m5.get("per_grade", {}).keys()))

    per_grade_recs: Dict[str, Dict] = {}
    for grade in grades:
        m5_grade = m5.get("per_grade", {}).get(grade, {})
        m6_grade = m6.get("per_grade", {}).get(grade, {})
        per_grade_recs[grade] = _synthesize_grade(grade, m5_grade, m6_grade, cfg)

    # --- Save JSON ---
    results = {
        "module": "module_07_policy_recommendation",
        "input_intervention_results": intervention_results_path,
        "input_robustness_results": robustness_results_path,
        "output_dir": output_dir,
        "grades": grades,
        "config": cfg,
        "generated_at": generated_at,
        "rule_version": rule_version,
        "per_grade": per_grade_recs,
    }
    json_path = out_dir / "policy_recommendation_results.json"
    _save_json(results, json_path)

    # --- Save CSV ---
    df = _build_summary_df(per_grade_recs)
    csv_path = out_dir / "policy_recommendation_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # --- Save Markdown ---
    md_path = out_dir / "policy_recommendation_summary.md"
    _save_markdown(df, md_path, generated_at, rule_version)

    # --- Save HTML dashboard ---
    html_content = _build_dashboard_html(
        per_grade_recs=per_grade_recs,
        generated_at=generated_at,
        grade_count=len(grades),
        rule_version=rule_version,
    )
    html_path = out_dir / "ews_dashboard.html"
    html_path.write_text(html_content, encoding="utf-8")

    results["outputs"] = {
        "policy_recommendation_results.json": str(json_path),
        "policy_recommendation_summary.csv": str(csv_path),
        "policy_recommendation_summary.md": str(md_path),
        "ews_dashboard.html": str(html_path),
    }

    print(f"✓ Modul 7 selesai. Output di: {out_dir}")
    print(f"  grades: {grades}")
    for grade, rec in per_grade_recs.items():
        print(
            f"  {grade}: trigger={rec['trigger_level']}, "
            f"confidence={rec['confidence_level']}, "
            f"stability={rec['stability_score']:.4f}"
        )

    return results
