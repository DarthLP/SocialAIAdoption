"""
Generate README.md files beside DiD figure folders (overview and per-family plot types).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from src.did.outcomes import (
    DEFAULT_FAMILIES,
    FAMILY_FIGURE_DIRS,
    outcome_label,
)
from src.did.specs import PLOT_STRATEGY_GROUPS, strategy_label

OVERVIEW_FIGURES: tuple[tuple[str, str, str], ...] = (
    (
        "significance_heatmap.png",
        "Heatmap of DiD β by outcome (rows) and identification strategy (columns).",
        "Headline + by-topic strategies; TWFE treat×post; * marks p<0.05.",
    ),
    (
        "headline_forest_lexical_semantic.png",
        "Forest plot of pre-registered headline outcomes.",
        f"Strategy: {strategy_label('cross_country_all')}; rows labeled (full ban) vs (early ban) when both post windows are estimated.",
    ),
    (
        "ddd_political_specificity.png",
        "Within-Italy triple-difference (ban × political-tree) across outcomes.",
        f"Strategy: {strategy_label('within_italy_ddd')}.",
    ),
    (
        "first_stage_aiwriting.png",
        "Mean DiD β for AI-writing stylometric outcomes (first stage).",
        "Averaged across strategies where estimated.",
    ),
    (
        "pretrend_summary.png",
        "Joint pre-trend F-test p-values for headline outcomes.",
        f"Strategy: {strategy_label('cross_country_all')}, full_ban only; rows with pretrend_quality=ok "
        "(reliable TWFE β). Dashed line at α=0.05.",
    ),
    (
        "early_ban_net_ideology.png",
        "Full-ban vs early-ban (7d / 14d) windows for net ideology.",
        "Compares post windows on cross-country IT vs control strategies.",
    ),
    (
        "post_phase_net_ideology.png",
        "Short (0–2d), medium (3–9d), and long (10d+) post-ban β for net ideology.",
        f"Strategy: {strategy_label('cross_country_all')}; TWFE treat×post with post indicator set per phase only.",
    ),
    (
        "early_ban_ai_style_rate.png",
        "Full-ban vs early-ban windows for AI style rate.",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_ai_style_rate.png",
        "Post-phase short / medium / long β for AI style rate.",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_sem_axis_ideology.png",
        "Full-ban vs early-ban windows for semantic-axis ideology.",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_sem_axis_ideology.png",
        "Post-phase short / medium / long β for semantic-axis ideology.",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_pole_share.png",
        "Full-ban vs early-ban windows for lexical pole share.",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_pole_share.png",
        "Post-phase β for lexical pole share.",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_pole_rate.png",
        "Full-ban vs early-ban windows for the additive pole-word rate "
        "(left+right hits per 100 words; no center bucket, no ratio ceiling).",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_pole_rate.png",
        "Post-phase β for the additive pole-word rate "
        "(left+right hits per 100 words; no center bucket, no ratio ceiling).",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_esteban_ray.png",
        "Full-ban vs early-ban windows for lexical Esteban–Ray.",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_esteban_ray.png",
        "Post-phase β for lexical Esteban–Ray.",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_sem_axis_ideology_pole_share.png",
        "Full-ban vs early-ban windows for semantic-axis ideology pole share "
        "(absolute cutoffs; IT-scale — controls ~0 by construction).",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_sem_axis_ideology_pole_share.png",
        "Post-phase β for semantic-axis ideology pole share "
        "(absolute cutoffs; IT-scale — controls ~0 by construction).",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_sem_axis_ideology_pole_share_pct.png",
        "Full-ban vs early-ban windows for the per-lexicon p10/p90 percentile "
        "pole share (cross-arm comparable variant).",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_sem_axis_ideology_pole_share_pct.png",
        "Post-phase β for the per-lexicon p10/p90 percentile pole share "
        "(cross-arm comparable variant).",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
    (
        "early_ban_sem_axis_ideology_esteban_ray.png",
        "Full-ban vs early-ban windows for semantic-axis Esteban–Ray.",
        "Same strategy set as other early-ban overview plots.",
    ),
    (
        "post_phase_sem_axis_ideology_esteban_ray.png",
        "Post-phase β for semantic-axis Esteban–Ray.",
        f"Strategy: {strategy_label('cross_country_all')}.",
    ),
)

PLOT_TYPE_README: dict[str, tuple[str, str]] = {
    "coefplots_headline": (
        "Headline strategy comparison (post phases)",
        "TWFE DiD β for headline strategies with post indicator restricted to short (0–2d), medium (3–9d), or long (10d+) ban windows; see did_summary spec column. Forest/heatmap use full_ban; early_ban_7d/14d rows remain in did_summary.",
    ),
    "coefplots_full": (
        "All strategies",
        "TWFE DiD β across all estimated strategies for this outcome family.",
    ),
    "event_study": (
        "Event study",
        "Dynamic γ_k coefficients (treated×relative day); ref day −1; ban at k=0. "
        "Thesis overlay `{outcome}_events.png` (e.g. `sem_axis_emotion_events.png`) adds "
        "Italian political-event vertical markers on the same estimates.",
    ),
    "robustness": (
        "Robustness checks",
        "Placebo launch and alternative specifications vs baseline cross-country DiD.",
    ),
}


def _section(filename: str, what: str, estimation: str, notes: str = "") -> List[str]:
    """Function summary: format one figure entry as markdown lines."""
    lines = [
        f"### `{filename}`",
        "",
        f"- **What it shows:** {what}",
        f"- **Estimation:** {estimation}",
    ]
    if notes:
        lines.append(f"- **Notes:** {notes}")
    lines.append("")
    return lines


def write_overview_readme(overview_dir: Path) -> None:
    """Function summary: write overview/README.md describing cross-family diagnostic PNGs.

    Parameters:
    - overview_dir: path to did/overview/ under figures root.
    """
    overview_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DiD overview figures",
        "",
        "Generated by `scripts/analysis/did_event_study.py` (`generate_overview_figures`).",
        "Plot axes use short labels; full strategy and outcome names appear below.",
        "",
        "## Strategy glossary (full labels)",
        "",
    ]
    for sid in PLOT_STRATEGY_GROUPS["headline"] + PLOT_STRATEGY_GROUPS["by_topic"]:
        lines.append(f"- `{sid}`: {strategy_label(sid)}")
    lines.extend(["", "## Figures", ""])
    for filename, what, estimation in OVERVIEW_FIGURES:
        if filename.startswith("early_ban_") or filename.startswith("post_phase_"):
            if filename.startswith("early_ban_"):
                oid = filename.replace("early_ban_", "").replace(".png", "")
            else:
                oid = filename.replace("post_phase_", "").replace(".png", "")
            full_oid = outcome_label(oid, short=False)
            lines.extend(
                _section(
                    filename,
                    what,
                    estimation,
                    notes=f"Outcome: {full_oid} ({oid}).",
                )
            )
        else:
            lines.extend(_section(filename, what, estimation))
    (overview_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_did_figures_index(fig_dir: Path) -> None:
    """Function summary: write did/README.md index under the figures root.

    Parameters:
    - fig_dir: did figures root (e.g. results/figures/<study>/did/).
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DiD figures",
        "",
        "Subfolders: `overview/` (cross-family diagnostics) and per-outcome-family trees.",
        "Each plot subdirectory includes a `README.md` describing its PNGs.",
        "",
        "## Families",
        "",
    ]
    for fam in DEFAULT_FAMILIES:
        sub = FAMILY_FIGURE_DIRS.get(fam, fam)
        lines.append(f"- `{sub}/` — {fam}")
    lines.append("")
    (fig_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_coefplots_readme(
    plot_dir: Path,
    family: str,
    plot_type: str,
    outcome_ids: Iterable[str],
) -> None:
    """Function summary: write README for a family plot-type folder listing each PNG.

    Parameters:
    - plot_dir: e.g. lexical/coefplots_headline/.
    - family: outcome family id.
    - plot_type: coefplots_headline, event_study, etc.
    - outcome_ids: outcomes that were estimated (for listing).
    """
    title, estimation = PLOT_TYPE_README.get(
        plot_type,
        (plot_type.replace("_", " ").title(), "See did_summary.csv."),
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {family} — {title}",
        "",
        f"**Estimation:** {estimation}",
        "",
    ]
    family_notes = _family_readme_notes(family)
    if family_notes:
        lines.extend(["## Family notes", "", family_notes, ""])
    if plot_type in ("coefplots_headline", "coefplots_full"):
        lines.extend(["## Strategies (headline set)", ""])
        strats = (
            PLOT_STRATEGY_GROUPS["headline"]
            if plot_type == "coefplots_headline"
            else ()
        )
        if strats:
            for sid in strats:
                lines.append(f"- `{sid}`: {strategy_label(sid)}")
        else:
            lines.append("- All strategies present in `did_summary.csv`.")
        lines.append("")
    lines.append("## Files")
    lines.append("")
    oids = sorted(set(outcome_ids))
    if not oids:
        lines.append("(no figures written in last run)")
    else:
        for oid in oids:
            fname = f"{oid}.png"
            if plot_type == "robustness":
                fname = f"placebo_{oid}.png"
            lines.extend(
                _section(
                    fname,
                    f"{outcome_label(oid)} (`{oid}`).",
                    estimation,
                )
            )
    (plot_dir / "README.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _family_readme_notes(family: str) -> str:
    """Function summary: family-specific caveats for README prose."""
    if family == "wordfish_forum":
        return "Cross-language comparisons: interpret sign/direction only."
    if family == "wordfish_forum_v2":
        return "Forum θ (v2); cross-language sign/direction only; validation gate may fail."
    if family == "wordfish_author":
        return "Italian-writing authors; week bins."
    if family == "wordfish_author_v2":
        return "Author θ (v2); validation-gated ideology interpretation."
    if family == "semantic_axis_author_week":
        return "Author×week semantic means; cross-language sign/direction only on levels."
    if family == "lexical_comment":
        return "Comment-level TWFE: author + calendar FE (pyfixest); cluster SEs at author."
    if family == "semantic_axis_comment":
        return "Comment-level semantic axes; author + day FE; political-universe sample."
    if family in ("lexical_author_day", "semantic_axis_author_day"):
        return "Author×day weighted means; PanelOLS TWFE robustness to comment-level pyfixest."
    return ""


def write_all_family_readmes(
    fig_dir: Path,
    families: Sequence[str],
    outcome_ids_by_family: dict[str, Sequence[str]],
    *,
    full_coefplots: bool = False,
) -> None:
    """Function summary: refresh README.md for each family plot-type subdirectory.

    Parameters:
    - fig_dir: did figures root.
    - families: outcome families that were run.
    - outcome_ids_by_family: outcome_id lists per family.
    - full_coefplots: if True, also write coefplots_full README.
    """
    write_did_figures_index(fig_dir)
    plot_types = ["coefplots_headline", "event_study", "robustness"]
    if full_coefplots:
        plot_types.insert(1, "coefplots_full")
    for fam in families:
        sub = FAMILY_FIGURE_DIRS.get(fam, fam)
        oids = outcome_ids_by_family.get(fam, ())
        for pt in plot_types:
            write_coefplots_readme(fig_dir / sub / pt, fam, pt, oids)
