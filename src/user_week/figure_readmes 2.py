"""
Generate README.md files beside user-week figure folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


COMPOSITE_SLUGS: tuple[str, ...] = ("polarization", "style", "semantic")


def _write_readme(path: Path, lines: Sequence[str]) -> None:
    """Function summary: write README if parent directory exists."""
    if not path.parent.is_dir():
        return
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_composite_readme(fig_dir: Path, cohort: str, slug: str) -> None:
    """Function summary: README for dist/spaghetti/component figures under one composite slug.

    Parameters:
    - fig_dir: e.g. figures/.../user_week/strict/polarization.
    - cohort: strict or loose.
    - slug: polarization, style, or semantic.
    """
    track = {
        "polarization": "Lexical / polarization features (rates, net_ideology, pole_share).",
        "style": "AI-writing style rates and composite.",
        "semantic": "Semantic axes (sem_axis_*); interpret shifts within language, not levels across languages.",
    }.get(slug, slug)
    lines = [
        f"# User-week figures — {cohort} / {slug}",
        "",
        "**Estimand:** Within-person pre vs post around Italy ChatGPT ban (`event_window.launch_day_utc`).",
        "Descriptive shifts (not cross-country DiD).",
        "",
        f"**Track:** {track}",
        "",
        "Key files: `dist_std_delta_composite.png`, `components_grid.png`, `spaghetti_sample.png`.",
        "Tables: `shift_per_user_{cohort}_{slug}.csv` in `results/tables/.../user_week/`.",
    ]
    _write_readme(fig_dir / "README.md", lines)


def write_event_study_readme(fig_dir: Path, cohort: str) -> None:
    """Function summary: README for author FE rel_week event-study plots."""
    lines = [
        f"# User-week event study — {cohort}",
        "",
        "**Estimand:** `y ~ C(rel_week) + author FE` on author×ISO-week panel; week −1 omitted.",
        "Clustered SE at author. Italy national shock only.",
        "",
        "Population analogue to dynamic reading of `author_it_ban` DiD; does not replace `cross_country_all`.",
    ]
    _write_readme(fig_dir / "README.md", lines)


def write_overview_readme(fig_dir: Path, cohort: str) -> None:
    """Function summary: README for headline median-shift overview."""
    lines = [
        f"# User-week overview — {cohort}",
        "",
        "Median pooled within-person deltas across users for headline lexical and semantic outcomes.",
        "Descriptive summary; see `regression_summary_{cohort}.csv` for panel post coefficients.",
    ]
    _write_readme(fig_dir / "README.md", lines)


def write_pole_readme(fig_dir: Path, cohort: str) -> None:
    """Function summary: README linking pole-margin shifts to forum pole_share DiD."""
    lines = [
        f"# Pole decomposition — {cohort}",
        "",
        "Within-user shifts in left/right/center rates and `pole_share`.",
        "Bridges to forum-day `pole_share` outcomes in `did_event_study.py`.",
    ]
    _write_readme(fig_dir / "README.md", lines)


def write_all_user_week_readmes(figures_root: Path, cohorts: Iterable[str] = ("strict", "loose")) -> None:
    """Function summary: write README stubs for standard user_week figure subtrees.

    Parameters:
    - figures_root: e.g. results/figures/italy_polarization/user_week.
    - cohorts: cohort labels to scan.
    """
    for cohort in cohorts:
        for slug in COMPOSITE_SLUGS:
            write_composite_readme(figures_root / cohort / slug, cohort, slug)
        write_event_study_readme(figures_root / cohort / "event_study", cohort)
        write_overview_readme(figures_root / cohort / "overview", cohort)
        write_pole_readme(figures_root / cohort / "pole_decomposition", cohort)
