"""
Resolved paths for DiD panels and estimation tables under results/tables/.../did/.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

from src.config_utils import tables_subdir

PanelKind = Literal["author", "comment", "country", "semantic", "subreddit", "aggregated"]
TableKind = Literal["coefficients", "robustness", "event_study"]


def did_root(config: Dict[str, Any]) -> Path:
    """Function summary: study DiD tables root (results/tables/<study>/did/).

    Parameters:
    - config: loaded YAML.

    Returns:
    - Path to did/ directory.
    """
    return tables_subdir(config, "did")


def did_panels_dir(config: Dict[str, Any], kind: PanelKind) -> Path:
    """Function summary: panels/{country|semantic|subreddit}/ under did/.

    Parameters:
    - config: loaded YAML.
    - kind: panel subgroup name.

    Returns:
    - Path to panels subdirectory.
    """
    return did_root(config) / "panels" / kind


def did_estimates_dir(config: Dict[str, Any]) -> Path:
    """Function summary: estimates/ root for DiD outputs.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Path to estimates/ directory.
    """
    return did_root(config) / "estimates"


def did_summary_dir(config: Dict[str, Any]) -> Path:
    """Function summary: estimates/summary/ for master and sliced summaries.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Path to summary/ directory.
    """
    return did_estimates_dir(config) / "summary"


def did_summary_paths(config: Dict[str, Any]) -> Tuple[Path, Path]:
    """Function summary: master did_summary.csv and labeled variant paths.

    Parameters:
    - config: loaded YAML.

    Returns:
    - Tuple of (did_summary.csv, did_summary_labeled.csv) paths.
    """
    summary_dir = did_summary_dir(config)
    return summary_dir / "did_summary.csv", summary_dir / "did_summary_labeled.csv"


def did_family_estimates_dir(config: Dict[str, Any], family: str) -> Path:
    """Function summary: estimates/{family}/ for per-outcome tables.

    Parameters:
    - config: loaded YAML.
    - family: outcome family id (lexical, semantic_axis, wordfish_*, ...).

    Returns:
    - Path to family estimates directory.
    """
    return did_estimates_dir(config) / family


def did_outcome_table_path(
    config: Dict[str, Any],
    family: str,
    table_kind: TableKind,
    outcome_id: str,
) -> Path:
    """Function summary: nested path for coefficients, robustness, or event_study CSV.

    Parameters:
    - config: loaded YAML.
    - family: outcome family id.
    - table_kind: coefficients | robustness | event_study.
    - outcome_id: outcome slug.

    Returns:
    - Full path, e.g. estimates/lexical/coefficients/aggression_rate.csv.
    """
    return did_family_estimates_dir(config, family) / table_kind / f"{outcome_id}.csv"


def did_legacy_coefficient_path(config: Dict[str, Any], outcome_id: str) -> Path:
    """Function summary: flat legacy did_coefficients_{outcome_id}.csv at did/ root.

    Parameters:
    - config: loaded YAML.
    - outcome_id: outcome slug.

    Returns:
    - Path for grep-compatible legacy coefficient export.
    """
    return did_root(config) / f"did_coefficients_{outcome_id}.csv"


def did_headline_event_study_table_path(config: Dict[str, Any], outcome_id: str) -> Path:
    """Function summary: flat event_study_{outcome_id}.csv for headline multi-strategy runs.

    Parameters:
    - config: loaded YAML.
    - outcome_id: outcome slug.

    Returns:
    - Path under did/ root.
    """
    return did_root(config) / f"event_study_{outcome_id}.csv"


def did_event_study_path(config: Dict[str, Any], family: str, outcome_id: str) -> Path:
    """Function summary: event-study coefficient table for one outcome.

    Parameters:
    - config: loaded YAML.
    - family: outcome family id.
    - outcome_id: outcome slug.

    Returns:
    - Path to event_study/{outcome_id}.csv.
    """
    return did_outcome_table_path(config, family, "event_study", outcome_id)


def did_aggregated_event_study_path(
    config: Dict[str, Any],
    family: str,
    panel_level: str,
    bundle: str,
    bin_days: int,
    strategy_id: str,
    outcome_id: str,
) -> Path:
    """Function summary: nested event-study CSV for aggregated panel runs.

    Parameters:
    - config: loaded YAML.
    - family: outcome family id.
    - panel_level: topic_family | language | language_universe.
    - bundle: figure bundle slug (e.g. subreddit, hub_pooled, in_out_slice).
    - bin_days: 1 or 3.
    - strategy_id: identification strategy key.
    - outcome_id: outcome slug.

    Returns:
    - Path under estimates/{family}/event_study/{panel_level}/{bundle}/{bin}d/{strategy_id}/.
    """
    base = (
        did_family_estimates_dir(config, family)
        / "event_study"
        / panel_level
        / bundle
        / f"{bin_days}d"
    )
    return base / strategy_id / f"{outcome_id}.csv"


def aggregated_event_study_figure_path(
    fig_dir: Path,
    panel_level: str,
    bundle: str,
    bin_days: int,
    outcome_id: str,
) -> Path:
    """Function summary: event-study PNG for one outcome × panel level × bundle × bin.

    Parameters:
    - fig_dir: did figures root.
    - panel_level: aggregation level slug.
    - bundle: figure bundle slug (e.g. overlay_pooled, subreddit, hub_pooled).
    - bin_days: 1 or 3.
    - outcome_id: outcome slug.

    Returns:
    - Path to PNG under event_study/{panel_level}/{bundle}/{bin}d/.
    """
    return fig_dir / "event_study" / panel_level / bundle / f"{bin_days}d" / f"{outcome_id}.png"


def aggregated_tail_shift_figure_path(
    fig_dir: Path,
    panel_level: str,
    bundle: str,
    bin_days: int,
    *,
    suffix: str = "",
) -> Path:
    """Function summary: dual-tail ideology event-study PNG for one bundle × bin.

    Parameters:
    - fig_dir: did figures root.
    - panel_level: aggregation level slug.
    - bundle: figure bundle slug.
    - bin_days: 1 or 3.
    - suffix: optional filename stem suffix (e.g. in_tree).

    Returns:
    - Path under event_study/{panel_level}/{bundle}/{bin}d/.
    """
    stem = "sem_axis_ideology_tail_shift"
    if suffix:
        stem = f"{stem}_{suffix}"
    return fig_dir / "event_study" / panel_level / bundle / f"{bin_days}d" / f"{stem}.png"


def _legacy_panel_path(root: Path, filename: str) -> Path:
    """Function summary: flat did/ path used before nested layout."""
    return root / filename


def resolve_panel_path(config: Dict[str, Any], kind: PanelKind, filename: str) -> Path:
    """Function summary: resolve panel CSV with optional legacy fallback.

    Parameters:
    - config: loaded YAML.
    - kind: country | semantic | subreddit.
    - filename: basename under panels/{kind}/.

    Returns:
    - Existing path (new layout preferred, else legacy flat did/).
    """
    new_path = did_panels_dir(config, kind) / filename
    if new_path.is_file():
        return new_path
    legacy = _legacy_panel_path(did_root(config), filename)
    if legacy.is_file():
        warnings.warn(
            f"DiD panel {filename} found at legacy flat path {legacy}; "
            f"run migrate_did_table_layout.py or re-run prepare_did_* scripts.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return new_path
