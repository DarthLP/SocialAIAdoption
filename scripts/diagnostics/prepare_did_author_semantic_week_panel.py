"""
Script summary:
Build author×ISO-week semantic-axis panel for cross-country TWFE DiD on within-author weekly means.

Functionality:
- Reads user_week_panel.parquet (author, iso_week_start, sem_axis_*_mean, share_scored).
- Joins assigned_primary_lexicon from wordfish_authors_assignment.csv (v2 preferred).
- Adds DiD calendar fields (rel_day, post, entity_id, treat) for did_event_study author strategies.

How to apply/run:
  .venv/bin/python scripts/diagnostics/prepare_did_author_semantic_week_panel.py \\
    --config config/italy_polarization_setup.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

KEEP_PANEL_COLS = (
    "author",
    "iso_week_start",
    "sem_axis_ideology_mean",
    "sem_axis_emotion_mean",
    "sem_axis_aggression_mean",
    "sem_axis_economic_mean",
    "sem_axis_cultural_mean",
    "sem_axis_nationalism_mean",
    "sem_axis_anti_establishment_mean",
    "sem_axis_coverage_mean",
    "share_scored",
    "n_words",
    "n_comments",
    "top_subreddit",
    "top_topic",
)


def _setup_project_root() -> Path:
    """Function summary: resolve repo root via scripts/_bootstrap.py."""
    caller = Path(__file__).resolve()
    for parent in caller.parents:
        if parent.name == "scripts" and (parent / "_bootstrap.py").is_file():
            spec = importlib.util.spec_from_file_location(
                "_socialai_bootstrap_mod", parent / "_bootstrap.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load scripts/_bootstrap.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.setup_project_path(caller)
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


PROJECT_ROOT = _setup_project_root()

from scripts.diagnostics.descriptives_util import event_dates_from_config  # noqa: E402
from src.config_utils import load_config, tables_subdir  # noqa: E402
from src.did.paths import did_panels_dir  # noqa: E402
from src.did.specs import rel_day_from_date  # noqa: E402


def wordfish_author_cohort(config: Dict[str, Any]) -> pd.Series:
    """Function summary: author ids from Wordfish extremity panel (v2 preferred).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Series of author string ids (possibly empty).
    """
    for sub in ("wordfish_authors_v2", "wordfish_authors"):
        path = tables_subdir(config, sub) / "wordfish_authors_extremity_panel.csv"
        if path.is_file():
            wf = pd.read_csv(path, usecols=["author"])
            return wf["author"].astype(str).drop_duplicates()
    return pd.Series(dtype=str)


def parse_args() -> argparse.Namespace:
    """Function summary: CLI for author semantic week panel build."""
    parser = argparse.ArgumentParser(description="Prepare author×week semantic DiD panel.")
    parser.add_argument("--config", type=str, default="config/italy_polarization_setup.yaml")
    parser.add_argument(
        "--restrict-to-wordfish-authors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only authors in wordfish_authors_extremity_panel (default: True, ~2.5k authors for feasible TWFE).",
    )
    return parser.parse_args()


def resolve_assignment_path(config: Dict[str, Any]) -> Optional[Path]:
    """Function summary: locate wordfish author assignment CSV (v2 then v1).

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Path if found, else None.
    """
    for sub in ("wordfish_authors_v2", "wordfish_authors"):
        path = tables_subdir(config, sub) / "wordfish_authors_assignment.csv"
        if path.is_file():
            return path
    return None


def load_assignment(config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: load author→primary_lexicon assignment table.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - DataFrame with author, assigned_primary_lexicon (it/en/de only).
    """
    path = resolve_assignment_path(config)
    if path is None:
        raise FileNotFoundError(
            "Missing wordfish_authors_assignment.csv under wordfish_authors_v2/ or wordfish_authors/; "
            "run prepare_wordfish_authors_v2.py first."
        )
    df = pd.read_csv(path)
    if "author" not in df.columns or "assigned_primary_lexicon" not in df.columns:
        raise ValueError(f"Assignment CSV missing required columns: {path}")
    out = df[["author", "assigned_primary_lexicon"]].copy()
    out["author"] = out["author"].astype(str)
    out["assigned_primary_lexicon"] = out["assigned_primary_lexicon"].astype(str).str.lower()
    out = out[out["assigned_primary_lexicon"].isin({"it", "en", "de"})].drop_duplicates(
        subset=["author"], keep="last"
    )
    return out


def annotate_author_semantic_week_panel(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Function summary: add DiD entity/time/treatment columns for author×week panel.

    Parameters:
    - df: merged author-week rows with primary_lexicon.
    - config: loaded study YAML.

    Returns:
    - Panel restricted to event_window and annotated for TWFE.
    """
    start, end_excl, launch, _ = event_dates_from_config(config)
    out = df.copy()
    out["date_utc"] = out["iso_week_start"].astype(str)
    out["rel_day"] = rel_day_from_date(out["iso_week_start"], launch)
    out["post"] = (out["iso_week_start"].astype(str) >= launch).astype(int)
    out["IT"] = (out["primary_lexicon"].astype(str) == "it").astype(int)
    out["treat"] = out["IT"]
    out["entity_id"] = out["author"].astype(str)
    out["time_id"] = out["iso_week_start"].astype(str)
    return out[(out["date_utc"] >= start) & (out["date_utc"] < end_excl)].reset_index(drop=True)


def build_author_semantic_week_panel(
    config: Dict[str, Any],
    *,
    restrict_to_wordfish_authors: bool = True,
) -> pd.DataFrame:
    """Function summary: assemble author×week semantic panel from user-week + assignment.

    Parameters:
    - config: loaded study YAML.
    - restrict_to_wordfish_authors: when True, inner-join Wordfish extremity authors (~2.5k) so cross-country TWFE is feasible.

    Returns:
    - Annotated panel ready for did_event_study.
    """
    uw_path = Path(config["paths"]["tables_dir"]) / "user_week" / "user_week_panel.parquet"
    if not uw_path.is_file():
        raise FileNotFoundError(
            f"Missing {uw_path}; run scripts/user_week/prepare_user_week_style_panel.py first."
        )
    panel = pd.read_parquet(uw_path)
    if panel.empty:
        return panel
    keep = [c for c in KEEP_PANEL_COLS if c in panel.columns]
    panel = panel[keep].copy()
    panel["author"] = panel["author"].astype(str)

    assignment = load_assignment(config)
    merged = panel.merge(assignment, on="author", how="inner")
    merged = merged.rename(columns={"assigned_primary_lexicon": "primary_lexicon"})
    if restrict_to_wordfish_authors:
        cohort = wordfish_author_cohort(config)
        if not cohort.empty:
            keep = set(cohort.astype(str))
            merged = merged[merged["author"].astype(str).isin(keep)].copy()
    return annotate_author_semantic_week_panel(merged, config)


def author_semantic_week_panel_path(config: Dict[str, Any]) -> Path:
    """Function summary: output CSV path for author semantic week panel.

    Parameters:
    - config: loaded study YAML.

    Returns:
    - Path under did/panels/author/.
    """
    return did_panels_dir(config, "author") / "did_author_semantic_week_panel.csv"


def main() -> None:
    """Function summary: write did_author_semantic_week_panel.csv."""
    args = parse_args()
    config = load_config(args.config)
    panel = build_author_semantic_week_panel(
        config,
        restrict_to_wordfish_authors=bool(args.restrict_to_wordfish_authors),
    )
    out_path = author_semantic_week_panel_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)
    n_lex = panel["primary_lexicon"].value_counts().to_dict() if not panel.empty else {}
    print(
        f"[prepare_did_author_semantic_week_panel] wrote rows={len(panel)} path={out_path} "
        f"lexicon_counts={n_lex}",
        flush=True,
    )


if __name__ == "__main__":
    main()
