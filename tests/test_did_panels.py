"""Tests for DiD panel paths and DDD entity indexing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.diagnostics.prepare_did_merged_panels import TOPIC_FAMILY_TO_COUNTRY_PANEL
from src.config_utils import load_config
from src.did.paths import resolve_panel_path
from src.did.panels import (
    _author_extremity_panel_candidates,
    _panel_file_suffix,
    resolve_author_wordfish_spec,
    slice_panel_for_ddd,
    wordfish_forum_v2_available,
)


def test_us_topic_family_maps() -> None:
    """Function summary: semantic DiD map uses us not us_political."""
    assert TOPIC_FAMILY_TO_COUNTRY_PANEL["us"] == "US_political"


def test_did_subreddit_panel_exists() -> None:
    """Function summary: prepared subreddit panel has IT and rel_day."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        legacy = root / "results/tables/italy_polarization/did/did_subreddit_panel_1d.csv"
        path = legacy if legacy.is_file() else None
    else:
        config = load_config(cfg_path)
        path = resolve_panel_path(config, "subreddit", "did_subreddit_panel_1d.csv")
    if path is None or not path.is_file():
        return
    df = pd.read_csv(path, nrows=50)
    assert "IT" in df.columns
    assert "rel_day" in df.columns
    assert "topic_family" in df.columns


def test_slice_panel_ddd_entity_subreddit() -> None:
    """Function summary: DDD slice panel uses subreddit entity with political variation."""
    sl = pd.DataFrame(
        {
            "subreddit": ["r1", "r1", "r2", "r2"],
            "universe_slice": [
                "in_political_tree",
                "out_political_tree",
                "in_political_tree",
                "out_political_tree",
            ],
            "date_utc": ["2023-03-30", "2023-03-30", "2023-03-31", "2023-03-31"],
            "IT": [1, 1, 1, 1],
        }
    )
    ddd = slice_panel_for_ddd(sl)
    assert (ddd["entity_id"] == ddd["subreddit"]).all()
    pol_by_ent = ddd.groupby("entity_id")["universe_slice"].nunique()
    assert pol_by_ent.max() >= 2


def test_resolve_author_wordfish_spec() -> None:
    """Function summary: author spec resolves from did block then wordfish_authors headline."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        return
    config = load_config(cfg_path)
    assert resolve_author_wordfish_spec(config) in ("week7", "week3")
    assert resolve_author_wordfish_spec(config, override="week3") == "week3"


def test_author_extremity_panel_candidates_week3() -> None:
    """Function summary: week3 spec produces balanced_week3 filenames first."""
    tab = Path("/tmp/wordfish_authors")
    cands = _author_extremity_panel_candidates(tab, "week3", lang="it")
    assert "balanced_week3" in cands[0]
    assert cands[0].endswith("balanced_week3_it.csv")


def test_quantity_panel_exbantopic_suffix() -> None:
    """Function summary: quantity panel variant uses _exbantopic filename suffix."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        return
    config = load_config(cfg_path)
    base = resolve_panel_path(config, "subreddit", "did_subreddit_quantity_panel_1d.csv")
    ex = resolve_panel_path(
        config, "subreddit", "did_subreddit_quantity_panel_1d_exbantopic.csv"
    )
    assert base != ex
    assert ex.name == "did_subreddit_quantity_panel_1d_exbantopic.csv"
    assert _panel_file_suffix("exbantopic") == "_exbantopic"
    assert _panel_file_suffix(None) == ""


def test_wordfish_v2_path_resolution() -> None:
    """Function summary: v2 availability check resolves config path."""
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config/italy_polarization_setup.yaml"
    if not cfg_path.is_file():
        return
    config = load_config(cfg_path)
    _ = wordfish_forum_v2_available(config)  # no raise
