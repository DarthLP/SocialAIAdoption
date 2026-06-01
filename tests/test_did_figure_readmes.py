"""Tests for DiD figure folder README generation."""

from __future__ import annotations

from pathlib import Path

from src.did.figure_readmes import (
    write_all_family_readmes,
    write_coefplots_readme,
    write_overview_readme,
)
from src.did.specs import strategy_label


def test_write_overview_readme(tmp_path: Path) -> None:
    """Function summary: overview README includes full strategy glossary."""
    overview = tmp_path / "overview"
    write_overview_readme(overview)
    text = (overview / "README.md").read_text(encoding="utf-8")
    assert "significance_heatmap.png" in text
    assert strategy_label("cross_country_all") in text
    assert "post_phase_net_ideology.png" in text


def test_write_coefplots_readme(tmp_path: Path) -> None:
    """Function summary: family README lists outcome PNGs."""
    plot_dir = tmp_path / "lexical" / "coefplots_headline"
    write_coefplots_readme(plot_dir, "lexical", "coefplots_headline", ["net_ideology"])
    text = (plot_dir / "README.md").read_text(encoding="utf-8")
    assert "net_ideology.png" in text
    assert "net_ideology" in text


def test_write_all_family_readmes(tmp_path: Path) -> None:
    """Function summary: index README is written at did figures root."""
    write_all_family_readmes(
        tmp_path,
        ["lexical"],
        {"lexical": ["net_ideology"]},
        full_coefplots=False,
    )
    assert (tmp_path / "README.md").is_file()
    assert (tmp_path / "lexical" / "coefplots_headline" / "README.md").is_file()
