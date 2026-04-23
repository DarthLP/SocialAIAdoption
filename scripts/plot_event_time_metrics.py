"""
Script summary:
This script reads daily event-time metrics and creates the requested event-time
line plots for semicolon rate, comment length, complexity index, AI-likeness,
AI-typical word intensity (including delve), and toxicity score.

How to apply/run:
- Run after setup pipeline:
  `.venv/bin/python scripts/plot_event_time_metrics.py --config config/political_forums_setup.yaml`
- Figures are saved into the configured figures directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config


def parse_args() -> argparse.Namespace:
    """Function summary: parse CLI args and return plotting runtime options."""
    parser = argparse.ArgumentParser(description="Plot event-time metrics.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/political_forums_setup.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def plot_metric(df: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    """Function summary: generate and save one event-time line plot for a chosen metric."""
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="event_time_t", y=y_col, marker="o")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Event time (days from 2022-11-30)")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> None:
    """Function summary: load daily data and write all requested event-time figures."""
    args = parse_args()
    config = load_config(args.config)
    figures_dir = Path(config["paths"]["figures_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    daily_path = tables_dir / "event_time_daily_metrics.csv"
    df = pd.read_csv(daily_path)
    df = df.sort_values("event_time_t")

    plot_metric(
        df,
        "semicolon_rate_100w",
        "Event-time: Semicolon Rate (per 100 words)",
        figures_dir / "event_time_semicolon_rate.png",
    )
    plot_metric(
        df,
        "comment_length_words",
        "Event-time: Average Comment Length (words)",
        figures_dir / "event_time_comment_length.png",
    )
    plot_metric(
        df,
        "complexity_index",
        "Event-time: Complexity Index",
        figures_dir / "event_time_complexity_index.png",
    )
    plot_metric(
        df,
        "ai_likeness_index",
        "Event-time: AI-likeness Index",
        figures_dir / "event_time_ai_likeness.png",
    )
    plot_metric(
        df,
        "ai_word_rate_100w",
        "Event-time: AI-typical Word Rate (includes delve)",
        figures_dir / "event_time_ai_word_rate.png",
    )
    plot_metric(
        df,
        "toxicity_score",
        "Event-time: Toxicity Proxy (VADER compound)",
        figures_dir / "event_time_toxicity_score.png",
    )


if __name__ == "__main__":
    main()
