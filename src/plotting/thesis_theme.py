"""
Script summary:
Centralized colors, axis labels, and ban-window guides for thesis figures.

Functionality:
- Exposes thesis palette (Italy red, control steel blue, control band fill).
- Standard x/y label strings for event-study and calendar time-series plots.
- shade_ban_window() draws identical onset/lift guides on every figure.

How to apply/run:
  from src.plotting.thesis_theme import shade_ban_window, xlabel_event_study, THESIS_ITALY
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# Thesis palette (reference: chatgpt_mention_rate pooled-range figure).
THESIS_ITALY = "#CC0000"
THESIS_CONTROL = "#34708F"
THESIS_CONTROL_BAND = "#B8CEDD"
# Marker edge/error-bar color for single-series coefficient plots (neutral
# dark gray; reserves Italy red / control blue for group identity).
THESIS_COEF_MARKER = "#333333"

BAN_ONSET_LINE = "#4D4D4D"
BAN_LIFT_LINE = "#B3B3B3"
BAN_SHADE_FILL = "#D9D9D9"
BAN_SHADE_ALPHA = 0.30
BAN_LINE_WIDTH = 1.2

DEFAULT_BAN_START = "2023-03-31"
DEFAULT_BAN_END = "2023-04-28"
BAN_LIFT_REL_DAY = 28

XLABEL_CALENDAR = "Date (UTC)"

THESIS_EVENT_STUDY_TITLES: Dict[str, str] = {
    "sem_axis_emotion": "Emotion–cognition axis",
    "sem_axis_emotion_events": "Emotion–cognition axis",
    "aggression_rate": "Lexical aggression rate",
    "sem_axis_aggression": "Semantic aggression",
    "sem_axis_ideology_var": "Semantic ideology variance",
    "style_index_llm": "AI-style index",
    "ai_style_rate": "AI-style rate",
}


def ylabel_italy_bin_coefficient() -> str:
    """Function summary: standardized y-axis for DiD event-study coefficient plots.

    Returns:
    - Exact string 'Italy × bin coefficient' with multiplication sign U+00D7.
    """
    return "Italy \u00d7 bin coefficient"


def xlabel_event_study(bin_days: int) -> str:
    """Function summary: standardized x-axis for binned event-study plots.

    Parameters:
    - bin_days: calendar width of each event-time bin (1 or 3).

    Returns:
    - Label like 'Days relative to ban onset (3-day bins)'.
    """
    n = int(bin_days)
    return f"Days relative to ban onset ({n}-day bins)"


def xlabel_event_study_days() -> str:
    """Function summary: x-axis for raw trajectory plots without bin parenthetical.

    Returns:
    - 'Days relative to ban onset'.
    """
    return "Days relative to ban onset"


def thesis_title_for_outcome(outcome_id: str, *, fallback: Optional[str] = None) -> str:
    """Function summary: neutral thesis figure title for a known outcome slug.

    Parameters:
    - outcome_id: outcome identifier (e.g. sem_axis_emotion).
    - fallback: optional default when outcome is not in THESIS_EVENT_STUDY_TITLES.

    Returns:
    - Short plain-English title or fallback / outcome_id.
    """
    if outcome_id in THESIS_EVENT_STUDY_TITLES:
        return THESIS_EVENT_STUDY_TITLES[outcome_id]
    if fallback is not None:
        return fallback
    return outcome_id


def event_study_ban_boundaries(
    *,
    bin_days: int = 1,
    x_scale: Literal["days", "period"] = "days",
) -> Tuple[float, float]:
    """Function summary: x-positions for ban onset and lift at bin boundaries.

    Parameters:
    - bin_days: width of each event-time bin in calendar days.
    - x_scale: 'days' when x-axis is calendar rel-days; 'period' when x is rel_period index.

    Returns:
    - Tuple (onset_x, lift_x) placed in gaps between bin markers, not on marker centers.
    """
    bd = max(1, int(bin_days))
    if x_scale == "period":
        onset_x = -0.5
        lift_period = BAN_LIFT_REL_DAY // bd
        lift_x = float(lift_period) + 0.5
        return onset_x, lift_x
    onset_x = -0.5
    lift_x = float(BAN_LIFT_REL_DAY) + 0.5
    return onset_x, lift_x


def shade_ban_window(
    ax: plt.Axes,
    *,
    mode: Literal["calendar", "event_study"] = "event_study",
    bin_days: int = 1,
    x_scale: Literal["days", "period"] = "days",
    ban_start: str = DEFAULT_BAN_START,
    ban_end: str = DEFAULT_BAN_END,
    zorder: int = 0,
) -> None:
    """Function summary: draw identical ban onset/lift guides and shaded ban window.

    Parameters:
    - ax: matplotlib axes.
    - mode: 'calendar' for Date (UTC) axes; 'event_study' for rel-day/rel-period axes.
    - bin_days: event-time bin width (1 or 3) for event_study boundary math.
    - x_scale: 'days' or 'period' for event_study x-axis units.
    - ban_start: ISO ban launch date for calendar mode.
    - ban_end: ISO ban lift date for calendar mode.
    - zorder: draw order (keep below data markers).

    Returns:
    - None; mutates ax in place.
    """
    if mode == "calendar":
        onset_ts = pd.Timestamp(ban_start)
        lift_ts = pd.Timestamp(ban_end)
        ax.axvspan(onset_ts, lift_ts, color=BAN_SHADE_FILL, alpha=BAN_SHADE_ALPHA, zorder=zorder)
        ax.axvline(
            onset_ts,
            color=BAN_ONSET_LINE,
            linestyle="--",
            linewidth=BAN_LINE_WIDTH,
            zorder=zorder + 1,
        )
        ax.axvline(
            lift_ts,
            color=BAN_LIFT_LINE,
            linestyle="--",
            linewidth=BAN_LINE_WIDTH,
            zorder=zorder + 1,
        )
        return

    onset_x, lift_x = event_study_ban_boundaries(bin_days=bin_days, x_scale=x_scale)
    ax.axvspan(onset_x, lift_x, color=BAN_SHADE_FILL, alpha=BAN_SHADE_ALPHA, zorder=zorder)
    ax.axvline(
        onset_x,
        color=BAN_ONSET_LINE,
        linestyle="--",
        linewidth=BAN_LINE_WIDTH,
        zorder=zorder + 1,
    )
    ax.axvline(
        lift_x,
        color=BAN_LIFT_LINE,
        linestyle="--",
        linewidth=BAN_LINE_WIDTH,
        zorder=zorder + 1,
    )


def ban_dates_from_config(config: Dict[str, Any]) -> Tuple[str, str]:
    """Function summary: read ban launch and lift ISO dates from study YAML.

    Parameters:
    - config: loaded project config dict.

    Returns:
    - Tuple (ban_start, ban_end) as YYYY-MM-DD strings.
    """
    ew = config.get("event_window") or {}
    launch = str(ew.get("launch_day_utc") or DEFAULT_BAN_START)
    lift = str(ew.get("lift_day_utc") or DEFAULT_BAN_END)
    return launch, lift
