"""Tests for Wordfish change/placebo helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.wordfish import (
    add_date_utc_column,
    add_placebo_flags,
    compute_change_outcomes,
)


class TestWordfishChange(unittest.TestCase):
    """Rolling change and placebo schema helpers."""

    def test_compute_change_outcomes_day_rolling(self) -> None:
        """Rolling change uses prior bins within window_days for day bins."""
        ext = pd.DataFrame(
            {
                "subreddit": ["a"] * 4,
                "primary_lexicon": ["it"] * 4,
                "time_bin": ["day"] * 4,
                "bin_start": ["2023-03-01", "2023-03-02", "2023-03-03", "2023-03-10"],
                "extremity": [1.0, 2.0, 4.0, 10.0],
            }
        )
        out = compute_change_outcomes(ext, "2023-03-31", window_days=7)
        by_date = dict(zip(out["bin_start"], out["change"]))
        self.assertTrue(pd.isna(by_date["2023-03-01"]))
        self.assertEqual(by_date["2023-03-02"], 1.0)
        self.assertEqual(by_date["2023-03-03"], 2.5)
        self.assertIn("change_z", out.columns)

    def test_add_date_utc_and_placebo_flags(self) -> None:
        """date_utc populated for day rows; placebo flags bracket fake launch."""
        df = pd.DataFrame(
            {
                "bin_start": ["2023-03-10", "2023-03-20", "2023-04-01"],
                "time_bin": ["day", "day", "week"],
            }
        )
        dated = add_date_utc_column(df)
        self.assertEqual(dated.loc[0, "date_utc"], "2023-03-10")
        self.assertEqual(dated.loc[2, "date_utc"], "")

        flagged = add_placebo_flags(dated, "2023-03-16", "2023-03-31")
        self.assertEqual(int(flagged.loc[0, "pre_placebo"]), 1)
        self.assertEqual(int(flagged.loc[1, "post_placebo"]), 1)
        self.assertEqual(int(flagged.loc[2, "post_placebo"]), 0)


if __name__ == "__main__":
    unittest.main()
