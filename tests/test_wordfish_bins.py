"""Tests for event-anchored Wordfish bin assignment."""

from __future__ import annotations

import unittest

from src.wordfish import bin_start_for_week, days_from_anchor, parse_anchor_date


class TestWordfishBins(unittest.TestCase):
    """Event-week bins relative to ban anchor t*."""

    def test_weekly_bin_boundary_at_anchor(self) -> None:
        """t* is left edge of [0,+7) block."""
        t_star = parse_anchor_date("2023-03-31")
        self.assertEqual(bin_start_for_week("2023-03-31", t_star), "2023-03-31")
        self.assertEqual(bin_start_for_week("2023-04-06", t_star), "2023-03-31")
        self.assertEqual(bin_start_for_week("2023-04-07", t_star), "2023-04-07")

    def test_weekly_pre_ban_block(self) -> None:
        """[-7,0) block starts seven days before t*."""
        t_star = parse_anchor_date("2023-03-31")
        self.assertEqual(bin_start_for_week("2023-03-30", t_star), "2023-03-24")
        self.assertEqual(days_from_anchor("2023-03-24", t_star), -7)


if __name__ == "__main__":
    unittest.main()
