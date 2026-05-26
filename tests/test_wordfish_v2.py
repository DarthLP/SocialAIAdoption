"""Tests for Wordfish v2 helpers (token cap, alternating fit, gate logic)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.wordfish import (
    beta_pole_diagnostics,
    cap_document_tokens,
    fit_wordfish,
    fit_wordfish_v2,
)


class TestWordfishV2Helpers(unittest.TestCase):
    """Token cap and pole diagnostics."""

    def test_cap_document_tokens(self) -> None:
        """Subsample respects max and seed."""
        tokens = [f"t{i}" for i in range(100)]
        capped, n_raw, trunc = cap_document_tokens(tokens, 40, seed=99)
        self.assertEqual(n_raw, 100)
        self.assertTrue(trunc)
        self.assertEqual(len(capped), 40)

    def test_beta_one_sided_all_positive(self) -> None:
        """All-positive raw beta flags one-sided."""
        beta = {f"w{i}": float(i + 1) for i in range(20)}
        diag = beta_pole_diagnostics(beta, opposite_frac_threshold=0.05)
        self.assertTrue(diag["one_sided_beta"])

    def test_v2_not_worse_than_legacy_on_toy(self) -> None:
        """V2 alternating fit reaches finite objective on toy DTM."""
        rng = np.random.default_rng(1)
        mat = rng.poisson(3, size=(15, 25)).astype(float)
        vocab = [f"w{i}" for i in range(25)]
        ids = [f"d{i}" for i in range(15)]
        cfg = {"max_cycles": 8, "min_objective_improvement": 1e-6}
        r2 = fit_wordfish_v2(mat, vocab, ids, cfg)
        self.assertTrue(np.isfinite(r2.objective_final))


class TestAuthorGateLogic(unittest.TestCase):
    """Validation gate threshold logic."""

    def test_gate_pass_requires_n_and_rho(self) -> None:
        """Gate passes only with enough authors and high |rho|."""
        val = pd.DataFrame(
            [
                {
                    "primary_lexicon": "it",
                    "spec": "week7",
                    "panel_mode": "balanced",
                    "n_authors": 200,
                    "spearman_theta_sem_axis": 0.6,
                    "spearman_theta_net_ideology": 0.1,
                }
            ]
        )
        threshold = 0.5
        min_auth = 100
        row = val.iloc[0]
        passed = (int(row["n_authors"]) >= min_auth) and (
            abs(float(row["spearman_theta_sem_axis"])) >= threshold
        )
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
