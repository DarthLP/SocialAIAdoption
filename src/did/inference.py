"""
Wild cluster bootstrap and permutation inference for thin Italian cells.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.did.estimate import estimate_twfe
from src.did.specs import StrategySpec, filter_strategy_sample


def wild_cluster_bootstrap_p(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    n_draws: int = 199,
    seed: int = 42,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> float:
    """Function summary: Rademacher wild cluster bootstrap p-value for treat×post coef.

    Parameters:
    - panel: analysis panel.
    - strategy: strategy filter.
    - y_col: outcome column.
    - n_draws: bootstrap replications.
    - seed: RNG seed.

    Returns:
    - Two-sided bootstrap p-value (NaN if estimation fails).
    """
    sample = filter_strategy_sample(panel, strategy)
    if sample.empty or sample["treat"].nunique() < 2:
        return float("nan")
    base = estimate_twfe(sample, y_col, entity_col, time_col)
    b0 = base.get("beta", np.nan)
    if not np.isfinite(b0):
        return float("nan")
    rng = np.random.default_rng(seed)
    entities = sample[entity_col].astype(str).unique()
    betas = []
    for _ in range(n_draws):
        weights = rng.choice([-1.0, 1.0], size=len(entities))
        wmap = dict(zip(entities, weights))
        boot = sample.copy()
        boot["_w"] = boot[entity_col].astype(str).map(wmap)
        boot[y_col] = boot[y_col].astype(float) * boot["_w"]
        r = estimate_twfe(boot, y_col, entity_col, time_col)
        if np.isfinite(r.get("beta", np.nan)):
            betas.append(r["beta"])
    if len(betas) < 20:
        return float("nan")
    betas_arr = np.asarray(betas)
    p = float(np.mean(np.abs(betas_arr) >= abs(b0)))
    return min(1.0, max(0.0, p))


def permutation_test_p(
    panel: pd.DataFrame,
    strategy: StrategySpec,
    y_col: str,
    n_perm: int = 199,
    seed: int = 42,
    entity_col: str = "entity_id",
    time_col: str = "time_id",
) -> float:
    """Function summary: randomize treat labels within topic_family strata.

    Returns:
    - Two-sided permutation p-value.
    """
    sample = filter_strategy_sample(panel, strategy)
    if sample.empty or "topic_family" not in sample.columns:
        return float("nan")
    base = estimate_twfe(sample, y_col, entity_col, time_col)
    b0 = base.get("beta", np.nan)
    if not np.isfinite(b0):
        return float("nan")
    rng = np.random.default_rng(seed)
    betas = []
    entities = sample[[entity_col, "topic_family"]].drop_duplicates()
    from src.did.specs import ITALY_FAMILIES

    entities["treat"] = entities["topic_family"].astype(str).isin(ITALY_FAMILIES).astype(int)
    for _ in range(n_perm):
        perm = sample.copy()
        shuffled = []
        for fam, grp in entities.groupby("topic_family"):
            g = grp.copy()
            if len(g) > 1:
                g["treat"] = rng.permutation(g["treat"].values)
            shuffled.append(g)
        assign = pd.concat(shuffled, ignore_index=True)[[entity_col, "treat"]]
        perm = perm.drop(columns=["treat"], errors="ignore").merge(assign, on=entity_col, how="left")
        r = estimate_twfe(perm, y_col, entity_col, time_col)
        if np.isfinite(r.get("beta", np.nan)):
            betas.append(r["beta"])
    if len(betas) < 20:
        return float("nan")
    return float(np.mean(np.abs(np.asarray(betas)) >= abs(b0)))
