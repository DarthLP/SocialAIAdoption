"""
Script summary:
Comment-level political universe definitions (tree, comment, frozen thread modes).

Functionality:
- Normalizes Reddit comment ids (bare id vs t1_/t3_ parent_id).
- Computes comparison booleans on a Mar–Apr concatenated frame per subreddit.
- Tree propagation: political seeds, descendant replies, optional one-up parent, orphan thread fallback.

How to apply/run:
- Imported by scripts/features/apply_political_universe.py and diagnostics scripts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.political_lexicon import political_rate_100w

UNIVERSE_MODE_COLUMNS: Dict[str, str] = {
    "comment": "in_political_universe_comment",
    "tree": "in_political_universe_tree",
    "thread_sum": "in_political_universe_thread_sum",
    "thread_rate": "in_political_universe_thread_rate",
    "embedding": "in_political_universe_embedding",
}

ALL_UNIVERSE_BOOL_COLUMNS: Tuple[str, ...] = (
    "in_political_universe_comment",
    "in_political_universe_tree",
    "in_political_universe_thread_sum",
    "in_political_universe_thread_rate",
    "in_political_universe_embedding",
    "comment_in_political_universe",
)


def _scalar_str(value: Any) -> str:
    """Function summary: coerce a scalar (incl. pandas NA) to a clean string.

    Parameters:
    - value: id, parent_id, or other cell value.

    Returns:
    - Stripped string, or empty if missing/NA.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if s.lower() in ("nan", "<na>", "none"):
        return ""
    return s


def comment_id_key(raw_id: Any) -> str:
    """Function summary: normalize a comment id to bare form for graph edges.

    Parameters:
    - raw_id: shard id or t1_-prefixed parent reference.

    Returns:
    - Bare comment id string (empty if invalid).
    """
    s = _scalar_str(raw_id)
    if not s:
        return ""
    if s.startswith("t1_"):
        return s[3:]
    return s


def parent_comment_key(parent_id: Any) -> Optional[str]:
    """Function summary: map parent_id to in-corpus comment key if reply-to-comment.

    Parameters:
    - parent_id: Reddit parent_id (t1_* comment or t3_* submission).

    Returns:
    - Bare parent comment id, or None if submission / missing / invalid.
    """
    s = _scalar_str(parent_id)
    if not s:
        return None
    if s.startswith("t1_"):
        return s[3:]
    return None


def comment_is_political(points: Any, min_points: int) -> bool:
    """Function summary: lexical comment-only political seed (mode comment).

    Parameters:
    - points: political_weighted_points value.
    - min_points: minimum graded points threshold.

    Returns:
    - True if points meet threshold.
    """
    try:
        return int(points) >= int(min_points)
    except (TypeError, ValueError):
        return False


def frozen_thread_flags(
    df: pd.DataFrame,
    screening: Dict[str, Any],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Function summary: full-window per-link_id thread sum and rate flags.

    Parameters:
    - df: concatenated subreddit comments with link_id, political_weighted_points, n_words.
    - screening: screening config (thread_political_min_points, thread_political_rate_threshold).

    Returns:
    - Tuple (thread_sum_flag, thread_rate_flag, thread_political_rate_100w) indexed like df.
    """
    min_points = int(screening.get("thread_political_min_points", 3))
    rate_threshold = float(screening.get("thread_political_rate_threshold", 0.45))
    if "link_id" not in df.columns:
        empty = pd.Series(False, index=df.index)
        return empty, empty.copy(), pd.Series(0.0, index=df.index)

    stats = (
        df.groupby("link_id", as_index=False)
        .agg(
            thread_political_weighted_points=("political_weighted_points", "sum"),
            thread_n_words=("n_words", "sum"),
        )
    )
    stats["thread_political_rate_100w"] = stats.apply(
        lambda r: political_rate_100w(
            int(r["thread_political_weighted_points"]),
            int(r["thread_n_words"]),
        ),
        axis=1,
    )
    stats["thread_sum_flag"] = stats["thread_political_weighted_points"] >= min_points
    stats["thread_rate_flag"] = stats["thread_political_rate_100w"] >= rate_threshold
    merged = df[["link_id"]].merge(
        stats[
            [
                "link_id",
                "thread_sum_flag",
                "thread_rate_flag",
                "thread_political_rate_100w",
            ]
        ],
        on="link_id",
        how="left",
    )
    return (
        merged["thread_sum_flag"].fillna(False).astype(bool),
        merged["thread_rate_flag"].fillna(False).astype(bool),
        merged["thread_political_rate_100w"].fillna(0.0),
    )


def _build_children_index(
    id_keys: List[str],
    parent_keys: List[Optional[str]],
) -> Dict[str, List[str]]:
    """Function summary: map parent comment key -> list of child comment keys."""
    children: Dict[str, List[str]] = defaultdict(list)
    for cid, pkey in zip(id_keys, parent_keys, strict=True):
        if pkey and cid:
            children[pkey].append(cid)
    return children


def _collect_descendants(
    seeds: Set[str],
    children: Dict[str, List[str]],
    max_depth: Optional[int],
) -> Set[str]:
    """Function summary: BFS from seeds following reply edges downward."""
    if not seeds:
        return set()
    if max_depth is not None and max_depth < 0:
        return set(seeds)
    out: Set[str] = set(seeds)
    queue: deque[Tuple[str, int]] = deque((s, 0) for s in seeds)
    while queue:
        node, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        for child in children.get(node, []):
            if child not in out:
                out.add(child)
                queue.append((child, depth + 1))
    return out


def propagate_tree_universe(
    df: pd.DataFrame,
    pu_cfg: Dict[str, Any],
    screening: Dict[str, Any],
    thread_sum_flag: pd.Series,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Function summary: tree political universe with orphan thread fallback.

    Parameters:
    - df: comment frame with id, parent_id, political_weighted_points.
    - pu_cfg: political_universe config.
    - screening: screening thresholds (unused except via thread_sum_flag).
    - thread_sum_flag: frozen full-window thread sum boolean per row.

    Returns:
    - Tuple (in_tree boolean series, diagnostic stats dict).
    """
    del screening
    min_points = int(pu_cfg.get("comment_political_min_points", 3))
    include_parent = bool(pu_cfg.get("tree_include_parent", True))
    max_depth = pu_cfg.get("tree_max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)

    n = len(df)
    id_keys = [comment_id_key(x) for x in df["id"]]
    parent_keys = [parent_comment_key(x) for x in df["parent_id"]]
    id_set = {k for k in id_keys if k}
    points = df["political_weighted_points"].fillna(0).astype(int)

    seeds: Set[str] = set()
    for cid, pt in zip(id_keys, points, strict=True):
        if cid and comment_is_political(pt, min_points):
            seeds.add(cid)

    children = _build_children_index(id_keys, parent_keys)
    in_tree_keys = _collect_descendants(seeds, children, max_depth)

    if include_parent and seeds:
        for cid, pkey in zip(id_keys, parent_keys, strict=True):
            if cid in seeds and pkey and pkey in id_set:
                in_tree_keys.add(pkey)

    orphan_mask = np.zeros(n, dtype=bool)
    missing_parent_mask = np.zeros(n, dtype=bool)
    for i, pkey in enumerate(parent_keys):
        raw_parent = _scalar_str(df["parent_id"].iloc[i])
        if raw_parent.startswith("t1_") and pkey and pkey not in id_set:
            missing_parent_mask[i] = True
        if raw_parent.startswith("t1_") and (pkey is None or pkey not in id_set):
            orphan_mask[i] = True

    in_tree = np.array([k in in_tree_keys for k in id_keys], dtype=bool)
    fallback_applied = 0
    if orphan_mask.any():
        ts = thread_sum_flag.to_numpy(dtype=bool)
        for i in np.where(orphan_mask)[0]:
            if not in_tree[i] and ts[i]:
                in_tree[i] = True
                fallback_applied += 1

    stats = {
        "n_comments": float(n),
        "n_seeds": float(len(seeds)),
        "n_in_tree_before_orphan_fallback": float(in_tree.sum() - fallback_applied),
        "n_orphans": float(orphan_mask.sum()),
        "n_missing_parent_t1": float(missing_parent_mask.sum()),
        "n_orphan_fallback_applied": float(fallback_applied),
        "orphan_share": float(orphan_mask.mean()) if n else 0.0,
        "missing_parent_share": float(missing_parent_mask.mean()) if n else 0.0,
        "orphan_fallback_share": float(fallback_applied / n) if n else 0.0,
    }
    return pd.Series(in_tree, index=df.index), stats


def apply_all_modes(
    df: pd.DataFrame,
    pu_cfg: Dict[str, Any],
    screening: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Function summary: compute all political-universe boolean columns on one frame.

    Parameters:
    - df: concatenated subreddit comments (Mar–Apr).
    - pu_cfg: political_universe config from load_political_universe_config.
    - screening: screening config from load_screening_config.

    Returns:
    - Tuple (df with bool columns added, merged diagnostic stats).
    """
    out = df.copy()
    min_points = int(pu_cfg.get("comment_political_min_points", 3))
    points = out["political_weighted_points"].fillna(0)

    out["in_political_universe_comment"] = points.apply(
        lambda p: comment_is_political(p, min_points)
    )

    thread_sum_flag, thread_rate_flag, _thread_rate = frozen_thread_flags(out, screening)
    out["in_political_universe_thread_sum"] = thread_sum_flag
    out["in_political_universe_thread_rate"] = thread_rate_flag

    tree_series, tree_stats = propagate_tree_universe(
        out, pu_cfg, screening, thread_sum_flag
    )
    out["in_political_universe_tree"] = tree_series
    out["in_political_universe_embedding"] = False

    mode = str(pu_cfg.get("mode", "tree")).strip().lower()
    col = UNIVERSE_MODE_COLUMNS.get(mode, "in_political_universe_tree")
    if col not in out.columns:
        col = "in_political_universe_tree"
    out["comment_in_political_universe"] = out[col].astype(bool)

    stats = dict(tree_stats)
    stats["political_universe_share"] = float(out["comment_in_political_universe"].mean())
    stats["mode"] = mode
    return out, stats
