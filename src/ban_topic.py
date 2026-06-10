"""
Script summary:
Comment-level ban-topic flag for Italy ChatGPT-ban attention-shock diagnostics.

Functionality:
- Multilingual case-insensitive regex on comment bodies (ChatGPT, Garante, privacy,
  VPN, ban/block vocabulary across IT/EN/DE).
- Used by enriched-shard pass and descriptives exclusion (--exclude-ban-topic).

How to apply/run:
- Imported by scripts/features/compute_ban_topic_flag.py and diagnostics prep scripts.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

# Deliberately exclude bare ``ai`` (Italian preposition).
_BAN_TOPIC_PATTERN = re.compile(
    r"(?:"
    r"chatgpt|openai|gpt-?[34]|"
    r"garante|gdpr|privacy|dati personali|datenschutz|data protection|"
    r"intelligenza artificiale|artificial intelligence|k[uü]nstliche intelligenz|"
    r"\bvpn\b|"
    r"\bban\b|bannato|blocco|bloccato|divieto|vietato|sperre|verbot|censura"
    r")",
    re.IGNORECASE,
)

BAN_TOPIC_COLUMN = "is_ban_topic"


def ban_topic_regex() -> re.Pattern[str]:
    """Function summary: compiled multilingual ban-topic regex.

    Returns:
    - Compiled pattern for reuse across shards and tests.
    """
    return _BAN_TOPIC_PATTERN


def is_ban_topic_text(text: str | None) -> bool:
    """Function summary: True when comment body matches ban-topic regex.

    Parameters:
    - text: comment body string (None/empty -> False).

    Returns:
    - Boolean flag.
    """
    if not text or not str(text).strip():
        return False
    return bool(_BAN_TOPIC_PATTERN.search(str(text)))


def ensure_ban_topic_column(
    df: pd.DataFrame,
    *,
    log_prefix: str = "ban_topic",
) -> pd.DataFrame:
    """Function summary: resolve is_ban_topic without NaN-to-True bool coercion.

    Parameters:
    - df: comment frame that may have missing column, NaN values, or body text.
    - log_prefix: logger tag for warnings when body is unavailable.

    Returns:
    - Copy with boolean is_ban_topic (regex fallback for missing/NaN when body present).
    """
    out = df.copy()
    if BAN_TOPIC_COLUMN not in out.columns:
        if "body" in out.columns:
            out[BAN_TOPIC_COLUMN] = is_ban_topic_series(out["body"]).astype(bool)
        else:
            _warn_missing_ban_topic_rows(out, out.index, log_prefix=log_prefix)
            out[BAN_TOPIC_COLUMN] = False
        return out

    missing_mask = out[BAN_TOPIC_COLUMN].isna()
    if missing_mask.any():
        if "body" in out.columns:
            fallback = is_ban_topic_series(out.loc[missing_mask, "body"])
            out.loc[missing_mask, BAN_TOPIC_COLUMN] = fallback.to_numpy()
        else:
            _warn_missing_ban_topic_rows(out, out.index[missing_mask], log_prefix=log_prefix)
            out.loc[missing_mask, BAN_TOPIC_COLUMN] = False

    out[BAN_TOPIC_COLUMN] = out[BAN_TOPIC_COLUMN].astype(bool)
    return out


def _warn_missing_ban_topic_rows(
    df: pd.DataFrame,
    row_index: pd.Index,
    *,
    log_prefix: str,
) -> None:
    """Function summary: log subreddit row counts when is_ban_topic cannot be inferred."""
    n = len(row_index)
    if n == 0:
        return
    msg = f"[{log_prefix}] is_ban_topic missing for {n} rows (no body); filled False."
    if "subreddit" in df.columns:
        counts = df.loc[row_index, "subreddit"].astype(str).value_counts()
        sub_parts = ", ".join(f"{k}={int(v)}" for k, v in counts.items())
        msg = f"{msg} subreddits: {sub_parts}"
    print(msg, flush=True)


def is_ban_topic_series(bodies: Sequence[str] | pd.Series) -> pd.Series:
    """Function summary: vectorized ban-topic flags for many comment bodies.

    Parameters:
    - bodies: comment body strings aligned with shard rows.

    Returns:
    - Boolean pandas Series named is_ban_topic.
    """
    if isinstance(bodies, pd.Series):
        texts = bodies.fillna("").astype(str)
    else:
        texts = pd.Series(list(bodies)).fillna("").astype(str)
    flags = texts.str.contains(_BAN_TOPIC_PATTERN, regex=True, na=False)
    return flags.rename(BAN_TOPIC_COLUMN)


def flag_share_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Function summary: flagged share per aggregation group (forum-day diagnostics).

    Parameters:
    - df: comment frame with is_ban_topic and group columns.
    - group_cols: e.g. (subreddit, date_utc).

    Returns:
    - DataFrame with n_comments, n_flagged, share_flagged per group.
    """
    if df.empty or BAN_TOPIC_COLUMN not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work[BAN_TOPIC_COLUMN] = work[BAN_TOPIC_COLUMN].astype(bool)
    rows = []
    for key, grp in work.groupby(list(group_cols), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        n = len(grp)
        n_flagged = int(grp[BAN_TOPIC_COLUMN].sum())
        row = dict(zip(group_cols, key))
        row["n_comments"] = n
        row["n_flagged"] = n_flagged
        row["share_flagged"] = float(n_flagged) / float(n) if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
