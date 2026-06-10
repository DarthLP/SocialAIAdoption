"""Unit tests for ban-topic comment flag regex."""

from __future__ import annotations

import pandas as pd

from src.ban_topic import (
    BAN_TOPIC_COLUMN,
    is_ban_topic_series,
    is_ban_topic_text,
)


def test_flags_chatgpt_and_garante():
    """Ban-discussion vocabulary should match."""
    assert is_ban_topic_text("OpenAI ha bloccato ChatGPT in Italia")
    assert is_ban_topic_text("Il Garante privacy ha ordinato il blocco")
    assert is_ban_topic_text("dati personali e GDPR")
    assert is_ban_topic_text("everyone is using a VPN now")


def test_does_not_flag_bare_ai_italian_preposition():
    """Bare 'ai' must not match (Italian preposition)."""
    assert not is_ban_topic_text("vado ai mercati")
    assert not is_ban_topic_text("sono ai confini")


def test_series_vectorized():
    """is_ban_topic_series returns aligned boolean column."""
    bodies = ["normale discussione", "ban on ChatGPT", ""]
    flags = is_ban_topic_series(bodies)
    assert flags.name == BAN_TOPIC_COLUMN
    assert list(flags) == [False, True, False]


def test_series_from_dataframe_column():
    """Series input works for shard-style frames."""
    df = pd.DataFrame({"body": ["privacy policy ok", "OpenAI ban"]})
    flags = is_ban_topic_series(df["body"])
    assert flags.iloc[1] is True or flags.iloc[1] == True  # noqa: E712
