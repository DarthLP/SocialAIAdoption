"""Tests for semaxis_extend pass preserving legacy semantic columns."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.config_utils import load_config, load_semantic_axis_config
from src.embeddings import (
    SEMAXIS_EXTENDED_KEYS,
    SEMAXIS_LEGACY_KEYS,
    build_axis,
    clear_embedding_caches,
    load_seed_poles,
)


def _mock_kv() -> KeyedVectors:
    """Build 8-d space with separable directions for legacy and extended axes."""
    kv = KeyedVectors(vector_size=8)
    rng = np.random.default_rng(1)
    words = [
        "liberismo",
        "redistribuzione",
        "privatizzazioni",
        "disuguaglianze",
        "tradizione",
        "antifascista",
        "porti",
        "ius",
        "poteri",
        "magistratura",
        "hello",
        "world",
    ]
    for i, word in enumerate(words):
        v = rng.standard_normal(8)
        if word == "liberismo":
            v[0] = 2.0
        if word == "redistribuzione":
            v[0] = -2.0
        if word == "privatizzazioni":
            v[3] = 2.0
        if word == "disuguaglianze":
            v[3] = -2.0
        kv.add_vector(word, v)
    return kv


def test_semaxis_extend_preserves_legacy_columns(tmp_path: Path) -> None:
    """Extend pass adds extended columns without changing legacy sem_axis values."""
    root = Path(__file__).resolve().parent.parent
    shard = tmp_path / "test_shard.parquet"
    legacy = {
        "sem_axis_ideology": [0.42, -0.11],
        "sem_axis_emotion": [0.05, 0.07],
        "sem_axis_aggression": [0.01, -0.02],
        "sem_axis_coverage": [0.9, 0.8],
        "has_sem_axis": [1.0, 1.0],
    }
    df = pd.DataFrame(
        {
            "id": ["c1", "c2"],
            "body": ["privatizzazioni hello", "disuguaglianze world"],
            "primary_lexicon": ["it", "it"],
            "n_words": [2, 2],
            **legacy,
        }
    )
    df.to_parquet(shard, index=False)

    config = load_config(root / "config/italy_polarization_setup.yaml")
    sem_cfg = load_semantic_axis_config(config)
    sem_cfg["write_vector_cache"] = False
    clear_embedding_caches()

    import scripts.features._enriched_shard_runner as runner

    kv = _mock_kv()

    def _fake_load(lang: str, project_root: Path, axes_cfg):  # noqa: ANN001
        return kv

    poles = load_seed_poles("it", root, sem_cfg)
    ext_axes = {
        "economic": build_axis(poles["economic_pos"], poles["economic_neg"], kv),
        "cultural": build_axis(poles["cultural_pos"], poles["cultural_neg"], kv),
        "nationalism": build_axis(poles["nationalism_pos"], poles["nationalism_neg"], kv),
        "anti_establishment": build_axis(
            poles["anti_establishment_pos"], poles["anti_establishment_neg"], kv
        ),
    }

    with patch("src.embeddings.load_vectors", _fake_load):
        with patch("src.embeddings.get_axes_for_language") as mock_axes:
            mock_axes.return_value = ext_axes
            n = runner._process_semaxis_extend_shard(
                shard, "TestSub", "it", sem_cfg, root, tmp_path
            )
    assert n == 2
    out = pd.read_parquet(shard)
    for col in SEMAXIS_LEGACY_KEYS:
        np.testing.assert_allclose(
            out[col].astype(float).values,
            df[col].astype(float).values,
            err_msg=col,
        )
    for col in SEMAXIS_EXTENDED_KEYS:
        assert col in out.columns
