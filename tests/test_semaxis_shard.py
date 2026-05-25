"""Integration test for semaxis shard processor (mock fastText, real parquet if present)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from src.config_utils import load_config, load_semantic_axis_config
from src.embeddings import SEMAXIS_SCORE_KEYS, clear_embedding_caches


def _mock_kv() -> KeyedVectors:
    """Build a tiny 8-d keyed space with separable pole directions."""
    kv = KeyedVectors(vector_size=8)
    rng = np.random.default_rng(0)
    for i, word in enumerate(
        [
            "liberismo",
            "redistribuzione",
            "amore",
            "logica",
            "idiota",
            "grazie",
            "hello",
            "world",
            "test",
            "word",
        ]
    ):
        v = rng.standard_normal(8)
        if word in ("liberismo", "conservative"):
            v[0] = 2.0
        if word in ("redistribuzione", "progressive"):
            v[0] = -2.0
        if word in ("amore", "love"):
            v[1] = 2.0
        if word in ("logica", "logic"):
            v[1] = -2.0
        if word in ("idiota",):
            v[2] = 2.0
        if word in ("grazie",):
            v[2] = -2.0
        kv.add_vector(word, v)
    return kv


def test_semaxis_shard_idempotent(tmp_path: Path) -> None:
    """Re-running semaxis pass on a copied shard leaves scores unchanged."""
    root = Path(__file__).resolve().parent.parent
    shards = list(
        (root / "data/interim/italy_polarization/cleaned_monthly_chunks").rglob("*.parquet")
    )
    if not shards:
        return
    source = shards[0]
    sub = source.parent.name
    dest_dir = tmp_path / "cleaned_monthly_chunks" / sub
    dest_dir.mkdir(parents=True)
    shard = dest_dir / source.name
    shutil.copy2(source, shard)

    config = load_config(root / "config/italy_polarization_setup.yaml")
    sem_cfg = load_semantic_axis_config(config)
    sem_cfg["write_vector_cache"] = True
    interim = tmp_path
    clear_embedding_caches()

    import scripts.features._enriched_shard_runner as runner

    kv = _mock_kv()

    def _fake_load(lang: str, project_root: Path, axes_cfg):  # noqa: ANN001
        return kv

    with patch("src.embeddings.load_vectors", _fake_load):
        with patch("src.embeddings.get_axes_for_language") as mock_axes:
            from src.embeddings import build_axis, load_seed_poles

            poles = load_seed_poles("it", root, sem_cfg)
            mock_axes.return_value = {
                "ideology": build_axis(poles["ideology_pos"], poles["ideology_neg"], kv),
                "emotion": build_axis(poles["emotion_pos"], poles["emotion_neg"], kv),
                "aggression": build_axis(poles["aggression_pos"], poles["aggression_neg"], kv),
            }
            n1 = runner._process_semaxis_shard(
                shard, sub, "it", sem_cfg, root, interim
            )
            assert n1 > 0
            df1 = pd.read_parquet(shard)
            for col in SEMAXIS_SCORE_KEYS:
                assert col in df1.columns
            n2 = runner._process_semaxis_shard(
                shard, sub, "it", sem_cfg, root, interim
            )
            df2 = pd.read_parquet(shard)
            for col in SEMAXIS_SCORE_KEYS:
                np.testing.assert_allclose(
                    df1[col].astype(float).values,
                    df2[col].astype(float).values,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=col,
                )
            assert n2 == n1
