"""
Smoke testy pro audit_pipeline.

Cíl: ověřit, že refactored kód NEROZBIJE výsledky oproti původním notebookům.
Spuštění:  pytest tests/

Nejde o exhaustivní coverage — pro jednorázovou analýzu stačí, že hlavní
funkce na malém synthetickém DF dělají to, co mají.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit_pipeline import (
    AnalysisConfig,
    REELS_REQUIRED_COLS,
    TIKTOK_REQUIRED_COLS,
    deduplicate_videos,
    enrich_reels,
    enrich_tiktok,
    file_sha256,
    user_engagement_metrics,
    age_discrepancy_metrics,
    validate_schema,
)
from audit_pipeline.metrics import top_words, top_hashtags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> AnalysisConfig:
    return AnalysisConfig(
        collection_date=datetime(2026, 3, 16),
        declared_birth_year=2008,
        real_birth_year=2012,
    )


@pytest.fixture
def reels_df() -> pd.DataFrame:
    """Mini Reels dataset — 5 řádků, 1 topic match."""
    return pd.DataFrame({
        "video_id": ["v1", "v2", "v3", "v4", "v5"],
        "topic": ["gambling, adult"] * 5,
        "predicted_topic": ["random", "random", "gambling", "random", np.nan],
        "video_action_watch": [True, False, True, True, False],
        "video_action_like": [False, False, True, False, False],
        "video_action_bookmark": [False, False, False, False, False],
        "video_time_duration": [12.0, 30.5, 8.0, 45.0, 22.0],
        "video_author": ["a", "b", "c", "d", "e"],
        "video_description": [
            "casino #win night out",
            "spent the day cooking",
            "big bet #gambling #poker",
            "morning coffee routine",
            None,                           # NaN-safe?
        ],
        "date_of_birth": ["15.06.2008"] * 5,
    })


@pytest.fixture
def tiktok_df() -> pd.DataFrame:
    """Mini TikTok dataset — 6 řádků, různé kategorie."""
    return pd.DataFrame({
        "topic": ["donald_trump"] * 6,
        "stance": ["support"] * 6,
        "predicted_topic": ["donald_trump", "donald_trump", "recipes", "random", "donald_trump", np.nan],
        "predicted_topic_match": [True, True, False, False, True, False],
        "predicted_stance_match": [True, False, False, False, True, False],
        "video_action_watch": [True, True, False, True, True, False],
        "video_action_like": [True, False, False, False, True, False],
        "video_action_skip": [False, False, True, False, False, True],
        "video_action_bookmark": [False, False, False, False, True, False],
        "video_time_duration": [15.0, 22.0, 8.0, 30.0, 12.0, 5.0],
        "interaction_number": [1, 2, 3, 4, 5, 6],
        "user_email": ["test@example.com"] * 6,
        "run_id": ["run_001"] * 6,
        "date_of_birth": ["2008-06-15"] * 6,
    })


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_schema_passes_on_valid_df(reels_df):
    validate_schema(reels_df, REELS_REQUIRED_COLS, source="test")  # no raise


def test_validate_schema_raises_on_missing_column(reels_df):
    df = reels_df.drop(columns=["video_action_like"])
    with pytest.raises(ValueError, match="postrádá povinné sloupce"):
        validate_schema(df, REELS_REQUIRED_COLS, source="test")


# ---------------------------------------------------------------------------
# Reels enrichment
# ---------------------------------------------------------------------------

def test_enrich_reels_adds_expected_columns(reels_df, cfg):
    out = enrich_reels(reels_df, user_label="user1", cfg=cfg) if False else \
          enrich_reels(reels_df, user_label="user1", config=cfg)
    expected = {
        "user_label", "interaction_number", "interest_topics",
        "is_topic_match", "declared_age", "real_age",
    }
    assert expected.issubset(out.columns)


def test_enrich_reels_topic_match_excludes_nan_and_random(reels_df, cfg):
    out = enrich_reels(reels_df, user_label="user1", config=cfg)
    # row 0,1,3 = 'random' → False
    # row 2 = 'gambling' → True
    # row 4 = NaN → False (NaN-safe!)
    assert list(out["is_topic_match"]) == [False, False, True, False, False]


def test_enrich_reels_age_is_deterministic(reels_df, cfg):
    out1 = enrich_reels(reels_df, user_label="user1", config=cfg)
    out2 = enrich_reels(reels_df, user_label="user1", config=cfg)
    # Stejný vstup + stejný config = stejný výstup, vždy.
    # (Na rozdíl od původního pd.Timestamp.now() v tiktok notebooku.)
    assert out1["declared_age"].iloc[0] == out2["declared_age"].iloc[0]
    assert out1["real_age"].iloc[0] == out2["real_age"].iloc[0]
    # Deklarovaný (z 2008) musí být starší než reálný (z 2012).
    assert out1["declared_age"].iloc[0] > out1["real_age"].iloc[0]


def test_enrich_reels_handles_leap_year_dob(cfg):
    """29.2. v přestupném roce se nesmí pokoušet přesunout do ne-přestupného."""
    df = pd.DataFrame({
        "video_id": ["v1"],
        "topic": ["x"], "predicted_topic": ["random"],
        "video_action_watch": [True], "video_action_like": [False],
        "video_action_bookmark": [False], "video_time_duration": [10.0],
        "video_author": ["a"], "video_description": ["x"],
        "date_of_birth": ["29.02.2008"],   # přestupný rok
    })
    # Tohle by NEMĚLO spadnout (původní `dob.replace(year=2012)` by to ustál,
    # ale když by někdo dal real_birth_year=2009, spadl by.)
    cfg_nonleap = AnalysisConfig(
        collection_date=cfg.collection_date,
        declared_birth_year=2008,
        real_birth_year=2009,  # ne-přestupný
    )
    out = enrich_reels(df, "user1", cfg_nonleap)
    assert not np.isnan(out["real_age"].iloc[0])


# ---------------------------------------------------------------------------
# TikTok enrichment
# ---------------------------------------------------------------------------

def test_enrich_tiktok_categorizes_correctly(tiktok_df, cfg):
    out = enrich_tiktok(tiktok_df, cfg)
    expected_categories = [
        "Topic+Stance",      # row 0: topic=T, stance=T
        "Topic+Opačný",      # row 1: topic=T, stance=F
        "Recepty",           # row 2: predicted_topic = recipes
        "Náhodný",           # row 3: topic=F, není recipe
        "Topic+Stance",      # row 4: topic=T, stance=T
        "Náhodný",           # row 5: topic=F + NaN predicted_topic
    ]
    assert list(out["video_category"]) == expected_categories


def test_enrich_tiktok_no_apply_axis1(tiktok_df, cfg):
    """Kontrola: výsledek je stejný, ale řešeno přes np.select (rychlejší)."""
    out = enrich_tiktok(tiktok_df, cfg)
    # Sanity check vektorizovaných sloupců
    assert out["is_topic_stance"].dtype == bool
    assert out["is_topic_opposite"].dtype == bool
    assert out["is_recipes"].dtype == bool


# ---------------------------------------------------------------------------
# Deduplikace
# ---------------------------------------------------------------------------

def test_deduplicate_collapses_to_unique_videos(cfg):
    """Stejné video_id 4× → 1 řádek."""
    df = pd.DataFrame({
        "video_id": ["A", "A", "A", "A", "B"],
        "topic": ["x"] * 5,
        "predicted_topic": ["random"] * 5,
        "video_action_watch": [False, False, False, False, True],
        "video_action_like": [False, False, False, False, False],
        "video_action_bookmark": [False] * 5,
        "video_time_duration": [10.0] * 5,
        "video_author": ["a"] * 5, "video_description": ["x"] * 5,
        "date_of_birth": ["15.06.2008"] * 5,
    })
    enriched = enrich_reels(df, "user1", cfg)
    deduped = deduplicate_videos(enriched)
    assert len(deduped) == 2
    assert set(deduped["video_id"]) == {"A", "B"}


def test_deduplicate_action_uses_logical_or(cfg):
    """Pokud byl uživatel aktivní v ALESPOŇ jednom výskytu, video je interagované."""
    df = pd.DataFrame({
        "video_id": ["A", "A", "A"],
        "topic": ["x"] * 3,
        "predicted_topic": ["random"] * 3,
        "video_action_watch": [False, True, False],   # 1× watched
        "video_action_like": [False, False, False],   # never liked
        "video_action_bookmark": [False, False, False],
        "video_time_duration": [10.0] * 3,
        "video_author": ["a"] * 3, "video_description": ["x"] * 3,
        "date_of_birth": ["15.06.2008"] * 3,
    })
    enriched = enrich_reels(df, "user1", cfg)
    deduped = deduplicate_videos(enriched)
    assert deduped["video_action_watch"].iloc[0] == True
    assert deduped["video_action_like"].iloc[0] == False


def test_deduplicate_predicted_topic_prefers_non_random(cfg):
    """
    Když se video objeví 100× jako 'random' a 1× jako 'gambling',
    deduplikovaný řádek MUSÍ mít predicted_topic='gambling'.
    Jinak by is_topic_match=True bylo nekonzistentní s 'random'.
    """
    n = 100
    df = pd.DataFrame({
        "video_id": ["A"] * n + ["A"],   # 101 výskytů, jeden je 'gambling'
        "topic": ["x"] * (n + 1),
        "predicted_topic": ["random"] * n + ["gambling"],
        "video_action_watch": [False] * (n + 1),
        "video_action_like": [False] * (n + 1),
        "video_action_bookmark": [False] * (n + 1),
        "video_time_duration": [10.0] * (n + 1),
        "video_author": ["a"] * (n + 1), "video_description": ["x"] * (n + 1),
        "date_of_birth": ["15.06.2008"] * (n + 1),
    })
    enriched = enrich_reels(df, "user1", cfg)
    deduped = deduplicate_videos(enriched)

    assert len(deduped) == 1
    assert deduped["predicted_topic"].iloc[0] == "gambling"
    assert deduped["is_topic_match"].iloc[0] == True


def test_deduplicate_keeps_min_interaction_number(cfg):
    """Po dedupu se zachová pořadí PRVNÍHO výskytu."""
    df = pd.DataFrame({
        "video_id": ["A", "B", "A", "C", "A"],   # A se opakuje (1, 3, 5)
        "topic": ["x"] * 5, "predicted_topic": ["random"] * 5,
        "video_action_watch": [True] * 5, "video_action_like": [False] * 5,
        "video_action_bookmark": [False] * 5, "video_time_duration": [10.0] * 5,
        "video_author": ["a"] * 5, "video_description": ["x"] * 5,
        "date_of_birth": ["15.06.2008"] * 5,
    })
    enriched = enrich_reels(df, "user1", cfg)
    deduped = deduplicate_videos(enriched)

    a_row = deduped[deduped["video_id"] == "A"].iloc[0]
    assert a_row["interaction_number"] == 1   # ne 3, ne 5


def test_deduplicate_is_idempotent(cfg, reels_df):
    """Druhé volání dedup nesmí nic změnit."""
    enriched = enrich_reels(reels_df, "user1", cfg)
    once = deduplicate_videos(enriched)
    twice = deduplicate_videos(once)
    pd.testing.assert_frame_equal(once, twice)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_user_engagement_metrics_basic(reels_df, cfg):
    enriched = enrich_reels(reels_df, "user1", cfg)
    m = user_engagement_metrics(enriched, label="test")
    assert m["n_videos"] == 5
    assert m["topic_match_count"] == 1
    assert m["topic_match_pct"] == pytest.approx(20.0)
    assert m["watch_rate_pct"] == pytest.approx(60.0)
    assert m["like_rate_pct"] == pytest.approx(20.0)


def test_age_discrepancy_metrics(reels_df, cfg):
    enriched = enrich_reels(reels_df, "user1", cfg)
    m = age_discrepancy_metrics(enriched, label="test")
    assert m["available"] is True
    assert m["discrepancy_years"] == pytest.approx(4.0, abs=0.1)


def test_top_words_filters_stopwords_and_short_tokens(reels_df, cfg):
    out = enrich_reels(reels_df, "user1", cfg)
    words = top_words(out["video_description"], n=5, min_len=4)
    word_list = [w for w, _ in words]
    # 'the', 'a', 'and' atd. nesmí projít (stopwords)
    assert "the" not in word_list
    # příliš krátká slova taky ne
    assert all(len(w) >= 4 for w in word_list)


def test_top_hashtags_extraction(reels_df, cfg):
    out = enrich_reels(reels_df, "user1", cfg)
    tags = top_hashtags(out["video_description"], n=5)
    tag_list = [t for t, _ in tags]
    assert "win" in tag_list
    assert "gambling" in tag_list
    assert "poker" in tag_list


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def test_file_sha256_is_deterministic(tmp_path: Path):
    f = tmp_path / "test.csv"
    f.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    h1 = file_sha256(f)
    h2 = file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex length
