"""
Výpočet všech klíčových metrik pro reporting.

Funkce vrací plain Python typy (dict, list, float) → snadno serializovatelné
do JSON pro thesis/SOČ.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Engagement metriky pro 1 uživatele/profil
# ---------------------------------------------------------------------------

def user_engagement_metrics(df: pd.DataFrame, label: str) -> dict:
    """
    Vrací slovník se všemi engagement metrikami pro daný profil.
    Používá se shodně pro Reels (User1/User2) i TikTok (per topic+stance).
    """
    n = len(df)
    if n == 0:
        return {"label": label, "n_videos": 0}

    metrics = {
        "label": label,
        "n_videos": int(n),
        "watch_rate_pct": float(df["video_action_watch"].mean() * 100),
        "like_rate_pct": float(df["video_action_like"].mean() * 100),
        "avg_video_duration_s": float(df["video_time_duration"].mean()),
        "median_video_duration_s": float(df["video_time_duration"].median()),
    }

    if "video_action_bookmark" in df.columns:
        metrics["bookmark_rate_pct"] = float(df["video_action_bookmark"].mean() * 100)
    if "video_action_skip" in df.columns:
        metrics["skip_rate_pct"] = float(df["video_action_skip"].mean() * 100)
    if "is_topic_match" in df.columns:
        n_match = int(df["is_topic_match"].sum())
        metrics["topic_match_count"] = n_match
        metrics["topic_match_pct"] = float(n_match / n * 100)
        metrics["matched_topics"] = sorted(
            df.loc[df["is_topic_match"], "predicted_topic"].dropna().unique().tolist()
        )
    return metrics


# ---------------------------------------------------------------------------
# Metriky věkové diskrepance (Reels-specific kontext)
# ---------------------------------------------------------------------------

def age_discrepancy_metrics(df: pd.DataFrame, label: str) -> dict:
    """Vrací declared vs real age. Předpokládá enriched DataFrame."""
    if "declared_age" not in df.columns:
        return {"label": label, "available": False}

    declared = float(df["declared_age"].iloc[0])
    real = float(df["real_age"].iloc[0])
    return {
        "label": label,
        "available": not (np.isnan(declared) or np.isnan(real)),
        "declared_age_years": declared,
        "real_age_years": real,
        "discrepancy_years": declared - real,
    }


# ---------------------------------------------------------------------------
# TikTok-specific: bubble index per topic/stance
# ---------------------------------------------------------------------------

def tiktok_topic_stance_breakdown(
    df: pd.DataFrame,
    target_topics: Iterable[str],
    target_stances: Iterable[str],
) -> list[dict]:
    """
    Vrací per (topic, stance) breakdown — vstup do summary tabulky pro thesis.
    """
    rows: list[dict] = []
    for topic in target_topics:
        for stance in target_stances:
            subset = df[(df["topic"] == topic) & (df["stance"] == stance)]
            if subset.empty:
                continue
            rows.append({
                "topic": topic,
                "stance": stance,
                "n_videos": int(len(subset)),
                "topic_match_pct": float(subset["predicted_topic_match"].mean() * 100),
                "stance_match_pct": float(subset["predicted_stance_match"].mean() * 100),
                "watch_rate_pct": float(subset["video_action_watch"].mean() * 100),
                "like_rate_pct": float(subset["video_action_like"].mean() * 100),
                "skip_rate_pct": float(subset["video_action_skip"].mean() * 100),
            })
    return rows


# ---------------------------------------------------------------------------
# Word / hashtag extrakce (pro Reels)
# ---------------------------------------------------------------------------

CZECH_EN_STOPWORDS: frozenset[str] = frozenset({
    "a", "je", "na", "se", "to", "v", "the", "and", "is", "in", "of",
    "for", "to", "reel", "by", "reels", "profilový", "obrázek",
    "double-tap", "play", "pause", "or", "with", "this", "that", "from",
    "you", "your", "we", "our", "my", "me", "it", "its", "was", "are",
    "be", "do", "an", "at", "as", "but", "not", "so", "he", "she", "they",
    "them", "his", "her", "has", "had", "have", "been", "if", "no", "up",
    "out", "all", "can", "get", "got", "just", "when", "one", "what", "i",
    "who", "will", "how", "about", "like", "which", "more", "than", "also",
})

_WORD_RE = re.compile(r"[a-záčďéěíňóřšťúůýž]+")
_HASHTAG_RE = re.compile(r"#([\w]+)")


def top_words(
    descriptions: pd.Series,
    n: int = 15,
    min_len: int = 3,
    stopwords: frozenset[str] = CZECH_EN_STOPWORDS,
) -> list[tuple[str, int]]:
    """Top-N slov v popisech videí (bez stopwords)."""
    counter: Counter[str] = Counter()
    for desc in descriptions.dropna():
        tokens = _WORD_RE.findall(str(desc).lower())
        counter.update(t for t in tokens if len(t) >= min_len and t not in stopwords)
    return counter.most_common(n)


def top_hashtags(descriptions: pd.Series, n: int = 15) -> list[tuple[str, int]]:
    """Top-N hashtagů (bez '#')."""
    counter: Counter[str] = Counter()
    for desc in descriptions.dropna():
        counter.update(_HASHTAG_RE.findall(str(desc).lower()))
    return counter.most_common(n)


# ---------------------------------------------------------------------------
# Sliding window metrics (sdílené)
# ---------------------------------------------------------------------------

def sliding_topic_match(df: pd.DataFrame, window: int) -> pd.Series:
    """Klouzavý průměr topic_match v %."""
    return (
        df.sort_values("interaction_number")
        ["is_topic_match"]
        .rolling(window, min_periods=1)
        .mean()
        .mul(100)
        .reset_index(drop=True)
    )


def cumulative_topic_match(df: pd.DataFrame) -> pd.Series:
    """Kumulativní topic_match v %."""
    s = df.sort_values("interaction_number")["is_topic_match"].reset_index(drop=True)
    return (s.cumsum() / (s.index + 1) * 100)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_metrics_json(metrics: dict, path: Path) -> None:
    """Bezpečný JSON dump — i s numpy/pandas typy."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        raise TypeError(f"Nelze serializovat: {type(o)}")

    path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_default),
        encoding="utf-8",
    )
