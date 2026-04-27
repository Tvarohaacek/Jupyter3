"""
Feature engineering pro Reels a TikTok datasety.

Klíčové principy:
  * vše vektorizované (žádné `.apply(..., axis=1)`)
  * NaN-safe (žádné tiché chyby na chybějících hodnotách)
  * deterministické (žádné `Timestamp.now()`, vše vůči `config.collection_date`)
"""
from __future__ import annotations

from datetime import datetime, date

import numpy as np
import pandas as pd

from .config import AnalysisConfig


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def _parse_dob(dob_value: str) -> datetime | None:
    """
    Bezpečný parser pro datum narození.
    Zkouší několik formátů; při selhání vrací None místo crashe.
    """
    if pd.isna(dob_value):
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str(dob_value).strip(), fmt)
        except ValueError:
            continue
    return None


def _safe_replace_year(d: datetime, year: int) -> datetime:
    """
    Náhrada `dob.replace(year=...)`, která NEspadne na 29.2.
    Pokud target rok není přestupný, posune datum na 28.2.
    """
    try:
        return d.replace(year=year)
    except ValueError:  # 29.2. → ne-přestupný rok
        return d.replace(year=year, day=28)


def _years_between(d_from: datetime, d_to: datetime) -> float:
    """Počet let v desetinné formě, deterministický."""
    delta = d_to - d_from
    return delta.days / 365.25


# ---------------------------------------------------------------------------
# Reels enrichment
# ---------------------------------------------------------------------------

def enrich_reels(
    df: pd.DataFrame,
    user_label: str,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """
    Přidá do Reels DataFrame odvozené sloupce:
      * user_label
      * interaction_number  (1..N podle pořadí v souboru)
      * interest_topics     (List[str], z čárkou oddělené topic kolony)
      * is_topic_match      (predicted_topic není 'random' a není NaN)
      * declared_age, real_age  (vůči config.collection_date)
    """
    df = df.copy()

    df["user_label"] = user_label
    df["interaction_number"] = np.arange(1, len(df) + 1)

    # NaN-safe split: prázdné topic kolony → prázdný list
    df["interest_topics"] = (
        df["topic"]
        .fillna("")
        .map(lambda s: [t.strip() for t in s.split(",") if t.strip()])
    )

    # NaN-safe: chybějící predicted_topic se neoznačí jako match
    pt = df["predicted_topic"].fillna("random").astype(str).str.strip().str.lower()
    df["is_topic_match"] = pt.ne("random") & pt.ne("")

    # Věk — počítáno deterministicky vůči config.collection_date
    declared_age, real_age = _compute_ages(df, config)
    df["declared_age"] = declared_age
    df["real_age"] = real_age

    # Bezpečné boolean konverze pro action sloupce
    for col in ("video_action_watch", "video_action_like", "video_action_bookmark"):
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)

    return df


def _compute_ages(df: pd.DataFrame, config: AnalysisConfig) -> tuple[float, float]:
    """
    Spočítá deklarovaný a reálný věk z prvního validního DOB v datasetu.
    Pokud nelze parsovat, vrací (NaN, NaN) — analýza pokračuje, ale chybí info.
    """
    dob = next(
        (parsed for v in df["date_of_birth"].head(20) if (parsed := _parse_dob(v))),
        None,
    )
    if dob is None:
        return float("nan"), float("nan")

    declared_dob = _safe_replace_year(dob, config.declared_birth_year)
    real_dob = _safe_replace_year(dob, config.real_birth_year)

    return (
        _years_between(declared_dob, config.collection_date),
        _years_between(real_dob, config.collection_date),
    )


# ---------------------------------------------------------------------------
# Deduplikace
# ---------------------------------------------------------------------------

def _first_non_random(s: pd.Series) -> str:
    """
    Vrací první non-random / non-NaN hodnotu z group.
    Pokud žádná taková není, vrací 'random'.

    Použití: aggregátor pro `predicted_topic` při dedupu, aby zůstal
    konzistentní s `is_topic_match` (který agreguje přes `any`).
    """
    for v in s:
        if pd.notna(v) and str(v).strip().lower() != "random":
            return v
    return "random"


def deduplicate_videos(
    df: pd.DataFrame,
    key: str = "video_id",
) -> pd.DataFrame:
    """
    Sjednotí opakované výskyty stejného videa do jednoho řádku.

    V auditních datech se některé video objeví ve feedu mnohokrát (v jednom
    z testovacích datasetů 577×). Pokud bychom procenta počítali přes řádky,
    jediné spamované video by dominovalo všem statistikám. Tato funkce
    proto agreguje per video_id následovně:

      * action sloupce (watch/like/bookmark) → logické OR: pokud byl uživatel
        kdykoli aktivní, video se počítá jako interagované;
      * interaction_number → min (pořadí prvního výskytu v session);
      * is_topic_match → any (pokud byl klasifikován jako match alespoň 1×);
      * ostatní (autor, popis, predicted_topic, ...) → first.

    Vrací DF seřazený podle interaction_number.
    """
    if key not in df.columns:
        raise ValueError(f"Sloupec '{key}' není v DataFrame.")

    n_before = len(df)
    n_unique = df[key].nunique()
    if n_before == n_unique:
        return df.reset_index(drop=True)

    # Strategie agregace pro každý sloupec
    agg_map: dict = {}
    for col in df.columns:
        if col == key:
            continue
        if col in ("video_action_watch", "video_action_like",
                   "video_action_bookmark", "video_action_skip",
                   "is_topic_match", "is_topic_stance",
                   "is_topic_opposite", "is_recipes", "is_random"):
            agg_map[col] = "any"
        elif col == "interaction_number":
            agg_map[col] = "min"
        elif col == "predicted_topic":
            # Preferuj non-random / non-NaN klasifikaci, jinak vezmi první.
            # Nutné, aby is_topic_match=True bylo konzistentní s predicted_topic.
            agg_map[col] = _first_non_random
        else:
            agg_map[col] = "first"

    deduped = (
        df.groupby(key, as_index=False, sort=False)
        .agg(agg_map)
        .sort_values("interaction_number" if "interaction_number" in df.columns else key)
        .reset_index(drop=True)
    )
    return deduped


# ---------------------------------------------------------------------------
# TikTok enrichment
# ---------------------------------------------------------------------------

def enrich_tiktok(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Přidá do TikTok DataFrame odvozené sloupce a přečistí typy.
    Vše vektorizované — žádné `.apply(axis=1)`.
    """
    df = df.copy()

    # Identifikátory
    df["session_id"] = df["run_id"]
    df["user_id"] = (
        df["user_email"].astype(str)
        + " | " + df["topic"].astype(str)
        + " | " + df["stance"].astype(str)
    )

    # Boolean sloupce — NaN-safe
    bool_cols = [
        "predicted_topic_match", "predicted_stance_match",
        "video_action_watch", "video_action_like",
        "video_action_skip", "video_action_bookmark",
    ]
    for col in bool_cols:
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)

    # Kategorie videa — vektorizováno přes np.select (zhruba 50× rychlejší
    # než df.apply(get_video_category, axis=1)).
    is_recipes = df["predicted_topic"].fillna("").eq("recipes")
    is_topic = df["predicted_topic_match"]
    is_stance = df["predicted_stance_match"]

    df["is_recipes"] = is_recipes
    df["is_topic_stance"] = is_topic & is_stance
    df["is_topic_opposite"] = is_topic & ~is_stance
    df["is_random"] = ~is_topic & ~is_recipes

    df["video_category"] = np.select(
        condlist=[
            df["is_topic_stance"],
            df["is_topic_opposite"],
            df["is_recipes"],
        ],
        choicelist=["Topic+Stance", "Topic+Opačný", "Recepty"],
        default="Náhodný",
    )

    # Věk — deterministicky vůči config.collection_date.
    # `format='mixed'` tiše akceptuje různé formáty (DD.MM.YYYY i YYYY-MM-DD).
    df["date_of_birth"] = pd.to_datetime(
        df["date_of_birth"], errors="coerce", format="mixed"
    )
    age_days = (config.collection_date - df["date_of_birth"]).dt.days
    df["age"] = (age_days / 365.25).astype("Float64")  # nullable float

    return df
