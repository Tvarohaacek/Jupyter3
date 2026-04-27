"""
Centralizovaná konfigurace pro audit sociálních médií.

Všechny "magické konstanty" (datum sběru, roky narození, sliding windows,
barvy, cesty) jsou definovány zde. Změna na jednom místě se propaguje
do celé pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Schémata datasetů — povinné sloupce. Validace selže, pokud chybí.
# ---------------------------------------------------------------------------

REELS_REQUIRED_COLS: tuple[str, ...] = (
    "topic",
    "predicted_topic",
    "video_action_watch",
    "video_action_like",
    "video_action_bookmark",
    "video_time_duration",
    "video_author",
    "video_description",
    "date_of_birth",
)

TIKTOK_REQUIRED_COLS: tuple[str, ...] = (
    "topic",
    "stance",
    "predicted_topic",
    "predicted_topic_match",
    "predicted_stance_match",
    "video_action_watch",
    "video_action_like",
    "video_action_skip",
    "video_action_bookmark",
    "video_time_duration",
    "interaction_number",
    "user_email",
    "run_id",
    "date_of_birth",
)

TIKTOK_TARGET_TOPICS: tuple[str, ...] = (
    "donald_trump",
    "vaccines",
    "flatearth",
    "climate_change",
)

TIKTOK_TARGET_STANCES: tuple[str, ...] = ("support", "oppose")


# ---------------------------------------------------------------------------
# Vizuální konstanty (sdílené mezi oběma analýzami)
# ---------------------------------------------------------------------------

COLORS: dict = {
    # primární profilové barvy
    "user1": "#e74c3c",       # nevhodný / podporující
    "user2": "#2ecc71",       # neutrální / kontrolní
    "random": "#95a5a6",      # generický feed
    # sémantické kategorie
    "topic_stance": "#2ecc71",
    "topic_opposite": "#e74c3c",
    "recipes": "#3498db",
    # akce
    "watch": "#3498db",
    "like": "#f39c12",
    "bookmark": "#9b59b6",
    "skip": "#e74c3c",
    # neutrální
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "declared_age": "#3498db",
    "real_age": "#e74c3c",
}

PLOT_RC_PARAMS: dict = {
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}


# ---------------------------------------------------------------------------
# Konfigurace běhu — všechny parametry, které ovlivňují výstup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnalysisConfig:
    """
    Konfigurace jednotného běhu analýzy.

    Frozen=True → instance je immutable, takže není možné omylem změnit
    parametry během běhu (klíčové pro reprodukovatelnost).
    """
    # datum, ke kterému se vztahuje výpočet věku
    collection_date: datetime

    # rok, který profily v audit experimentu deklarovaly v UI
    declared_birth_year: int

    # rok, který odpovídá reálnému modelovému věku (mladší)
    real_birth_year: int

    # parametry vizualizací
    sliding_window_default: int = 30
    sliding_window_engagement: int = 50
    sliding_window_pattern: int = 40

    # výstupy
    output_dir: Path = field(default_factory=lambda: Path("output"))
    plots_subdir: str = "plots"

    # metadata
    project_label: str = "social_media_audit"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / self.plots_subdir

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    @property
    def audit_log_path(self) -> Path:
        return self.output_dir / "data_audit.txt"


# Defaultní config odpovídající existujícímu stavu obou notebooků
DEFAULT_CONFIG = AnalysisConfig(
    collection_date=datetime(2026, 3, 16),
    declared_birth_year=2008,
    real_birth_year=2012,
)
