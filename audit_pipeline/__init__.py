"""
audit_pipeline — sdílená logika pro audit Reels & TikTok algoritmů.

Použití:
    from audit_pipeline import (
        AnalysisConfig, DEFAULT_CONFIG,
        load_csv, write_audit_log,
        enrich_reels, enrich_tiktok,
        user_engagement_metrics, age_discrepancy_metrics,
        export_metrics_json,
        setup_matplotlib, two_panel,
    )
"""
from .config import (
    AnalysisConfig,
    COLORS,
    DEFAULT_CONFIG,
    REELS_REQUIRED_COLS,
    TIKTOK_REQUIRED_COLS,
    TIKTOK_TARGET_TOPICS,
    TIKTOK_TARGET_STANCES,
)
from .features import deduplicate_videos, enrich_reels, enrich_tiktok
from .io_layer import (
    DataAudit,
    file_sha256,
    load_csv,
    validate_schema,
    write_audit_log,
)
from .metrics import (
    age_discrepancy_metrics,
    cumulative_topic_match,
    export_metrics_json,
    sliding_topic_match,
    tiktok_topic_stance_breakdown,
    top_hashtags,
    top_words,
    user_engagement_metrics,
)
from .plots import (
    plot_engagement_funnel,
    plot_predicted_topics_bar,
    plot_sliding_window,
    plot_topic_match_pie,
    save_fig,
    setup_matplotlib,
    two_panel,
)

__all__ = [
    "AnalysisConfig", "COLORS", "DEFAULT_CONFIG",
    "REELS_REQUIRED_COLS", "TIKTOK_REQUIRED_COLS",
    "TIKTOK_TARGET_TOPICS", "TIKTOK_TARGET_STANCES",
    "DataAudit", "file_sha256", "load_csv", "validate_schema", "write_audit_log",
    "deduplicate_videos", "enrich_reels", "enrich_tiktok",
    "user_engagement_metrics", "age_discrepancy_metrics",
    "tiktok_topic_stance_breakdown", "top_words", "top_hashtags",
    "sliding_topic_match", "cumulative_topic_match", "export_metrics_json",
    "setup_matplotlib", "two_panel", "save_fig",
    "plot_topic_match_pie", "plot_predicted_topics_bar",
    "plot_sliding_window", "plot_engagement_funnel",
]
