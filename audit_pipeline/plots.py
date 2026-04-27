"""
Plotting helpers.

Hlavní cíl: zlikvidovat 15× opakovaný 2-panel pattern z `reels.ipynb`.
Místo psaní `for idx, (df, label, color) in enumerate(...)` v každé buňce
voláme `two_panel(plot_fn, ...)` a předáváme pouze tu část, která se mění.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .config import COLORS, PLOT_RC_PARAMS


# ---------------------------------------------------------------------------

def setup_matplotlib() -> None:
    """
    Nastaví matplotlib styl pro celou analýzu.
    Volá se jednou na začátku notebooku — tím odpadne kopírování rcParams.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_RC_PARAMS)


def ensure_dir(path: Path) -> Path:
    """Vytvoří adresář pokud neexistuje (řeší crash z reels.ipynb)."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, name: str, plots_dir: Path) -> Path:
    """Uloží graf do plots_dir/name.png. Vrací cestu (užitečné pro logování)."""
    ensure_dir(plots_dir)
    out = plots_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return out


# ---------------------------------------------------------------------------
# Helper, který nahrazuje 15× opakovaný pattern v reels.ipynb
# ---------------------------------------------------------------------------

PlotFn = Callable[[plt.Axes, pd.DataFrame, str, str], None]


def two_panel(
    plot_fn: PlotFn,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "User 1 — nevhodný obsah",
    label_b: str = "User 2 — neutrální obsah",
    color_a: str = COLORS["user1"],
    color_b: str = COLORS["user2"],
    suptitle: str | None = None,
    figsize: tuple[float, float] = (16, 6),
    savename: str | None = None,
    plots_dir: Path | None = None,
    **plot_kwargs: Any,
) -> plt.Figure:
    """
    Renderuje dvojpanelový graf: levý panel = df_a, pravý panel = df_b.

    `plot_fn(ax, df, label, color, **kwargs)` je callback, který obsahuje
    pouze logiku konkrétního grafu (histogram / scatter / area / ...).

    Eliminuje duplicitu z reels.ipynb (15 opakování stejného forloop patternu).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_fn(axes[0], df_a, label_a, color_a, **plot_kwargs)
    plot_fn(axes[1], df_b, label_b, color_b, **plot_kwargs)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.03)

    fig.tight_layout()

    if savename and plots_dir is not None:
        save_fig(fig, savename, plots_dir)

    return fig


# ---------------------------------------------------------------------------
# Konkrétní plot funkce — používá se s two_panel(...)
# ---------------------------------------------------------------------------

def plot_topic_match_pie(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    """Donut chart: topic match vs náhodný obsah."""
    n = len(df)
    n_match = int(df["is_topic_match"].sum())
    n_random = n - n_match

    ax.pie(
        [n_match, n_random],
        labels=[
            f"Topic match\n{n_match} ({n_match / n * 100:.2f}%)",
            f"Náhodný obsah\n{n_random} ({n_random / n * 100:.1f}%)",
        ],
        colors=[color, COLORS["random"]],
        startangle=90,
        explode=(0.05, 0),
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    ax.add_patch(plt.Circle((0, 0), 0.5, fc="white"))
    ax.text(0, 0, f"n={n}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_title(label, fontweight="bold", fontsize=13)


def plot_predicted_topics_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    """Horizontal bar chart distribuce predicted topics."""
    counts = df["predicted_topic"].fillna("(NaN)").value_counts()
    bar_colors = [color if t != "random" else COLORS["random"] for t in counts.index]

    bars = ax.barh(range(len(counts)), counts.values, color=bar_colors,
                   edgecolor="white", linewidth=1.5)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels([str(t).replace("_", " ").title() for t in counts.index])
    ax.invert_yaxis()

    n = len(df)
    for bar, val in zip(bars, counts.values):
        pct = val / n * 100
        ax.text(val + n * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_xlabel("Počet videí")
    ax.set_title(label, fontweight="bold")
    ax.set_xlim(0, max(counts.values) * 1.25)


def plot_sliding_window(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    color: str,
    *,
    window: int = 30,
) -> None:
    """Sliding window stacked area: topic match vs random."""
    sorted_df = df.sort_values("interaction_number").reset_index(drop=True)
    rolling_match = sorted_df["is_topic_match"].rolling(window, min_periods=1).mean() * 100

    ax.fill_between(sorted_df.index, 0, rolling_match,
                    alpha=0.7, label="Topic match", color=color)
    ax.fill_between(sorted_df.index, rolling_match, 100,
                    alpha=0.4, label="Náhodný obsah", color=COLORS["random"])
    ax.set_xlabel("Interakce #")
    ax.set_ylabel("% feedu (klouzavé okno)")
    ax.set_title(label, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)


def plot_engagement_funnel(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    color: str,
) -> None:
    """Engagement funnel: zobrazeno → dokoukáno → like → bookmark."""
    total = len(df)
    stages = ["Zobrazeno", "Dokoukáno", "Liked", "Bookmarked"]
    values = [
        total,
        int(df["video_action_watch"].sum()),
        int(df["video_action_like"].sum()),
        int(df.get("video_action_bookmark", pd.Series(False, index=df.index)).sum()),
    ]
    funnel_colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    y_pos = list(range(len(stages) - 1, -1, -1))

    bars = ax.barh(y_pos, values, color=funnel_colors,
                   edgecolor="white", linewidth=2, height=0.6)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total else 0
        ax.text(val + total * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold")

    # Konverze
    for i in range(len(values) - 1):
        if values[i] > 0:
            conv = values[i + 1] / values[i] * 100
            ax.annotate(f"→ {conv:.1f}%",
                        xy=(values[i + 1], y_pos[i + 1] + 0.45),
                        fontsize=9, color="gray", style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.set_xlabel("Počet videí")
    ax.set_title(label, fontweight="bold")
    ax.set_xlim(0, total * 1.35 if total else 1)
