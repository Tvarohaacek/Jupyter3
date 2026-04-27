"""
Načítání a validace datasetů.

Každé `load_*` voláni:
  1. ověří existenci souboru
  2. spočítá SHA-256 hash (pro audit log)
  3. načte CSV s explicitními encoding/dtype pravidly
  4. ověří přítomnost všech povinných sloupců

Pokud cokoli z toho selže, funkce raise-ne s konkrétní chybou
a NEPOKRAČUJE — to je záměr, "fail loud and early".
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataAudit:
    """Snapshot stavu datasetu — slouží jako součást reprodukovatelného logu."""
    source_path: str
    sha256: str
    n_rows: int
    n_cols: int
    columns: list[str]
    null_counts: dict[str, int]
    duplicate_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------

def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """SHA-256 hash souboru po blocích — bezpečné i pro velká CSV."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(df: pd.DataFrame, required_cols: Iterable[str], source: str) -> None:
    """Ověří přítomnost povinných sloupců. Raise s konkrétním seznamem chybějících."""
    required = set(required_cols)
    actual = set(df.columns)
    missing = required - actual

    if missing:
        raise ValueError(
            f"Dataset '{source}' postrádá povinné sloupce: {sorted(missing)}. "
            f"Skutečné sloupce: {sorted(actual)}"
        )

    log.info("Schema OK pro %s (%d sloupců, %d řádků)", source, len(actual), len(df))


def audit_dataframe(df: pd.DataFrame, sha: str, source_path: Path) -> DataAudit:
    """Vytvoří snapshot pro reprodukovatelný log."""
    null_counts = (
        df.isna()
        .sum()
        .loc[lambda s: s > 0]   # jen sloupce s alespoň jedním NaN
        .to_dict()
    )
    return DataAudit(
        source_path=str(source_path),
        sha256=sha,
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=list(df.columns),
        null_counts={k: int(v) for k, v in null_counts.items()},
        duplicate_rows=int(df.duplicated().sum()),
    )


# ---------------------------------------------------------------------------

def load_csv(
    path: Path | str,
    required_cols: Iterable[str],
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, DataAudit]:
    """
    Načte CSV s validací a auditem.

    Vrací (df, audit). Audit je určen pro zápis do `output/data_audit.txt`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Soubor '{path}' nenalezen.")
    if not path.is_file():
        raise ValueError(f"'{path}' není soubor.")

    sha = file_sha256(path)
    log.info("Načítám %s (SHA-256=%s...)", path, sha[:12])

    df = pd.read_csv(path, encoding=encoding)
    validate_schema(df, required_cols, source=str(path))
    audit = audit_dataframe(df, sha, path)

    return df, audit


def write_audit_log(audits: Iterable[DataAudit], output_path: Path) -> None:
    """Zapíše čitelný plain-text audit log (součást reprodukovatelnosti)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("DATA AUDIT LOG")
    lines.append("=" * 70)

    for a in audits:
        lines.append("")
        lines.append(f"Soubor:        {a.source_path}")
        lines.append(f"SHA-256:       {a.sha256}")
        lines.append(f"Řádků:         {a.n_rows:,}")
        lines.append(f"Sloupců:       {a.n_cols}")
        lines.append(f"Duplicitních:  {a.duplicate_rows}")
        if a.null_counts:
            lines.append("Sloupce s NaN:")
            for col, n in sorted(a.null_counts.items()):
                pct = n / a.n_rows * 100
                lines.append(f"   {col}: {n} ({pct:.1f}%)")
        else:
            lines.append("Sloupce s NaN: žádné")
        lines.append("-" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Audit log zapsán: %s", output_path)
