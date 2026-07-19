"""Helpers to build valid Mermaid diagrams for OpenWebUI rendering."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Number = Union[int, float]


def _clean_label(label: str) -> str:
    text = str(label or "N/D").replace('"', "'").replace("\n", " ").strip()
    return text[:48] if text else "N/D"


def wrap_mermaid(diagram: str) -> str:
    body = (diagram or "").strip()
    if body.startswith("```"):
        return body
    return f"```mermaid\n{body}\n```"


def build_pie_chart(
    title: str,
    slices: Mapping[str, Number] | Sequence[Tuple[str, Number]],
    *,
    show_data: bool = True,
) -> str:
    """
    Build a Mermaid pie chart.

    OpenWebUI renders blocks like:
    ```mermaid
    pie showData
        title Allocazione
        \"Tech\" : 40
    ```
    """
    items: List[Tuple[str, float]] = []
    if isinstance(slices, Mapping):
        iterable: Iterable[Tuple[str, Number]] = slices.items()
    else:
        iterable = slices

    for label, value in iterable:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        items.append((_clean_label(label), round(numeric, 2)))

    if not items:
        items = [("Dati non disponibili", 1.0)]

    header = "pie showData" if show_data else "pie"
    lines = [header, f"    title {_clean_label(title)}"]
    for label, value in items:
        lines.append(f'    "{label}" : {value}')
    return "\n".join(lines)


def build_xychart_line(
    title: str,
    x_labels: Sequence[str],
    y_values: Sequence[Number],
    *,
    y_axis_label: str = "Prezzo",
) -> str:
    """
    Build a Mermaid XY line chart (xychart / xychart-beta compatible).
    """
    labels = [_clean_label(label) for label in x_labels]
    values: List[float] = []
    for value in y_values:
        try:
            values.append(round(float(value), 2))
        except (TypeError, ValueError):
            continue

    n = min(len(labels), len(values))
    if n == 0:
        labels = ["N/D"]
        values = [0.0]
        n = 1
    else:
        labels = labels[:n]
        values = values[:n]

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min = max(0.0, y_min * 0.95)
        y_max = y_max * 1.05 if y_max else 1.0
    else:
        padding = (y_max - y_min) * 0.08
        y_min = max(0.0, y_min - padding)
        y_max = y_max + padding

    x_axis = ", ".join(f'"{label}"' for label in labels)
    y_series = ", ".join(str(v) for v in values)
    y_label = _clean_label(y_axis_label)

    return "\n".join(
        [
            "xychart-beta",
            f'    title "{_clean_label(title)}"',
            f"    x-axis [{x_axis}]",
            f'    y-axis "{y_label}" {round(y_min, 2)} --> {round(y_max, 2)}',
            f"    line [{y_series}]",
        ]
    )


def default_sector_slices() -> Dict[str, float]:
    return {
        "Information Technology": 23.5,
        "Financials": 15.2,
        "Healthcare": 12.1,
        "Industrials": 11.0,
        "Consumer Discretionary": 10.4,
        "Communication": 7.5,
        "Altri": 20.3,
    }


def default_asset_slices() -> Dict[str, float]:
    return {
        "Azioni": 98.0,
        "Liquidita / Altro": 2.0,
    }
