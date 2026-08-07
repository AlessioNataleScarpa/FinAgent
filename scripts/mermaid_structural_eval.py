"""Structural validation of deterministic Mermaid builders (no LLM)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.utils.mermaid import (  # noqa: E402
    build_pie_chart,
    build_xychart_line,
    build_xychart_lines,
    wrap_mermaid,
)


def _ok_pie(body: str) -> bool:
    return (
        bool(body.strip())
        and body.lstrip().startswith("pie")
        and "title" in body
        and "```" not in body
        and wrap_mermaid(body).startswith("```mermaid")
    )


def _ok_xy(body: str) -> bool:
    return (
        bool(body.strip())
        and "xychart" in body
        and "line" in body
        and "```" not in body
        and wrap_mermaid(body).startswith("```mermaid")
    )


def main() -> None:
    cases: list[tuple[str, str, Callable[[str], bool]]] = []

    pies = [
        ("pie_sectors", {"Tech": 40.5, "Health": 25.2, "Finance": 20.1, "Other": 14.2}),
        ("pie_geo", {"USA": 60, "EU": 25, "EM": 15}),
        ("pie_empty_fallback", {}),
        ("pie_negatives_filtered", {"A": 10, "B": -5, "C": 0, "D": 90}),
        ("pie_messy_labels", {'Tech "AI"\nSector': 33.333, "  Bonds  ": 66.667}),
    ]
    for name, data in pies:
        cases.append((name, build_pie_chart(name, data), _ok_pie))

    labels = [f"202{i}" for i in range(5)]
    values = [100.0, 102.5, 98.3, 105.1, 110.0]
    cases.append(("xy_single", build_xychart_line("Prezzo", labels, values), _ok_xy))
    cases.append(
        (
            "xy_multi",
            build_xychart_lines(
                "Forecast",
                labels,
                [
                    values,
                    [v * 0.9 for v in values],
                    [v * 1.1 for v in values],
                ],
            ),
            _ok_xy,
        )
    )

    passed = 0
    print("name,pass")
    for name, body, checker in cases:
        ok = checker(body)
        passed += int(ok)
        print(f"{name},{int(ok)}")

    total = len(cases)
    print(f"summary,{passed}/{total}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
