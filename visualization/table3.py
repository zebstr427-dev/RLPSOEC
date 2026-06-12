from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.paper_outputs import write_table3_csv


METRICS = [
    ("capacity_mbps", "Total Capacity\n(Mbps)", "{:.3f}", 0.60, 1.045),
    ("trigger_rate", "Trigger Rate", "{:.3f}", 0.66, 0.94),
    ("relay_jump_m", "Relay Jump\n(m)", "{:.3f}", 0.66, 0.94),
    ("optimization_time_s", "Optimization Time\n(s)", "{:.3f}", 0.66, 0.94),
    ("success_rate", "Success Rate", "{:.3f}", 0.66, 0.94),
]


def render(root: str | Path = ROOT, output_name: str = "table3_metrics_graphical_comparison_final.png") -> Path:
    root = Path(root)
    table_csv = write_table3_csv(root)
    df = pd.read_csv(table_csv)
    methods = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(13.6, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.58, len(methods) + 0.95)
    ax.axis("off")

    label_x = 0.3
    panel_start_x = 21.5
    panel_gap = 1.35
    panel_width = (99.2 - panel_start_x - panel_gap * (len(METRICS) - 1)) / len(METRICS)
    y_positions = list(range(len(methods)))[::-1]

    ax.text(
        50,
        len(methods) + 0.66,
        "Graphical Comparison of Average Performance Metrics over the Full Mission Horizon",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(label_x, len(methods) + 0.08, "Method", ha="left", va="center", fontsize=14, fontweight="bold")
    for y, method in zip(y_positions, methods):
        ax.text(label_x, y, method, ha="left", va="center", fontsize=13.2)

    for idx, (column, title, fmt, bar_ratio, value_ratio) in enumerate(METRICS):
        x0 = panel_start_x + idx * (panel_width + panel_gap)
        values = pd.to_numeric(df[column], errors="coerce").tolist()
        valid = [value for value in values if pd.notna(value)]
        max_value = max(valid) if valid else 1.0
        ax.text(x0 + panel_width / 2, len(methods) + 0.08, title, ha="center", va="center", fontsize=13.2, fontweight="bold")
        if idx > 0:
            ax.axvline(x0 - panel_gap / 2, ymin=0.13, ymax=0.86, linewidth=0.9, alpha=0.45)
        value_x = x0 + panel_width * value_ratio
        for y, value in zip(y_positions, values):
            if pd.isna(value):
                ax.text(x0 + panel_width * 0.50, y, "N/A", ha="center", va="center", fontsize=11.6, alpha=0.78)
                continue
            width = (float(value) / max_value) * (panel_width * bar_ratio) if max_value > 0 else 0.0
            ax.barh(y=y, width=width, left=x0, height=0.60, alpha=0.82)
            ax.text(value_x, y, fmt.format(float(value)), ha="right", va="center", fontsize=11.6)

    plt.tight_layout(pad=0.12)
    out = table_csv.parent / output_name
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(render())
