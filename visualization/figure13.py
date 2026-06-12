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

from visualization.labels import legend_label_from_filename
from visualization.paper_outputs import prepare_analysis_inputs


def render(root: str | Path = ROOT, output_name: str = "fig13_relay_jump.png") -> Path:
    root = Path(root)
    draw_dir = prepare_analysis_inputs(root)
    plt.figure(figsize=(10, 6))
    for csv_path in sorted(draw_dir.glob("*.csv")):
        if csv_path.name == "table3_metrics.csv":
            continue
        df = pd.read_csv(csv_path)
        columns = [col for col in df.columns if "relay_jump" in col.lower()]
        if not columns:
            continue
        plt.plot(df.iloc[:, 0], pd.to_numeric(df[columns[0]], errors="coerce"), label=legend_label_from_filename(csv_path.name), linewidth=2.2)
    plt.xlabel("Time step")
    plt.ylabel("relay_jump (m)")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="center left", bbox_to_anchor=(0.95, 0.5), framealpha=0.85)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    out = draw_dir / output_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    print(render())
