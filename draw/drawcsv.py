import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.labels import legend_label_from_filename

column_substring = "relay_jump"
FONT_SCALE = 1.5


def prettify_label(raw_name: str) -> str:
    return legend_label_from_filename(raw_name)


def main():
    base = 12
    plt.rcParams.update(
        {
            "font.size": base * FONT_SCALE,
            "axes.titlesize": (base + 2) * FONT_SCALE,
            "axes.labelsize": base * FONT_SCALE,
            "xtick.labelsize": (base - 2) * FONT_SCALE,
            "ytick.labelsize": (base - 2) * FONT_SCALE,
            "legend.fontsize": (base - 2) * FONT_SCALE,
            "figure.titlesize": (base + 4) * FONT_SCALE,
        }
    )

    analysis_dir = Path(os.getcwd()) / "analysis_inputs"
    search_dir = analysis_dir if analysis_dir.exists() else Path(os.getcwd())
    csv_files = glob.glob(str(search_dir / "*.csv"))
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    plt.figure(figsize=(10, 6))
    y_label_name = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        matched_cols = [col for col in df.columns if column_substring.lower() in col.lower()]
        if not matched_cols:
            print(f"[skip] {csv_file} has no column containing {column_substring!r}.")
            continue

        col_name = matched_cols[0]
        if y_label_name is None:
            y_label_name = col_name
        x_vals = df.iloc[:, 0]
        y_vals = df[col_name]
        plt.plot(x_vals, y_vals, label=legend_label_from_filename(csv_file))

    if y_label_name is None:
        print(f"No CSV file had a column containing {column_substring!r}.")
        return

    plt.xticks(range(0, 501, 100))
    plt.xlabel("Time step")
    plt.ylabel(y_label_name, rotation=90)
    plt.title("")
    plt.grid(True, linewidth=1.2)
    ax = plt.gca()
    ax.legend(loc="center left", bbox_to_anchor=(0.95, 0.5), framealpha=0.85)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    plt.savefig("relay_jump_all_csv.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
