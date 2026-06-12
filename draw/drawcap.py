#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.labels import legend_label_from_filename

Y_COL_KEY = "total_cap"
FORCE_X_COL = None
FONT_BASE = 16
FONT_LABEL = 20
FONT_TICK = 16
FONT_LEGEND = 16
LINE_WIDTH = 2.4
FIG_W, FIG_H = 12.5, 7.2
DPI = 300
OUT_PNG = "total_cap_all_csv.png"


def normalize(value: str) -> str:
    return "".join(str(value).lower().strip().split())


def pick_y_column(columns):
    key = normalize(Y_COL_KEY)
    for column in columns:
        if key in normalize(column):
            return column
    return None


def main():
    analysis_dir = Path(os.getcwd()) / "analysis_inputs"
    search_dir = analysis_dir if analysis_dir.exists() else Path(os.getcwd())
    csv_files = sorted(glob.glob(str(search_dir / "*.csv")))
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    plt.rcParams.update(
        {
            "font.size": FONT_BASE,
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
        }
    )

    plt.figure(figsize=(FIG_W, FIG_H))
    plotted = 0
    used_labels = set()

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"[skip] failed to read {os.path.basename(file_path)}: {exc}")
            continue

        y_col = pick_y_column(df.columns)
        if y_col is None:
            print(f"[skip] {os.path.basename(file_path)} has no column containing {Y_COL_KEY!r}")
            continue

        if FORCE_X_COL is None:
            x = np.arange(len(df))
        else:
            if FORCE_X_COL not in df.columns:
                print(f"[skip] {os.path.basename(file_path)} has no x column {FORCE_X_COL!r}")
                continue
            x = pd.to_numeric(df[FORCE_X_COL], errors="coerce").to_numpy()

        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        label = legend_label_from_filename(file_path)
        if label in used_labels:
            label = f"{label} ({Path(file_path).stem})"
        used_labels.add(label)
        plt.plot(x, y, linewidth=LINE_WIDTH, label=label)
        plotted += 1

    if plotted == 0:
        print("No plottable total-capacity series were found.")
        return

    plt.xlabel("Time step")
    plt.ylabel("total_cap (Mbps)")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
