from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.figure13 import render as render_figure13
from visualization.figure14 import render as render_figure14
from visualization.paper_outputs import generate_paper_artifacts
from visualization.table3 import render as render_table3


def main():
    artifacts = generate_paper_artifacts(ROOT)
    artifacts["fig13"] = str(render_figure13(ROOT))
    artifacts["fig14"] = str(render_figure14(ROOT))
    artifacts["table3"] = str(render_table3(ROOT))
    print("Generated RLPSOEC analysis outputs:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
