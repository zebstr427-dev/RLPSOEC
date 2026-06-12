from __future__ import annotations

from pathlib import Path


def normalize_label_key(value: str) -> str:
    return "".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def legend_label_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    key = normalize_label_key(stem)

    if any(k in key for k in ("central", "centralized", "static", "baseline", "center")):
        return "Centralized deployment"
    if any(k in key for k in ("withoutppo", "woppo", "noppo")):
        return "RLPSOEC w/o PPO"
    if any(k in key for k in ("withouttrigger", "wotrig", "notrig", "notrigger")):
        return "RLPSOEC w/o Trigger"
    if any(k in key for k in ("rlpsoec", "caehop", "edehop", "ehop")):
        return "RLPSOEC"
    return stem
