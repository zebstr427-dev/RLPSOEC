from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from config.experiment_profiles import get_profile, paper_targets
from metrics.acceptance import assess_table3_support
from metrics.summary import summarize_csv


MODE_OUTPUTS = {
    "full": ("enhanced_sim_log.csv", "RLPSOEC.csv"),
    "no_ppo": ("test-noPPO/ablation_sim_log.csv", "RLPSOEC w-o PPO.csv"),
    "no_trigger": ("test-noTrigger/notrigger_sim_log.csv", "RLPSOEC w-o Trigger.csv"),
    "static": ("test-static/enhanced_sim_log.csv", "Centralized deployment.csv"),
}


def collect_summaries(root: str | Path) -> dict[str, dict]:
    root = Path(root)
    summaries = {}
    for mode, (source_rel, _name) in MODE_OUTPUTS.items():
        summaries[mode] = summarize_csv(root / source_rel, get_profile(mode))
    return summaries


def prepare_analysis_inputs(root: str | Path, output_dir: str | Path | None = None) -> Path:
    root = Path(root)
    output = Path(output_dir) if output_dir is not None else root / "draw" / "analysis_inputs"
    output.mkdir(parents=True, exist_ok=True)

    copied = {}
    for mode, (source_rel, draw_name) in MODE_OUTPUTS.items():
        src = root / source_rel
        dst = output / draw_name
        shutil.copyfile(src, dst)
        copied[mode] = {
            "source": source_rel,
            "draw_csv": str(dst.relative_to(root)),
            "label": get_profile(mode).label,
        }

    metadata = {
        "purpose": "Prepared inputs for reported Fig.13/Fig.14/Table3 analysis.",
        "source_data_policy": "Original speed_modeling_data CSV files are not modified.",
        "processing_policy": "These CSVs are prepared from simulation logs for metric aggregation.",
        "modes": copied,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def write_table3_csv(root: str | Path, output_dir: str | Path | None = None) -> Path:
    root = Path(root)
    output = Path(output_dir) if output_dir is not None else root / "draw" / "analysis_inputs"
    output.mkdir(parents=True, exist_ok=True)
    summaries = collect_summaries(root)
    path = output / "table3_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "label", "capacity_mbps", "trigger_rate", "relay_jump_m", "optimization_time_s", "success_rate"])
        for mode in ("full", "no_ppo", "no_trigger", "static"):
            profile = get_profile(mode)
            summary = summaries[mode]
            target = profile.target
            writer.writerow(
                [
                    mode,
                    profile.label,
                    summary["capacity"],
                    summary["trigger_rate"] if target.trigger_rate is not None else "",
                    summary["relay_jump"] if target.relay_jump_m is not None else "",
                    summary["opt_time"] if target.optimization_time_s is not None else "",
                    summary["success_rate"] if target.success_rate is not None else "",
                ]
            )
    return path


def write_acceptance_report(root: str | Path, output_dir: str | Path | None = None) -> Path:
    root = Path(root)
    output = Path(output_dir) if output_dir is not None else root / "draw" / "analysis_inputs"
    output.mkdir(parents=True, exist_ok=True)
    summaries = collect_summaries(root)
    report = assess_table3_support(summaries, paper_targets())
    report["summaries"] = summaries
    path = output / "result_acceptance_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def generate_paper_artifacts(root: str | Path) -> dict[str, str]:
    draw_dir = prepare_analysis_inputs(root)
    table_csv = write_table3_csv(root, draw_dir)
    acceptance = write_acceptance_report(root, draw_dir)
    return {
        "draw_inputs": str(draw_dir),
        "table3_csv": str(table_csv),
        "acceptance_report": str(acceptance),
    }
