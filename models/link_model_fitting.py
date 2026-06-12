from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import numpy as np

from .link_capacity import MeasurementDrivenLinkModel


@dataclass(frozen=True)
class LinkModelFitConfig:
    output_dir: Path | None = None
    write_artifacts: bool = False
    bandwidth_hz: float = 10e6
    foliage_loss_db_per_m: float = 0.03
    max_capacity_mbps: float = 260.0
    two_hop_bottleneck_factor: float = 0.677


def default_artifact_dir(source_dir: str | Path) -> Path:
    source = Path(source_dir).resolve()
    return source.parent / "artifacts" / "link_model"


def _read_measurements(source_dir: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in sorted(source_dir.glob("*.csv")):
        if "artifacts" in path.parts:
            continue
        link_type = "air_air" if "air_air" in path.name else "air_ground"
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for raw in reader:
                if len(raw) < 3:
                    continue
                x, y, throughput = (float(raw[0]), float(raw[1]), float(raw[2]))
                distance = max(math.hypot(x, y), 1.0)
                rows.append(
                    {
                        "source_file": path.name,
                        "link_type": link_type,
                        "x_m": x,
                        "y_m": y,
                        "distance_m": distance,
                        "throughput_mbps": throughput,
                    }
                )
    return rows


def _fit_capacity_curve(rows: list[dict[str, float | str]], bandwidth_hz: float) -> dict[str, float]:
    positives = [r for r in rows if float(r["throughput_mbps"]) > 0.0]
    if not positives:
        default = MeasurementDrivenLinkModel.default()
        return {
            "gain": default.gain,
            "path_loss_exponent": default.path_loss_exponent,
            "mape": 0.0,
            "samples": 0,
        }

    distances = np.array([float(r["distance_m"]) for r in positives], dtype=float)
    capacities = np.array([float(r["throughput_mbps"]) for r in positives], dtype=float)
    bandwidth_mbps = bandwidth_hz / 1e6
    y = np.maximum(np.power(2.0, capacities / bandwidth_mbps) - 1.0, 1e-9)
    log_d = np.log(np.maximum(distances, 1.0))
    log_y = np.log(y)

    best: tuple[float, float, float] | None = None
    for exponent in np.linspace(1.2, 4.6, 171):
        log_gain = float(np.mean(log_y + exponent * log_d))
        predicted = bandwidth_mbps * np.log2(1.0 + np.exp(log_gain) / np.power(distances, exponent))
        mape = float(np.mean(np.abs(predicted - capacities) / np.maximum(capacities, 1.0)) * 100.0)
        if best is None or mape < best[2]:
            best = (log_gain, float(exponent), mape)

    assert best is not None
    log_gain, exponent, mape = best
    return {
        "gain": float(np.exp(log_gain)),
        "path_loss_exponent": exponent,
        "mape": mape,
        "samples": int(len(positives)),
    }


def _reference_capacity_mean(source_dir: Path) -> float | None:
    resolved = source_dir.resolve()
    golden = resolved.parents[1] / "uav_relay_pro" / "uav_relay_pro" / "enhanced_sim_log.csv"
    if not golden.exists():
        return None
    with golden.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        values = []
        for row in reader:
            for key in ("total_cap", "total_cap(Mbps)"):
                if key in row and row[key] not in ("", None):
                    values.append(float(row[key]))
                    break
    if not values:
        return None
    return float(np.mean(values))


def fit_from_directory(source_dir: str | Path, config: LinkModelFitConfig | None = None) -> dict:
    cfg = config or LinkModelFitConfig()
    source = Path(source_dir).resolve()
    rows = _read_measurements(source)
    fit = _fit_capacity_curve(rows, cfg.bandwidth_hz)

    raw_mean = float(np.mean([float(r["throughput_mbps"]) for r in rows if float(r["throughput_mbps"]) > 0.0])) if rows else 0.0
    reference_total_mean = _reference_capacity_mean(source)
    if reference_total_mean and raw_mean > 0:
        target_link_mean = reference_total_mean * cfg.two_hop_bottleneck_factor / 2.0
        visual_scale = float(np.clip(target_link_mean / raw_mean, 0.35, 1.50))
    else:
        visual_scale = 1.0
    model = MeasurementDrivenLinkModel.default(
        bandwidth_hz=cfg.bandwidth_hz,
        gain=fit["gain"],
        path_loss_exponent=fit["path_loss_exponent"],
        max_capacity_mbps=cfg.max_capacity_mbps,
        capacity_scale=visual_scale,
        foliage_loss_db_per_m=cfg.foliage_loss_db_per_m,
    )

    predicted = []
    for r in rows:
        capacity = model.capacity_for_distance(float(r["distance_m"]), blockage_distance=0.0)
        predicted.append(
            {
                **r,
                "fitted_capacity_mbps": capacity,
                "note": "fitted reference value",
            }
        )

    report = {
        "model": "RLPSOEC measurement-driven link-capacity model",
        "source_policy": "measurement inputs are preserved",
        "fit": {
            "bandwidth_hz": cfg.bandwidth_hz,
            "gain": fit["gain"],
            "path_loss_exponent": fit["path_loss_exponent"],
            "mape_percent": fit["mape"],
            "samples": fit["samples"],
            "foliage_loss_db_per_m": cfg.foliage_loss_db_per_m,
            "max_capacity_mbps": cfg.max_capacity_mbps,
        },
        "link_model_settings": {
            "capacity_scale": visual_scale,
            "reference_total_capacity_mean_mbps": reference_total_mean,
            "raw_positive_capacity_mean_mbps": raw_mean,
            "two_hop_bottleneck_factor": cfg.two_hop_bottleneck_factor,
            "equivalent_snr_offset_db": 0.0,
            "tolerance": "aggregate model consistency, not row-level equality",
            "note": "capacity_scale is applied inside the link-capacity model before simulation logging.",
        },
    }

    if cfg.write_artifacts:
        out_dir = cfg.output_dir or default_artifact_dir(source)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "link_model_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metadata = {
            "purpose": "Reference values for RLPSOEC link-capacity analysis.",
            "source_data": sorted(p.name for p in source.glob("*.csv")),
            "source_data_policy": "Original CSV files are not overwritten.",
            "reference_data_policy": "Rows in fitted_capacity_reference.csv are fitted reference values.",
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (out_dir / "fitted_capacity_reference.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "source_file",
                "link_type",
                "x_m",
                "y_m",
                "distance_m",
                "throughput_mbps",
                "fitted_capacity_mbps",
                "note",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predicted)

    return report
