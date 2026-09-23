from typing import Sequence, Tuple

from .config import SHARED_BACKHAUL_ALLOCATION
from .dsm import DSMScene
from .link_model import FittedLinkModel


def relay_flow_capacities(
    survey_positions: Sequence[Sequence[float]],
    relay_position: Sequence[float],
    base_position: Sequence[float],
    scene: DSMScene,
    model: FittedLinkModel,
) -> Tuple[float, ...]:
    if not survey_positions:
        return ()
    access = [model.estimate(position, relay_position, scene).capacity_mbps for position in survey_positions]
    backhaul = model.estimate(relay_position, base_position, scene).capacity_mbps
    shares = SHARED_BACKHAUL_ALLOCATION[:len(access)]
    if len(shares) < len(access):
        shares = tuple(1.0 / len(access) for _ in access)
    return tuple(float(min(value, share * backhaul)) for value, share in zip(access, shares))


def shared_backhaul_capacity(
    survey_positions: Sequence[Sequence[float]],
    relay_position: Sequence[float],
    base_position: Sequence[float],
    scene: DSMScene,
    model: FittedLinkModel,
) -> float:
    return float(sum(relay_flow_capacities(survey_positions, relay_position, base_position, scene, model)))


def direct_equal_time_capacity(
    survey_positions: Sequence[Sequence[float]],
    base_position: Sequence[float],
    scene: DSMScene,
    model: FittedLinkModel,
) -> float:
    if not survey_positions:
        return 0.0
    share = 1.0 / len(survey_positions)
    return float(sum(
        share * model.estimate(position, base_position, scene).capacity_mbps
        for position in survey_positions
    ))


def average_active_snr(
    survey_positions: Sequence[Sequence[float]],
    relay_position: Sequence[float],
    base_position: Sequence[float],
    scene: DSMScene,
    model: FittedLinkModel,
) -> float:
    links = [model.estimate(position, relay_position, scene) for position in survey_positions]
    links.append(model.estimate(relay_position, base_position, scene))
    return float(sum(item.snr_db for item in links) / len(links))
