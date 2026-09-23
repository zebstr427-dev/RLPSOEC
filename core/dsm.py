from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config import CELL_SIZE_M, MAP_SIZE_M


@dataclass
class DSMScene:
    surface: np.ndarray
    cell_size: float
    vegetation: Optional[np.ndarray] = None

    @property
    def height(self):
        return int(self.surface.shape[0])

    @property
    def width(self):
        return int(self.surface.shape[1])

    def _index(self, x: float, y: float):
        return int(np.floor(y / self.cell_size)), int(np.floor(x / self.cell_size))

    def surface_height(self, x: float, y: float) -> float:
        row, col = self._index(x, y)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return float("nan")
        value = self.surface[row, col]
        return float(value) if np.isfinite(value) else float("nan")

    def _vegetation_at(self, row: int, col: int) -> bool:
        if self.vegetation is None:
            return False
        return bool(self.vegetation[row, col] > 0)

    def blocked_length(self, start, end) -> float:
        p0 = np.asarray(start, dtype=float)
        p1 = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(p1 - p0))
        if distance <= 0.0:
            return 0.0
        count = max(1, int(np.ceil(distance / max(self.cell_size, 1e-9))))
        segment = distance / count
        blocked = 0.0
        for index in range(count):
            point = p0 + (index + 0.5) / count * (p1 - p0)
            row, col = self._index(point[0], point[1])
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                continue
            terrain = self.surface[row, col]
            terrain_block = np.isfinite(terrain) and point[2] <= terrain
            if terrain_block or self._vegetation_at(row, col):
                blocked += segment
        return float(min(distance, blocked))


def _read_raster(path: Path):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required to read DSM files") from exc
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
        cell_size = float(abs(source.transform.a))
    if values.ndim != 2 or values.size == 0:
        raise ValueError("DSM must contain one non-empty two-dimensional raster")
    return values, cell_size


def load_scene(dsm_path: Path, vegetation_mask_path: Optional[Path] = None) -> DSMScene:
    surface, cell_size = _read_raster(Path(dsm_path))
    expected = (int(round(MAP_SIZE_M[1] / CELL_SIZE_M)), int(round(MAP_SIZE_M[0] / CELL_SIZE_M)))
    if surface.shape != expected or not np.isclose(cell_size, CELL_SIZE_M):
        raise ValueError("DSM must be a 1000 by 1000 metre grid at 1 metre resolution")
    vegetation = None
    if vegetation_mask_path is not None and Path(vegetation_mask_path).is_file():
        vegetation, mask_cell = _read_raster(Path(vegetation_mask_path))
        if vegetation.shape != surface.shape or not np.isclose(mask_cell, cell_size):
            raise ValueError("DSM and vegetation mask grids must match")
    return DSMScene(surface=surface, cell_size=cell_size, vegetation=vegetation)
