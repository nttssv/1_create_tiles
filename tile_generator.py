"""OpenSlide-compatible WSI tile generation.

This module only generates candidate image tiles. It deliberately has no
dependency on downstream inference code such as YOLO, SAM, or CellSeg models.
Every saved tile is read from OpenSlide level 0 at the requested native pixel
size; thumbnail images are used only for tissue detection and tile filtering.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import json
import logging
import math
import multiprocessing as mp
import queue as queue_module
import shutil
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover - optional dependency guard
    ndi = None

try:
    import openslide
except ImportError:  # pragma: no cover - optional dependency guard
    openslide = None


LOGGER = logging.getLogger(__name__)
COORDINATE_FILENAME_RE = re.compile(
    r"^[A-Za-z0-9-]+_x(?P<x>-?\d+)_y(?P<y>-?\d+)(?:_c(?P<c>\d+))?(?:_[A-Za-z0-9-]+)?\.[^.]+$"
)
SUPPORTED_WSI_EXTENSIONS = {".tif", ".tiff"}


DEFAULT_CONFIG: dict[str, Any] = {
    "slide_path": None,
    "output_dir": "tile_output",
    "roi_mode": "auto",
    "manual_roi": None,
    "thumbnail": {
        "max_size": [4096, 4096],
        "downsample_factor": None,
    },
    "tissue_detection": {
        "method": "hsv",
        "hsv": {
            "min_saturation": 0.08,
            "min_value": 0.05,
            "max_value": 0.93,
        },
        "otsu": {
            "channel": "saturation",
            "min_saturation": 0.04,
            "max_value": 0.97,
        },
        "lab": {
            "max_l": 92.0,
            "min_chroma": 5.0,
        },
    },
    "mask_cleaning": {
        "min_object_area": 256,
        "fill_holes": True,
        "opening_radius": 2,
        "closing_radius": 4,
    },
    "connected_components": {
        "min_area": 1024,
        "connectivity": 8,
    },
    "tiling": {
        "tile_size": 512,
        "overlap": 0,
        "stride": None,
        "include_edge_tiles": False,
        "roi_padding": 0,
        "min_tissue_percentage": 70.0,
        "white_filter": {
            "enabled": True,
            "max_white_percentage": 95.0,
            "min_value": 0.90,
            "max_saturation": 0.12,
        },
    },
    "saving": {
        "accepted_tiles_subdir": "accepted_tiles",
        "rejected_tiles_subdir": "rejected_tiles",
        "save_rejected_tiles": True,
        "image_format": "png",
        "jpeg_quality": 95,
        "overwrite": False,
        "resume": True,
        "validate_existing_tiles": True,
        "atomic_writes": True,
        "filename_prefix": "tile",
        "include_component_id": False,
    },
    "metadata": {
        "filename": "tile_metadata.csv",
        "coordinates_filename": "tile_coordinates.csv",
        "summary_filename": "tile_summary.csv",
    },
    "visualization": {
        "filename": "patient_tile_map.png",
        "tissue_mask_filename": "tissue_mask.png",
        "dpi": 160,
        "max_tiles_to_draw": None,
        "draw_tile_labels": True,
        "draw_component_labels": True,
        "max_tile_labels": 300,
    },
    "statistics": {
        "filename": "tile_statistics.json",
    },
    "batch": {
        "input_dir": None,
        "output_dir": "output",
        "completed_marker": "_COMPLETED.json",
        "batch_summary_filename": "batch_summary.csv",
        "force": False,
        "workers": 1,
        "resume_partial": True,
        "progress_update_every_tiles": 100,
        "progress_update_interval_seconds": 2.0,
    },
    "logging": {
        "level": "INFO",
    },
}


class TileGenerationError(RuntimeError):
    """Raised when tile generation cannot proceed safely."""


@dataclass(frozen=True)
class Thumbnail:
    """Thumbnail and coordinate mapping metadata."""

    image: Image.Image
    level0_width: int
    level0_height: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def scale_x(self) -> float:
        return self.level0_width / self.width

    @property
    def scale_y(self) -> float:
        return self.level0_height / self.height


@dataclass(frozen=True)
class ROI:
    """Level-0 rectangular region of interest."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Component:
    """Connected tissue component."""

    component_id: int
    area_pixels: int
    bbox_thumbnail: tuple[int, int, int, int]
    bbox_level0: tuple[int, int, int, int]


@dataclass
class TileRecord:
    """Candidate or accepted tile metadata in level-0 coordinates."""

    tile_id: str
    component_id: int
    x: int
    y: int
    width: int
    height: int
    thumb_x: int = 0
    thumb_y: int = 0
    thumb_width: int = 0
    thumb_height: int = 0
    tissue_percentage: float = 0.0
    white_percentage: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""
    path: Path | None = None
    skipped_existing: bool = False


@dataclass
class GeneratedTile:
    """Tile payload returned by the public ``generate_tiles`` API."""

    tile_id: str
    image: Image.Image | None
    level0_coordinates: tuple[int, int, int, int]
    thumbnail_coordinates: tuple[int, int, int, int]
    component_id: int
    metadata: dict[str, Any]
    path: Path | None = None
    skipped_existing: bool = False


@dataclass
class BatchSlideResult:
    """One WSI row for batch summaries."""

    wsi_filename: str
    slide_name: str
    output_dir: Path
    accepted_tiles: int = 0
    rejected_tiles: int = 0
    total_tiles: int = 0
    tissue_area: int = 0
    processing_time_seconds: float = 0.0
    completion_timestamp: str = ""
    status: str = "Pending"
    error: str = ""


class ProgressPrinter:
    """Small terminal progress helper with no external dependency."""

    def __init__(self, label: str, total: int, width: int = 24, enabled: bool = True) -> None:
        self.label = label
        self.total = max(0, int(total))
        self.width = max(10, int(width))
        self.enabled = enabled
        self.start_time = time.perf_counter()
        self.current = 0

    def update(self, current: int, detail: str = "") -> None:
        self.current = max(0, min(int(current), self.total if self.total else int(current)))
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start_time
        eta = _estimate_eta(elapsed, self.current, self.total)
        filled = self.width if self.total == 0 else int(self.width * self.current / max(1, self.total))
        bar = "█" * filled + "-" * (self.width - filled)
        lines = [
            self.label,
            f"{bar} {self.current}/{self.total}",
        ]
        if detail:
            lines.append(detail)
        lines.append(f"Elapsed: {_format_duration(elapsed)}")
        if eta is not None:
            lines.append(f"ETA: {_format_duration(eta)}")
        print("\n".join(lines), flush=True)

    def finish(self, detail: str = "") -> None:
        self.update(self.total, detail=detail)


def coordinate_to_filename(
    x: int,
    y: int,
    extension: str = "png",
    component_id: int | None = None,
    prefix: str = "tile",
    include_component_id: bool = False,
    suffix: str | None = None,
) -> str:
    """Build the standard coordinate-based filename for a level-0 tile.

    Examples
    --------
    ``tile_x18432_y22528.png``
    ``tile_x18432_y22528_c1.png``
    ``tile_x18432_y22528_yolo.json``
    """

    ext = str(extension).strip().lstrip(".")
    if not ext:
        raise TileGenerationError("Filename extension cannot be empty.")
    safe_prefix = _safe_filename_token(prefix)
    filename = f"{safe_prefix}_x{int(x)}_y{int(y)}"
    if include_component_id:
        if component_id is None:
            raise TileGenerationError("component_id is required when include_component_id=True.")
        filename += f"_c{int(component_id)}"
    if suffix:
        filename += f"_{_safe_filename_token(suffix)}"
    return f"{filename}.{ext}"


def filename_to_coordinate(filename: str | Path) -> tuple[int, int]:
    """Parse a coordinate-based tile filename and return ``(level0_x, level0_y)``."""

    name = Path(filename).name
    match = COORDINATE_FILENAME_RE.match(name)
    if match is None:
        raise TileGenerationError(f"Filename does not follow coordinate naming: {name}")
    return int(match.group("x")), int(match.group("y"))


def coordinate_to_rectangle(
    x: int,
    y: int,
    width: int = 512,
    height: int = 512,
) -> tuple[int, int, int, int]:
    """Return a level-0 rectangle as ``(x, y, width, height)``."""

    return int(x), int(y), int(width), int(height)


def setup_logging(level: str | int = "INFO") -> None:
    """Configure module logging for command-line use."""

    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and merge it with defaults."""

    config = _deep_copy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    if yaml is None:
        raise TileGenerationError("PyYAML is required to load YAML config files.")

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    if not isinstance(user_config, Mapping):
        raise TileGenerationError(f"Config file must contain a mapping: {path}")
    return _deep_merge(config, dict(user_config))


def create_thumbnail(
    slide: Any,
    max_size: Sequence[int] = (4096, 4096),
    downsample_factor: float | None = None,
) -> Thumbnail:
    """Create an RGB thumbnail and remember how it maps to level 0.

    Parameters
    ----------
    slide:
        OpenSlide-like object with ``dimensions`` and ``get_thumbnail``.
    max_size:
        Maximum thumbnail size as ``(width, height)`` when no explicit
        downsample factor is provided.
    downsample_factor:
        Optional level-0 downsample factor. For example, 64 creates a
        thumbnail approximately ``level0_size / 64``.
    """

    level0_width, level0_height = _slide_dimensions(slide)
    if downsample_factor is not None:
        if downsample_factor <= 0:
            raise TileGenerationError("thumbnail.downsample_factor must be > 0.")
        thumb_size = (
            max(1, int(math.ceil(level0_width / downsample_factor))),
            max(1, int(math.ceil(level0_height / downsample_factor))),
        )
    else:
        thumb_size = _as_size_tuple(max_size, "thumbnail.max_size")

    LOGGER.info("Creating thumbnail with max size %s.", thumb_size)
    image = slide.get_thumbnail(thumb_size).convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise TileGenerationError("OpenSlide returned an empty thumbnail.")
    LOGGER.info(
        "Thumbnail size is %sx%s for level-0 slide size %sx%s.",
        image.width,
        image.height,
        level0_width,
        level0_height,
    )
    return Thumbnail(image=image, level0_width=level0_width, level0_height=level0_height)


def create_tissue_mask(
    thumbnail: Image.Image,
    config: Mapping[str, Any] | None = None,
    method: str | None = None,
) -> np.ndarray:
    """Create a binary tissue mask from a thumbnail.

    Supported methods are ``hsv``, ``otsu``, ``lab``, and ``auto``. HSV is the
    default because saturation plus brightness thresholds are robust for common
    H&E slides with white background while remaining simple to tune.
    """

    cfg = dict(config or DEFAULT_CONFIG["tissue_detection"])
    selected_method = (method or cfg.get("method", "hsv")).lower()
    rgb = _image_to_rgb_array(thumbnail)

    if selected_method == "hsv":
        mask = _hsv_tissue_mask(rgb, cfg.get("hsv", {}))
    elif selected_method == "otsu":
        mask = _otsu_tissue_mask(rgb, cfg.get("otsu", {}))
    elif selected_method == "lab":
        mask = _lab_tissue_mask(rgb, cfg.get("lab", {}))
    elif selected_method == "auto":
        hsv_mask = _hsv_tissue_mask(rgb, cfg.get("hsv", {}))
        lab_mask = _lab_tissue_mask(rgb, cfg.get("lab", {}))
        otsu_mask = _otsu_tissue_mask(rgb, cfg.get("otsu", {}))
        mask = (hsv_mask & otsu_mask) | (lab_mask & otsu_mask) | (hsv_mask & lab_mask)
    else:
        raise TileGenerationError(
            f"Unsupported tissue detection method '{selected_method}'. "
            "Use hsv, otsu, lab, or auto."
        )

    LOGGER.info(
        "Created %s tissue mask with %.2f%% foreground.",
        selected_method,
        100.0 * float(mask.mean()),
    )
    return mask.astype(bool, copy=False)


def clean_mask(mask: np.ndarray, config: Mapping[str, Any] | None = None) -> np.ndarray:
    """Clean a binary mask with object removal, hole filling, opening, closing."""

    _require_scipy()
    cfg = dict(DEFAULT_CONFIG["mask_cleaning"])
    if config:
        cfg.update(config)

    cleaned = np.asarray(mask, dtype=bool)
    min_object_area = int(cfg.get("min_object_area", 0) or 0)
    if min_object_area > 0:
        cleaned = _remove_small_objects(cleaned, min_object_area)

    if bool(cfg.get("fill_holes", True)):
        cleaned = ndi.binary_fill_holes(cleaned)

    opening_radius = int(cfg.get("opening_radius", 0) or 0)
    if opening_radius > 0:
        cleaned = ndi.binary_opening(cleaned, structure=_disk_structure(opening_radius))

    closing_radius = int(cfg.get("closing_radius", 0) or 0)
    if closing_radius > 0:
        cleaned = ndi.binary_closing(cleaned, structure=_disk_structure(closing_radius))

    LOGGER.info("Cleaned tissue mask has %.2f%% foreground.", 100.0 * float(cleaned.mean()))
    return cleaned.astype(bool, copy=False)


def extract_connected_components(
    mask: np.ndarray,
    thumbnail: Thumbnail,
    min_area: int = 1024,
    connectivity: int = 8,
) -> tuple[np.ndarray, list[Component]]:
    """Label connected tissue components and keep components above ``min_area``."""

    _require_scipy()
    if connectivity not in (4, 8):
        raise TileGenerationError("connected_components.connectivity must be 4 or 8.")
    structure = ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    raw_labels, count = ndi.label(np.asarray(mask, dtype=bool), structure=structure)
    if count == 0:
        LOGGER.warning("No tissue components found.")
        return np.zeros_like(mask, dtype=np.int32), []

    areas = np.bincount(raw_labels.ravel())
    objects = ndi.find_objects(raw_labels)
    raw_components: list[tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []

    for raw_id in range(1, count + 1):
        area = int(areas[raw_id])
        if area < min_area:
            continue
        obj = objects[raw_id - 1]
        if obj is None:
            continue
        y_slice, x_slice = obj
        bbox_thumbnail = (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)
        bbox_level0 = _thumbnail_box_to_level0(thumbnail, bbox_thumbnail)
        raw_components.append((raw_id, area, bbox_thumbnail, bbox_level0))

    raw_components.sort(key=lambda item: (item[2][1], item[2][0], item[2][3], item[2][2]))
    id_mapping = np.zeros(count + 1, dtype=np.int32)
    components: list[Component] = []
    for component_id, (raw_id, area, bbox_thumbnail, bbox_level0) in enumerate(
        raw_components,
        start=1,
    ):
        id_mapping[raw_id] = component_id
        components.append(
            Component(
                component_id=component_id,
                area_pixels=area,
                bbox_thumbnail=bbox_thumbnail,
                bbox_level0=bbox_level0,
            )
        )

    labels = id_mapping[raw_labels]
    LOGGER.info(
        "Kept %s connected tissue components from %s raw components.",
        len(components),
        count,
    )
    return labels.astype(np.int32, copy=False), components


def manual_roi(
    slide_dimensions: Sequence[int],
    roi: Mapping[str, int],
    padding: int = 0,
) -> ROI:
    """Normalize, clip, and print a manual level-0 ROI."""

    if roi is None:
        raise TileGenerationError("manual_roi() requires a ROI mapping.")
    slide_width, slide_height = _as_size_tuple(slide_dimensions, "slide_dimensions")

    try:
        x1 = int(roi["x1"])
        y1 = int(roi["y1"])
        x2 = int(roi["x2"])
        y2 = int(roi["y2"])
    except KeyError as exc:
        raise TileGenerationError(f"MANUAL_ROI is missing key: {exc.args[0]}") from exc

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    normalized = _clip_roi(ROI(x1=x1, y1=y1, x2=x2, y2=y2), (slide_width, slide_height))
    if padding:
        normalized = pad_roi(normalized, (slide_width, slide_height), padding)
    if normalized.width <= 0 or normalized.height <= 0:
        raise TileGenerationError(
            "Manual ROI is empty after coordinate sorting and slide-boundary clipping."
        )

    print("ROI")
    print(f"x1: {normalized.x1}")
    print(f"y1: {normalized.y1}")
    print(f"x2: {normalized.x2}")
    print(f"y2: {normalized.y2}")
    print(f"width: {normalized.width}")
    print(f"height: {normalized.height}")
    LOGGER.info("Using manual ROI: %s.", normalized)
    return normalized


def pad_roi(roi: ROI, slide_dimensions: Sequence[int], padding: int) -> ROI:
    """Expand a level-0 ROI by ``padding`` pixels and clip to slide bounds."""

    if padding < 0:
        raise TileGenerationError("tiling.roi_padding cannot be negative.")
    slide_width, slide_height = _as_size_tuple(slide_dimensions, "slide_dimensions")
    padded = ROI(
        x1=roi.x1 - int(padding),
        y1=roi.y1 - int(padding),
        x2=roi.x2 + int(padding),
        y2=roi.y2 + int(padding),
    )
    return _clip_roi(padded, (slide_width, slide_height))


def interactive_roi(slide: Any, thumbnail: Thumbnail, padding: int = 0) -> ROI:
    """Select a manual ROI interactively on the thumbnail.

    Matplotlib's toolbar provides zoom and pan. The two clicked thumbnail
    corners are converted back to level-0 coordinates, displayed, and confirmed
    before tile generation proceeds.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise TileGenerationError("matplotlib is required for interactive ROI selection.") from exc

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(thumbnail.image)
    ax.set_title("Zoom/pan, then click first and second ROI corners")
    ax.set_axis_off()
    plt.tight_layout()
    LOGGER.info("Waiting for two ROI corner clicks in the interactive viewer.")
    points = plt.ginput(2, timeout=0)
    if len(points) != 2:
        plt.close(fig)
        raise TileGenerationError("Interactive ROI selection requires exactly two clicks.")

    (tx1, ty1), (tx2, ty2) = points
    roi = manual_roi(
        _slide_dimensions(slide),
        {
            "x1": round(tx1 * thumbnail.scale_x),
            "y1": round(ty1 * thumbnail.scale_y),
            "x2": round(tx2 * thumbnail.scale_x),
            "y2": round(ty2 * thumbnail.scale_y),
        },
        padding=padding,
    )

    rect_x = roi.x1 / thumbnail.scale_x
    rect_y = roi.y1 / thumbnail.scale_y
    rect_w = roi.width / thumbnail.scale_x
    rect_h = roi.height / thumbnail.scale_y
    ax.add_patch(
        Rectangle(
            (rect_x, rect_y),
            rect_w,
            rect_h,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
        )
    )
    fig.canvas.draw_idle()
    plt.show(block=False)

    response = input("Generate tiles for this ROI? [y/N]: ").strip().lower()
    plt.close(fig)
    if response not in {"y", "yes"}:
        raise TileGenerationError("Interactive ROI selection was not confirmed.")
    return roi


def generate_candidate_tiles(
    slide_dimensions: Sequence[int],
    thumbnail: Thumbnail,
    label_image: np.ndarray,
    components: Sequence[Component],
    tile_size: int = 512,
    stride: int | None = None,
    overlap: int | None = 0,
    roi_mode: str = "auto",
    roi: ROI | None = None,
    include_edge_tiles: bool = True,
) -> list[TileRecord]:
    """Generate candidate tile coordinates in level-0 space.

    The tile grid is anchored to the ROI origin. In automatic mode the ROI is
    the full slide unless an explicit ROI is passed. Component bounding boxes
    are used only to limit which aligned grid cells are evaluated.
    """

    slide_width, slide_height = _as_size_tuple(slide_dimensions, "slide_dimensions")
    stride = _resolve_stride(tile_size=tile_size, overlap=overlap, stride=stride)
    _validate_tile_grid(tile_size, stride)
    label_image = _validate_label_image(label_image, thumbnail)
    mode = roi_mode.lower()
    if mode not in {"auto", "manual"}:
        raise TileGenerationError("ROI_MODE must be 'auto' or 'manual'.")

    if include_edge_tiles:
        LOGGER.debug(
            "include_edge_tiles is ignored for grid alignment; all tiles remain "
            "anchored to the ROI origin."
        )

    if roi is None:
        if mode == "manual":
            raise TileGenerationError("Manual ROI mode requires an ROI.")
        grid_roi = ROI(0, 0, slide_width, slide_height)
    else:
        grid_roi = _clip_roi(roi, (slide_width, slide_height))
    if grid_roi.width < tile_size or grid_roi.height < tile_size:
        LOGGER.warning("ROI is smaller than one tile; no candidates generated.")
        return []

    windows: list[tuple[int, int, int, int, int]] = []
    for component in components:
        x1, y1, x2, y2 = component.bbox_level0
        intersection = _intersect_boxes((x1, y1, x2, y2), (grid_roi.x1, grid_roi.y1, grid_roi.x2, grid_roi.y2))
        if intersection is None:
            continue
        windows.append((component.component_id, *intersection))

    best_by_xy: dict[tuple[int, int], TileRecord] = {}
    for window_component_id, x1, y1, x2, y2 in windows:
        x_positions = _aligned_grid_positions_for_window(
            origin=grid_roi.x1,
            roi_stop=grid_roi.x2,
            window_start=x1,
            window_stop=x2,
            tile_size=tile_size,
            stride=stride,
        )
        y_positions = _aligned_grid_positions_for_window(
            origin=grid_roi.y1,
            roi_stop=grid_roi.y2,
            window_start=y1,
            window_stop=y2,
            tile_size=tile_size,
            stride=stride,
        )
        for y in y_positions:
            for x in x_positions:
                if x < 0 or y < 0 or x + tile_size > slide_width or y + tile_size > slide_height:
                    continue
                tissue_pct, dominant_component = _tile_tissue_percentage(
                    label_image=label_image,
                    thumbnail=thumbnail,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                )
                if dominant_component == 0:
                    continue
                if mode == "auto" and window_component_id not in {0, dominant_component}:
                    # Overlapping component bounding boxes can produce duplicate
                    # grid points. Keep the tile only for its dominant component.
                    continue
                tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(
                    thumbnail,
                    x,
                    y,
                    tile_size,
                    tile_size,
                )
                component_id = dominant_component or window_component_id
                record = TileRecord(
                    tile_id=_coordinate_tile_id(x, y),
                    component_id=component_id,
                    x=x,
                    y=y,
                    width=tile_size,
                    height=tile_size,
                    thumb_x=tx1,
                    thumb_y=ty1,
                    thumb_width=tx2 - tx1,
                    thumb_height=ty2 - ty1,
                    tissue_percentage=tissue_pct,
                    accepted=False,
                )
                key = (x, y)
                previous = best_by_xy.get(key)
                if previous is None or record.tissue_percentage > previous.tissue_percentage:
                    best_by_xy[key] = record

    candidates = sorted(best_by_xy.values(), key=lambda tile: (tile.y, tile.x))
    LOGGER.info("Generated %s candidate tiles.", len(candidates))
    return candidates


def filter_tiles(
    candidates: Iterable[TileRecord],
    label_image: np.ndarray,
    thumbnail: Thumbnail,
    min_tissue_percentage: float = 70.0,
    white_filter: Mapping[str, Any] | None = None,
) -> tuple[list[TileRecord], list[TileRecord]]:
    """Accept or reject candidate tiles using tissue and white-background filters."""

    label_image = _validate_label_image(label_image, thumbnail)
    tissue_threshold = _normalize_percentage(min_tissue_percentage, "minimum tissue")
    white_cfg = dict(DEFAULT_CONFIG["tiling"]["white_filter"])
    if white_filter:
        white_cfg.update(white_filter)
    white_enabled = bool(white_cfg.get("enabled", True))
    max_white_percentage = _normalize_percentage(
        white_cfg.get("max_white_percentage", 95.0),
        "maximum white",
    )
    thumbnail_rgb = _image_to_rgb_array(thumbnail.image)
    accepted: list[TileRecord] = []
    rejected: list[TileRecord] = []

    for candidate in candidates:
        tissue_pct, dominant_component = _tile_tissue_percentage(
            label_image=label_image,
            thumbnail=thumbnail,
            x=candidate.x,
            y=candidate.y,
            width=candidate.width,
            height=candidate.height,
        )
        tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(
            thumbnail,
            candidate.x,
            candidate.y,
            candidate.width,
            candidate.height,
        )
        white_pct = _tile_white_percentage(
            thumbnail=thumbnail,
            thumbnail_rgb=thumbnail_rgb,
            x=candidate.x,
            y=candidate.y,
            width=candidate.width,
            height=candidate.height,
            min_value=float(white_cfg.get("min_value", 0.90)),
            max_saturation=float(white_cfg.get("max_saturation", 0.12)),
        )
        rejection_reasons: list[str] = []
        if tissue_pct < tissue_threshold:
            rejection_reasons.append("low_tissue")
        if white_enabled and white_pct > max_white_percentage:
            rejection_reasons.append("high_white")

        record = TileRecord(
            tile_id=_coordinate_tile_id(candidate.x, candidate.y),
            component_id=dominant_component or candidate.component_id,
            x=candidate.x,
            y=candidate.y,
            width=candidate.width,
            height=candidate.height,
            thumb_x=tx1,
            thumb_y=ty1,
            thumb_width=tx2 - tx1,
            thumb_height=ty2 - ty1,
            tissue_percentage=tissue_pct,
            white_percentage=white_pct,
            accepted=not rejection_reasons,
            rejection_reason=";".join(rejection_reasons),
        )
        if record.accepted:
            accepted.append(record)
        else:
            rejected.append(record)

    LOGGER.info(
        "Accepted %s tiles and rejected %s tiles at %.2f%% minimum tissue and %.2f%% maximum white.",
        len(accepted),
        len(rejected),
        tissue_threshold,
        max_white_percentage,
    )
    return accepted, rejected


def save_tiles(
    slide: Any,
    tiles: Sequence[TileRecord],
    output_dir: str | Path,
    image_format: str = "png",
    jpeg_quality: int = 95,
    overwrite: bool = True,
    resume: bool = False,
    validate_existing_tiles: bool = True,
    atomic_writes: bool = True,
    filename_prefix: str = "tile",
    include_component_id: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    progress_label: str = "tiles",
) -> list[Path]:
    """Crop and save accepted native-resolution level-0 tiles."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ext = _image_extension(image_format)
    saved_paths: list[Path] = []

    total = len(tiles)
    for index, tile in enumerate(tiles, start=1):
        filename = coordinate_to_filename(
            tile.x,
            tile.y,
            extension=ext,
            component_id=tile.component_id,
            prefix=filename_prefix,
            include_component_id=include_component_id,
        )
        path = output_path / filename
        tile.path = path
        tile.skipped_existing = False
        existed_before = path.exists()
        save_action = "saved"
        if existed_before and resume:
            if not validate_existing_tiles or _is_existing_tile_valid(path, (tile.width, tile.height)):
                tile.skipped_existing = True
                saved_paths.append(path)
                if progress_callback:
                    progress_callback(
                        index,
                        total,
                        _save_progress_label(progress_label, "skipped", filename),
                    )
                continue
            LOGGER.warning("Existing tile is invalid and will be regenerated: %s", path)
            save_action = "regenerated"
        elif existed_before:
            save_action = "overwritten"
        if path.exists() and not overwrite:
            if not resume:
                raise TileGenerationError(f"Refusing to overwrite existing tile: {path}")
        image = slide.read_region((tile.x, tile.y), 0, (tile.width, tile.height)).convert("RGB")
        if image.size != (tile.width, tile.height):
            raise TileGenerationError(
                f"Unexpected tile size {image.size} for {tile.tile_id}; "
                f"expected {(tile.width, tile.height)}."
            )
        save_kwargs: dict[str, Any] = {}
        if ext in {"jpg", "jpeg"}:
            save_kwargs["quality"] = int(jpeg_quality)
            save_kwargs["subsampling"] = 0
        _save_image_atomic(image, path, atomic_writes=atomic_writes, save_kwargs=save_kwargs)
        saved_paths.append(path)
        if progress_callback:
            progress_callback(index, total, _save_progress_label(progress_label, save_action, filename))

    skipped = sum(1 for tile in tiles if tile.skipped_existing)
    LOGGER.info(
        "Saved %s native level-0 tiles to %s (%s skipped by resume mode).",
        len(saved_paths) - skipped,
        output_path,
        skipped,
    )
    return saved_paths


def visualize_tiles(
    thumbnail: Thumbnail,
    tissue_mask: np.ndarray,
    label_image: np.ndarray,
    components: Sequence[Component],
    accepted_tiles: Sequence[TileRecord],
    rejected_tiles: Sequence[TileRecord],
    output_path: str | Path = "patient_tile_map.png",
    roi: ROI | None = None,
    roi_mode: str = "auto",
    max_tiles_to_draw: int | None = None,
    dpi: int = 160,
    draw_tile_labels: bool = True,
    draw_component_labels: bool = True,
    max_tile_labels: int | None = 300,
) -> Path:
    """Save a tile-generation overview figure."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise TileGenerationError("matplotlib is required for visualization.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted_to_draw = list(_limit_tiles(accepted_tiles, max_tiles_to_draw))
    rejected_to_draw = list(_limit_tiles(rejected_tiles, max_tiles_to_draw))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    ax_thumb, ax_mask, ax_components, ax_tiles = axes.ravel()

    ax_thumb.imshow(thumbnail.image)
    ax_thumb.set_title("Thumbnail")
    _draw_roi_and_components(
        ax_thumb,
        thumbnail,
        components,
        roi,
        roi_mode,
        Rectangle,
        draw_component_labels=draw_component_labels,
    )

    ax_mask.imshow(tissue_mask, cmap="gray")
    ax_mask.set_title("Tissue mask")

    masked_labels = np.ma.masked_equal(label_image, 0)
    ax_components.imshow(thumbnail.image, alpha=0.35)
    ax_components.imshow(masked_labels, cmap="tab20", alpha=0.75, interpolation="nearest")
    ax_components.set_title("Connected components")
    _draw_roi_and_components(
        ax_components,
        thumbnail,
        components,
        roi,
        roi_mode,
        Rectangle,
        draw_component_labels=draw_component_labels,
    )

    ax_tiles.imshow(thumbnail.image)
    ax_tiles.set_title("Accepted (green) / rejected (red)")
    label_budget = _tile_label_budget(
        len(accepted_to_draw) + len(rejected_to_draw),
        draw_tile_labels,
        max_tile_labels,
    )
    _draw_tile_boxes(
        ax_tiles,
        thumbnail,
        rejected_to_draw,
        "red",
        Rectangle,
        label_budget=label_budget,
    )
    _draw_tile_boxes(
        ax_tiles,
        thumbnail,
        accepted_to_draw,
        "lime",
        Rectangle,
        label_budget=label_budget,
    )
    _draw_roi_and_components(
        ax_tiles,
        thumbnail,
        components,
        roi,
        roi_mode,
        Rectangle,
        draw_component_labels=draw_component_labels,
    )

    for ax in axes.ravel():
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved tile overview visualization to %s.", output_path)
    return output_path


def save_tissue_mask_image(mask: np.ndarray, output_path: str | Path) -> Path:
    """Save a binary tissue mask diagnostic image."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L")
    image.save(path)
    LOGGER.info("Saved tissue mask visualization to %s.", path)
    return path


def export_tile_metadata(tiles: Sequence[TileRecord], csv_path: str | Path) -> Path:
    """Write accepted tile metadata to CSV."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_filename",
                "tile_id",
                "component_id",
                "level0_x",
                "level0_y",
                "tile_width",
                "tile_height",
                "thumbnail_x",
                "thumbnail_y",
                "tissue_percentage",
                "white_percentage",
                "tile_status",
                "rejection_reason",
                "path",
                "skipped_existing",
            ],
        )
        writer.writeheader()
        for tile in tiles:
            writer.writerow(
                {
                    "tile_filename": tile.path.name if tile.path else coordinate_to_filename(tile.x, tile.y),
                    "tile_id": tile.tile_id,
                    "component_id": tile.component_id,
                    "level0_x": tile.x,
                    "level0_y": tile.y,
                    "tile_width": tile.width,
                    "tile_height": tile.height,
                    "thumbnail_x": tile.thumb_x,
                    "thumbnail_y": tile.thumb_y,
                    "tissue_percentage": f"{tile.tissue_percentage:.4f}",
                    "white_percentage": f"{tile.white_percentage:.4f}",
                    "tile_status": "accepted" if tile.accepted else "rejected",
                    "rejection_reason": tile.rejection_reason,
                    "path": str(tile.path) if tile.path else "",
                    "skipped_existing": tile.skipped_existing,
                }
            )
    LOGGER.info("Exported metadata for %s tiles to %s.", len(tiles), path)
    return path


def export_tile_coordinates(tiles: Sequence[TileRecord], csv_path: str | Path) -> Path:
    """Write a compact coordinate lookup CSV for downstream WSI overlays."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "tile_filename",
                "component_id",
                "level0_x",
                "level0_y",
                "tile_width",
                "tile_height",
                "tile_status",
            ],
        )
        writer.writeheader()
        for tile in tiles:
            writer.writerow(
                {
                    "tile_id": tile.tile_id,
                    "tile_filename": tile.path.name if tile.path else coordinate_to_filename(tile.x, tile.y),
                    "component_id": tile.component_id,
                    "level0_x": tile.x,
                    "level0_y": tile.y,
                    "tile_width": tile.width,
                    "tile_height": tile.height,
                    "tile_status": "accepted" if tile.accepted else "rejected",
                }
            )
    LOGGER.info("Exported coordinate lookup for %s tiles to %s.", len(tiles), path)
    return path


def export_tile_summary(statistics: Mapping[str, Any], csv_path: str | Path) -> Path:
    """Write a one-row per-slide summary CSV."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "slide_name",
        "ROI_width",
        "ROI_height",
        "num_components",
        "candidate_tiles",
        "accepted_tiles",
        "rejected_tiles",
        "tissue_area",
        "average_tissue_percent",
        "average_white_percent",
        "processing_time_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: statistics.get(key, "") for key in fieldnames})
    LOGGER.info("Exported tile summary to %s.", path)
    return path


def export_tile_statistics(statistics: Mapping[str, Any], json_path: str | Path) -> Path:
    """Write run-level tile-generation statistics to JSON."""

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=False)
        handle.write("\n")
    LOGGER.info("Exported tile statistics to %s.", path)
    return path


def generate_tiles(
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    slide: Any | None = None,
    return_images: bool = True,
    save_to_disk: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Generate WSI tiles through the standalone public API.

    Returns a dictionary whose ``tiles`` entry contains ``GeneratedTile``
    objects. Set ``return_images=False`` for batch jobs that only need files on
    disk; set ``save_to_disk=False`` when downstream inference should consume
    the returned PIL images directly.
    """

    start_time = time.perf_counter()

    cfg = load_config(config_path)
    if config:
        cfg = _deep_merge(cfg, dict(config))
    setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    slide_obj = slide
    close_slide = False
    if slide_obj is None:
        slide_path = cfg.get("slide_path")
        if not slide_path:
            raise TileGenerationError("Set slide_path in the config or pass a slide object.")
        slide_obj, close_slide = _open_slide(slide_path)

    try:
        slide_dimensions = _slide_dimensions(slide_obj)
        thumbnail_cfg = cfg.get("thumbnail", {})
        thumb = create_thumbnail(
            slide_obj,
            max_size=thumbnail_cfg.get("max_size", DEFAULT_CONFIG["thumbnail"]["max_size"]),
            downsample_factor=thumbnail_cfg.get("downsample_factor"),
        )
        tissue_mask = create_tissue_mask(thumb.image, cfg.get("tissue_detection", {}))
        cleaned_mask = clean_mask(tissue_mask, cfg.get("mask_cleaning", {}))

        component_cfg = cfg.get("connected_components", {})
        label_image, components = extract_connected_components(
            cleaned_mask,
            thumb,
            min_area=int(component_cfg.get("min_area", 1024)),
            connectivity=int(component_cfg.get("connectivity", 8)),
        )

        mode = str(cfg.get("roi_mode", "auto")).lower()
        roi: ROI | None = None
        tiling_cfg = cfg.get("tiling", {})
        roi_padding = int(tiling_cfg.get("roi_padding", 0) or 0)
        if mode == "manual":
            roi_config = cfg.get("manual_roi")
            if roi_config is None:
                roi = interactive_roi(slide_obj, thumb, padding=roi_padding)
            else:
                roi = manual_roi(slide_dimensions, roi_config, padding=roi_padding)
        elif mode == "auto":
            roi = ROI(0, 0, slide_dimensions[0], slide_dimensions[1])
        else:
            raise TileGenerationError("ROI_MODE must be 'auto' or 'manual'.")

        candidates = generate_candidate_tiles(
            slide_dimensions=slide_dimensions,
            thumbnail=thumb,
            label_image=label_image,
            components=components,
            tile_size=int(tiling_cfg.get("tile_size", 512)),
            stride=tiling_cfg.get("stride"),
            overlap=tiling_cfg.get("overlap", 0),
            roi_mode=mode,
            roi=roi,
            include_edge_tiles=bool(tiling_cfg.get("include_edge_tiles", True)),
        )
        accepted, rejected = filter_tiles(
            candidates,
            label_image=label_image,
            thumbnail=thumb,
            min_tissue_percentage=float(tiling_cfg.get("min_tissue_percentage", 70.0)),
            white_filter=tiling_cfg.get("white_filter", {}),
        )

        output_dir = Path(cfg.get("output_dir", "tile_output"))
        saving_cfg = cfg.get("saving", {})
        accepted_tiles_dir = output_dir / saving_cfg.get("accepted_tiles_subdir", "accepted_tiles")
        rejected_tiles_dir = output_dir / saving_cfg.get("rejected_tiles_subdir", "rejected_tiles")
        image_format = saving_cfg.get("image_format", "png")
        filename_prefix = saving_cfg.get("filename_prefix", "tile")
        include_component_id = bool(saving_cfg.get("include_component_id", False))
        save_rejected_tiles = bool(saving_cfg.get("save_rejected_tiles", True))
        saved_tiles: list[Path] = []
        saved_rejected_tiles: list[Path] = []
        progress_total = len(accepted) + (len(rejected) if save_rejected_tiles else 0)
        progress_offsets = {"accepted tiles": 0, "rejected tiles": len(accepted)}

        def _combined_progress(current: int, total: int, label: str) -> None:
            if progress_callback is None:
                return
            stage, action, filename = _parse_save_progress_label(label)
            offset = progress_offsets.get(stage, 0)
            if stage == "accepted tiles":
                accepted_done = current
                rejected_done = 0
            elif stage == "rejected tiles":
                accepted_done = len(accepted)
                rejected_done = current
            else:
                accepted_done = min(offset + current, len(accepted))
                rejected_done = max(0, offset + current - len(accepted))
            progress_callback(
                offset + current,
                progress_total,
                _format_tile_progress_detail(
                    stage=stage,
                    action=action,
                    filename=filename,
                    accepted_done=accepted_done,
                    accepted_total=len(accepted),
                    rejected_done=rejected_done,
                    rejected_total=len(rejected) if save_rejected_tiles else 0,
                ),
            )

        if save_to_disk:
            saved_tiles = save_tiles(
                slide_obj,
                accepted,
                accepted_tiles_dir,
                image_format=image_format,
                jpeg_quality=int(saving_cfg.get("jpeg_quality", 95)),
                overwrite=bool(saving_cfg.get("overwrite", False)),
                resume=bool(saving_cfg.get("resume", True)),
                validate_existing_tiles=bool(saving_cfg.get("validate_existing_tiles", True)),
                atomic_writes=bool(saving_cfg.get("atomic_writes", True)),
                filename_prefix=filename_prefix,
                include_component_id=include_component_id,
                progress_callback=_combined_progress,
                progress_label="accepted tiles",
            )
            if save_rejected_tiles:
                saved_rejected_tiles = save_tiles(
                    slide_obj,
                    rejected,
                    rejected_tiles_dir,
                    image_format=image_format,
                    jpeg_quality=int(saving_cfg.get("jpeg_quality", 95)),
                    overwrite=bool(saving_cfg.get("overwrite", False)),
                    resume=bool(saving_cfg.get("resume", True)),
                    validate_existing_tiles=bool(saving_cfg.get("validate_existing_tiles", True)),
                    atomic_writes=bool(saving_cfg.get("atomic_writes", True)),
                    filename_prefix=filename_prefix,
                    include_component_id=include_component_id,
                    progress_callback=_combined_progress,
                    progress_label="rejected tiles",
                )
            else:
                _assign_tile_paths(
                    rejected,
                    rejected_tiles_dir,
                    image_format=image_format,
                    filename_prefix=filename_prefix,
                    include_component_id=include_component_id,
                )
        else:
            _assign_tile_paths(
                accepted,
                accepted_tiles_dir,
                image_format=image_format,
                filename_prefix=filename_prefix,
                include_component_id=include_component_id,
            )
            _assign_tile_paths(
                rejected,
                rejected_tiles_dir,
                image_format=image_format,
                filename_prefix=filename_prefix,
                include_component_id=include_component_id,
            )

        metadata_cfg = cfg.get("metadata", {})
        metadata_path = export_tile_metadata(
            accepted + rejected,
            output_dir / metadata_cfg.get("filename", "tile_metadata.csv"),
        )
        coordinates_path = export_tile_coordinates(
            accepted + rejected,
            output_dir / metadata_cfg.get("coordinates_filename", "tile_coordinates.csv"),
        )

        visualization_cfg = cfg.get("visualization", {})
        visualization_path = visualize_tiles(
            thumbnail=thumb,
            tissue_mask=cleaned_mask,
            label_image=label_image,
            components=components,
            accepted_tiles=accepted,
            rejected_tiles=rejected,
            output_path=output_dir / visualization_cfg.get("filename", "patient_tile_map.png"),
            roi=roi,
            roi_mode=mode,
            max_tiles_to_draw=visualization_cfg.get("max_tiles_to_draw"),
            dpi=int(visualization_cfg.get("dpi", 160)),
            draw_tile_labels=bool(visualization_cfg.get("draw_tile_labels", True)),
            draw_component_labels=bool(visualization_cfg.get("draw_component_labels", True)),
            max_tile_labels=visualization_cfg.get("max_tile_labels", 300),
        )
        tissue_mask_path = save_tissue_mask_image(
            cleaned_mask,
            output_dir / visualization_cfg.get("tissue_mask_filename", "tissue_mask.png"),
        )

        processing_time = time.perf_counter() - start_time
        statistics_cfg = cfg.get("statistics", {})
        statistics = _build_statistics(
            slide_name=_slide_name(cfg, slide_obj),
            roi=roi,
            components=components,
            candidates=candidates,
            accepted=accepted,
            rejected=rejected,
            processing_time_seconds=processing_time,
        )
        statistics_path = export_tile_statistics(
            statistics,
            output_dir / statistics_cfg.get("filename", "tile_statistics.json"),
        )
        summary_path = export_tile_summary(
            statistics,
            output_dir / metadata_cfg.get("summary_filename", "tile_summary.csv"),
        )

        generated_tiles = _build_generated_tiles(slide_obj, accepted, return_images=return_images)

        LOGGER.info(
            "Done. Accepted %s tiles, rejected %s tiles in %.2f seconds.",
            len(accepted),
            len(rejected),
            processing_time,
        )

        return {
            "tiles": generated_tiles,
            "thumbnail": thumb,
            "tissue_mask": cleaned_mask,
            "label_image": label_image,
            "components": components,
            "roi": roi,
            "candidates": candidates,
            "accepted_tiles": accepted,
            "rejected_tiles": rejected,
            "saved_tile_paths": saved_tiles,
            "saved_rejected_tile_paths": saved_rejected_tiles,
            "metadata_path": metadata_path,
            "coordinates_path": coordinates_path,
            "visualization_path": visualization_path,
            "tissue_mask_path": tissue_mask_path,
            "statistics": statistics,
            "statistics_path": statistics_path,
            "summary_path": summary_path,
        }
    finally:
        if close_slide and hasattr(slide_obj, "close"):
            slide_obj.close()


def run_tile_generation(
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    slide: Any | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper around ``generate_tiles`` for file workflows."""

    return generate_tiles(
        config_path=config_path,
        config=config,
        slide=slide,
        return_images=False,
        save_to_disk=True,
    )


def process_wsi_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
    return_images: bool = False,
    slide_factory: Callable[[Path], Any] | None = None,
    show_progress: bool = True,
    workers: int | None = None,
) -> list[BatchSlideResult]:
    """Process every supported WSI in ``input_dir`` into per-slide output folders."""

    base_config = load_config(config_path)
    if config:
        base_config = _deep_merge(base_config, dict(config))
    setup_logging(base_config.get("logging", {}).get("level", "INFO"))

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    wsi_files = discover_wsi_files(input_path)
    if not wsi_files:
        raise TileGenerationError(f"No supported .tif/.tiff WSI files found in {input_path}")

    batch_cfg = base_config.get("batch", {})
    marker_name = str(batch_cfg.get("completed_marker", "_COMPLETED.json"))
    effective_force = bool(force or batch_cfg.get("force", False))
    worker_count = max(1, int(workers or batch_cfg.get("workers", 1) or 1))
    resume_partial = bool(batch_cfg.get("resume_partial", True))
    progress_update_every_tiles = max(1, int(batch_cfg.get("progress_update_every_tiles", 100) or 1))
    progress_update_interval_seconds = max(
        0.0,
        float(batch_cfg.get("progress_update_interval_seconds", 2.0) or 0.0),
    )
    if slide_factory is not None and worker_count > 1:
        LOGGER.warning("slide_factory was provided; falling back to one worker for safe testing.")
        worker_count = 1
    results: list[BatchSlideResult] = []
    overall = ProgressPrinter("Processing WSIs", len(wsi_files), enabled=show_progress)
    batch_start = time.perf_counter()

    if worker_count == 1:
        for index, slide_path in enumerate(wsi_files, start=1):
            overall.update(index - 1, detail=f"Current:\n{slide_path.name}")
            result = _process_batch_slide_item(
                index=index,
                slide_path=slide_path,
                output_path=output_path,
                base_config=base_config,
                marker_name=marker_name,
                force=effective_force,
                resume_partial=resume_partial,
                return_images=return_images,
                slide_factory=slide_factory,
                show_tile_progress=show_progress,
                print_summary=False,
                prefix_logs=False,
                progress_queue=None,
                progress_update_every_tiles=progress_update_every_tiles,
                progress_update_interval_seconds=progress_update_interval_seconds,
            )
            results.append(result)
            _report_batch_result(result)
            overall.update(
                index,
                detail=_overall_progress_detail(result, batch_start),
            )
    else:
        print(f"Processing {len(wsi_files)} WSIs with {worker_count} workers.", flush=True)
        ordered_results: dict[int, BatchSlideResult] = {}
        futures: dict[concurrent.futures.Future[tuple[int, BatchSlideResult]], tuple[int, Path]] = {}
        worker_states: dict[str, dict[str, Any]] = {}
        finished_wsi: set[str] = set()
        manager_context = mp.Manager() if show_progress else contextlib.nullcontext(None)
        with manager_context as manager:
            progress_queue = manager.Queue() if manager is not None else None
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                for index, slide_path in enumerate(wsi_files, start=1):
                    future = executor.submit(
                        _process_batch_slide_item_worker,
                        index,
                        str(slide_path),
                        str(output_path),
                        _json_safe(base_config),
                        marker_name,
                        effective_force,
                        resume_partial,
                        return_images,
                        progress_queue,
                        progress_update_every_tiles,
                        progress_update_interval_seconds,
                    )
                    futures[future] = (index, slide_path)

                pending = set(futures)
                completed = 0
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if progress_queue is not None and _drain_progress_events(
                        progress_queue,
                        worker_states,
                        finished_wsi,
                    ):
                        overall.update(
                            completed,
                            detail=_parallel_progress_detail(worker_states, batch_start),
                        )
                    for future in done:
                        fallback_index, fallback_slide_path = futures[future]
                        try:
                            index, result = future.result()
                        except Exception as exc:  # noqa: BLE001 - preserve batch summary on worker crashes
                            index = fallback_index
                            result = BatchSlideResult(
                                wsi_filename=fallback_slide_path.name,
                                slide_name=fallback_slide_path.stem,
                                output_dir=output_path / fallback_slide_path.stem,
                                completion_timestamp=_timestamp(),
                                status="Failed",
                                error=str(exc),
                            )
                            _write_failure_log(result.output_dir / "logs" / "processing.log", exc)
                        ordered_results[index] = result
                        completed += 1
                        finished_wsi.add(result.wsi_filename)
                        worker_states.pop(result.wsi_filename, None)
                        _report_batch_result(result)
                        overall.update(
                            completed,
                            detail=_overall_progress_detail(result, batch_start),
                        )
                if progress_queue is not None and _drain_progress_events(
                    progress_queue,
                    worker_states,
                    finished_wsi,
                ):
                    overall.update(
                        completed,
                        detail=_parallel_progress_detail(worker_states, batch_start),
                    )

        results = [ordered_results[index] for index in sorted(ordered_results)]

    batch_summary_path = output_path / str(batch_cfg.get("batch_summary_filename", "batch_summary.csv"))
    export_batch_summary(results, batch_summary_path)
    overall.finish(detail=f"Batch summary:\n{batch_summary_path}")
    return results


def _process_batch_slide_item_worker(
    index: int,
    slide_path: str,
    output_path: str,
    base_config: Mapping[str, Any],
    marker_name: str,
    force: bool,
    resume_partial: bool,
    return_images: bool,
    progress_queue: Any | None,
    progress_update_every_tiles: int,
    progress_update_interval_seconds: float,
) -> tuple[int, BatchSlideResult]:
    result = _process_batch_slide_item(
        index=index,
        slide_path=Path(slide_path),
        output_path=Path(output_path),
        base_config=base_config,
        marker_name=marker_name,
        force=force,
        resume_partial=resume_partial,
        return_images=return_images,
        slide_factory=None,
        show_tile_progress=False,
        print_summary=False,
        prefix_logs=True,
        progress_queue=progress_queue,
        progress_update_every_tiles=progress_update_every_tiles,
        progress_update_interval_seconds=progress_update_interval_seconds,
    )
    return index, result


def _process_batch_slide_item(
    index: int,
    slide_path: Path,
    output_path: Path,
    base_config: Mapping[str, Any],
    marker_name: str,
    force: bool,
    resume_partial: bool,
    return_images: bool,
    slide_factory: Callable[[Path], Any] | None,
    show_tile_progress: bool,
    print_summary: bool,
    prefix_logs: bool = False,
    progress_queue: Any | None = None,
    progress_update_every_tiles: int = 100,
    progress_update_interval_seconds: float = 2.0,
) -> BatchSlideResult:
    del index  # Ordering is handled by the caller.
    slide_start = time.perf_counter()
    slide_name = slide_path.stem
    slide_output_dir = output_path / slide_name
    marker_path = slide_output_dir / marker_name

    if _is_slide_completed(slide_output_dir, marker_name) and not force:
        marker_data = _read_completion_marker(marker_path)
        result = _result_from_marker(slide_path, slide_output_dir, marker_data)
        result.status = "Skipped"
        return result

    if force and slide_output_dir.exists():
        shutil.rmtree(slide_output_dir)
    elif slide_output_dir.exists() and not resume_partial:
        return BatchSlideResult(
            wsi_filename=slide_path.name,
            slide_name=slide_name,
            output_dir=slide_output_dir,
            completion_timestamp=_timestamp(),
            status="Failed",
            error=(
                "Partial output exists and batch.resume_partial is false. "
                "Use --force to restart or enable resume_partial."
            ),
        )

    _create_slide_output_layout(slide_output_dir)
    slide_config = _build_batch_slide_config(base_config, slide_path, slide_output_dir)
    _write_config_snapshot(slide_config, slide_output_dir / "logs" / "configuration_snapshot.yaml")
    existing_counts = _count_existing_slide_tiles(slide_config, slide_output_dir)
    tile_progress_holder: dict[str, ProgressPrinter] = {}
    progress_update_every_tiles = max(1, int(progress_update_every_tiles or 1))
    progress_update_interval_seconds = max(0.0, float(progress_update_interval_seconds or 0.0))
    tile_progress_state = {"last_current": 0, "last_time": 0.0}

    def _tile_progress(current: int, total: int, label: str) -> None:
        now = time.perf_counter()
        should_update = (
            current <= 1
            or current >= total
            or current - int(tile_progress_state["last_current"]) >= progress_update_every_tiles
            or now - float(tile_progress_state["last_time"]) >= progress_update_interval_seconds
        )
        if not should_update:
            return
        tile_progress_state["last_current"] = current
        tile_progress_state["last_time"] = now

        if progress_queue is not None:
            _put_progress_event(
                progress_queue,
                {
                    "wsi_filename": slide_path.name,
                    "slide_name": slide_name,
                    "current": int(current),
                    "total": int(total),
                    "detail": str(label),
                    "timestamp": time.time(),
                },
            )

        progress = tile_progress_holder.get("progress")
        if progress is None or progress.total != total:
            progress = ProgressPrinter("Tile generation", total, enabled=show_tile_progress)
            tile_progress_holder["progress"] = progress
        progress.update(
            current,
            detail=(
                f"Current:\n{slide_path.name}\n\n"
                f"{label}"
            ),
        )

    try:
        with _slide_file_logging(slide_output_dir / "logs" / "processing.log"):
            log_context = _slide_log_prefix(slide_path.name) if prefix_logs else contextlib.nullcontext()
            with log_context:
                LOGGER.info("Starting WSI batch item: %s", slide_path)
                if progress_queue is not None:
                    _put_progress_event(
                        progress_queue,
                        {
                            "wsi_filename": slide_path.name,
                            "slide_name": slide_name,
                            "current": 0,
                            "total": 0,
                            "detail": "Stage: start\nAction: opened worker",
                            "timestamp": time.time(),
                        },
                    )
                if existing_counts["total"] > 0:
                    LOGGER.info(
                        "Resuming partial output for %s: %s accepted files, %s rejected files already exist.",
                        slide_path.name,
                        existing_counts["accepted"],
                        existing_counts["rejected"],
                    )
                slide_obj = slide_factory(slide_path) if slide_factory else None
                try:
                    tile_results = generate_tiles(
                        config=slide_config,
                        slide=slide_obj,
                        return_images=return_images,
                        save_to_disk=True,
                        progress_callback=(
                            _tile_progress if show_tile_progress or progress_queue is not None else None
                        ),
                    )
                finally:
                    if slide_obj is not None and hasattr(slide_obj, "close"):
                        slide_obj.close()

                statistics = dict(tile_results["statistics"])
                elapsed = time.perf_counter() - slide_start
                result = BatchSlideResult(
                    wsi_filename=slide_path.name,
                    slide_name=slide_name,
                    output_dir=slide_output_dir,
                    accepted_tiles=int(statistics.get("accepted_tiles", 0)),
                    rejected_tiles=int(statistics.get("rejected_tiles", 0)),
                    total_tiles=int(statistics.get("candidate_tiles", 0)),
                    tissue_area=int(statistics.get("tissue_area", 0)),
                    processing_time_seconds=elapsed,
                    completion_timestamp=_timestamp(),
                    status="Completed",
                )
                _write_completion_marker(marker_path, result, statistics)
                LOGGER.info("Completed WSI batch item: %s", slide_path)
                if print_summary:
                    _print_slide_summary(result)
                return result
    except Exception as exc:  # noqa: BLE001 - batch processing should continue
        elapsed = time.perf_counter() - slide_start
        result = BatchSlideResult(
            wsi_filename=slide_path.name,
            slide_name=slide_name,
            output_dir=slide_output_dir,
            processing_time_seconds=elapsed,
            completion_timestamp=_timestamp(),
            status="Failed",
            error=str(exc),
        )
        _write_failure_log(slide_output_dir / "logs" / "processing.log", exc)
        return result


def _report_batch_result(result: BatchSlideResult) -> None:
    if result.status == "Completed":
        _print_slide_summary(result)
    elif result.status == "Skipped":
        print(f"Skipped (already processed): {result.wsi_filename}", flush=True)
    elif result.status == "Failed":
        print(f"Failed: {result.wsi_filename}: {result.error}", flush=True)


def _put_progress_event(progress_queue: Any, event: Mapping[str, Any]) -> None:
    try:
        progress_queue.put(dict(event), block=False)
    except Exception:  # pragma: no cover - progress must never stop tile generation
        return


def _drain_progress_events(
    progress_queue: Any,
    worker_states: dict[str, dict[str, Any]],
    finished_wsi: set[str] | None = None,
) -> bool:
    updated = False
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue_module.Empty:
            break
        except Exception:  # pragma: no cover - progress must never stop batch processing
            break
        if not isinstance(event, Mapping):
            continue
        wsi_filename = str(event.get("wsi_filename", ""))
        if not wsi_filename:
            continue
        if finished_wsi is not None and wsi_filename in finished_wsi:
            continue
        worker_states[wsi_filename] = dict(event)
        updated = True
    return updated


def _parallel_progress_detail(worker_states: Mapping[str, Mapping[str, Any]], batch_start: float) -> str:
    del batch_start  # Elapsed time is printed by ProgressPrinter.
    lines: list[str] = []
    if not worker_states:
        lines.append("Waiting for workers...")
        return "\n".join(lines)

    lines.append("Active workers:")
    for wsi_filename, state in sorted(worker_states.items())[:8]:
        current = int(state.get("current", 0) or 0)
        total = int(state.get("total", 0) or 0)
        lines.append(f"- {wsi_filename}: {current}/{total}")
        detail = str(state.get("detail", "")).strip()
        if detail:
            for detail_line in detail.splitlines():
                lines.append(f"  {detail_line}")
    if len(worker_states) > 8:
        lines.append(f"... {len(worker_states) - 8} more active workers")
    return "\n".join(lines)


def _overall_progress_detail(result: BatchSlideResult, batch_start: float) -> str:
    del batch_start  # Elapsed time is printed by ProgressPrinter.
    return (
        f"Current:\n{result.wsi_filename}\n\n"
        f"Accepted: {result.accepted_tiles}\n"
        f"Rejected: {result.rejected_tiles}"
    )


def _save_progress_label(stage: str, action: str, filename: str) -> str:
    return f"{stage}|{action}|{filename}"


def _parse_save_progress_label(label: str) -> tuple[str, str, str]:
    parts = str(label).split("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return str(label), "processing", ""


def _format_tile_progress_detail(
    stage: str,
    action: str,
    filename: str,
    accepted_done: int,
    accepted_total: int,
    rejected_done: int,
    rejected_total: int,
) -> str:
    lines = [
        f"Stage: {stage}",
        f"Action: {action}",
    ]
    if filename:
        lines.append(f"Last file: {filename}")
    lines.extend(
        [
            f"Accepted: {accepted_done}/{accepted_total}",
            f"Rejected: {rejected_done}/{rejected_total}",
        ]
    )
    return "\n".join(lines)


def discover_wsi_files(input_dir: str | Path) -> list[Path]:
    """Return supported WSI files sorted alphabetically by filename."""

    path = Path(input_dir)
    if not path.exists():
        raise TileGenerationError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise TileGenerationError(f"Input path is not a directory: {path}")
    return sorted(
        [
            item
            for item in path.iterdir()
            if (
                item.is_file()
                and item.suffix.lower() in SUPPORTED_WSI_EXTENSIONS
                and not item.name.startswith(".")
                and not item.name.startswith("._")
            )
        ],
        key=lambda item: item.name.lower(),
    )


def export_batch_summary(results: Sequence[BatchSlideResult], csv_path: str | Path) -> Path:
    """Write one batch summary row per WSI."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "WSI filename",
        "accepted tiles",
        "rejected tiles",
        "total tiles",
        "tissue area",
        "processing time",
        "processing time seconds",
        "completion timestamp",
        "status",
        "output folder",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "WSI filename": result.wsi_filename,
                    "accepted tiles": result.accepted_tiles,
                    "rejected tiles": result.rejected_tiles,
                    "total tiles": result.total_tiles,
                    "tissue area": result.tissue_area,
                    "processing time": _format_duration(result.processing_time_seconds),
                    "processing time seconds": f"{result.processing_time_seconds:.4f}",
                    "completion timestamp": result.completion_timestamp,
                    "status": result.status,
                    "output folder": str(result.output_dir),
                    "error": result.error,
                }
            )
    LOGGER.info("Exported batch summary to %s.", path)
    return path


def _build_batch_slide_config(
    base_config: Mapping[str, Any],
    slide_path: Path,
    slide_output_dir: Path,
) -> dict[str, Any]:
    cfg = _deep_copy(dict(base_config))
    overrides = {
        "slide_path": str(slide_path),
        "output_dir": str(slide_output_dir),
        "saving": {
            "accepted_tiles_subdir": "tiles/accepted_tiles",
            "rejected_tiles_subdir": "tiles/rejected_tiles",
        },
        "metadata": {
            "filename": "metadata/tile_metadata.csv",
            "coordinates_filename": "metadata/tile_coordinates.csv",
            "summary_filename": "metadata/tile_summary.csv",
        },
        "visualization": {
            "filename": "visualization/patient_tile_map.png",
            "tissue_mask_filename": "visualization/tissue_mask.png",
        },
        "statistics": {
            "filename": "metadata/tile_statistics.json",
        },
    }
    return _deep_merge(cfg, overrides)


def _create_slide_output_layout(slide_output_dir: Path) -> None:
    for subdir in [
        slide_output_dir / "tiles" / "accepted_tiles",
        slide_output_dir / "tiles" / "rejected_tiles",
        slide_output_dir / "metadata",
        slide_output_dir / "visualization",
        slide_output_dir / "logs",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)


def _is_slide_completed(slide_output_dir: Path, marker_name: str) -> bool:
    marker = slide_output_dir / marker_name
    metadata = slide_output_dir / "metadata" / "tile_metadata.csv"
    coordinates = slide_output_dir / "metadata" / "tile_coordinates.csv"
    return marker.exists() or (metadata.exists() and coordinates.exists())


def _count_existing_slide_tiles(config: Mapping[str, Any], slide_output_dir: Path) -> dict[str, int]:
    saving_cfg = config.get("saving", {})
    accepted_dir = slide_output_dir / str(saving_cfg.get("accepted_tiles_subdir", "tiles/accepted_tiles"))
    rejected_dir = slide_output_dir / str(saving_cfg.get("rejected_tiles_subdir", "tiles/rejected_tiles"))
    accepted = _count_tile_files(accepted_dir)
    rejected = _count_tile_files(rejected_dir)
    return {"accepted": accepted, "rejected": rejected, "total": accepted + rejected}


def _count_tile_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and not path.name.startswith(".") and not path.name.startswith("._")
    )


def _read_completion_marker(marker_path: Path) -> dict[str, Any]:
    if not marker_path.exists():
        return {}
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _result_from_marker(
    slide_path: Path,
    slide_output_dir: Path,
    marker_data: Mapping[str, Any],
) -> BatchSlideResult:
    statistics = marker_data.get("statistics", {})
    if not isinstance(statistics, Mapping):
        statistics = {}
    if not statistics:
        statistics = _read_slide_summary_statistics(slide_output_dir / "metadata" / "tile_summary.csv")
    return BatchSlideResult(
        wsi_filename=slide_path.name,
        slide_name=slide_path.stem,
        output_dir=slide_output_dir,
        accepted_tiles=int(statistics.get("accepted_tiles", marker_data.get("accepted_tiles", 0)) or 0),
        rejected_tiles=int(statistics.get("rejected_tiles", marker_data.get("rejected_tiles", 0)) or 0),
        total_tiles=int(statistics.get("candidate_tiles", marker_data.get("total_tiles", 0)) or 0),
        tissue_area=int(statistics.get("tissue_area", marker_data.get("tissue_area", 0)) or 0),
        processing_time_seconds=float(marker_data.get("processing_time_seconds", 0.0) or 0.0),
        completion_timestamp=str(marker_data.get("completion_timestamp", "")),
        status=str(marker_data.get("status", "Completed")),
    )


def _read_slide_summary_statistics(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {}
    if not rows:
        return {}
    row = rows[0]
    return {
        "accepted_tiles": row.get("accepted_tiles", 0),
        "rejected_tiles": row.get("rejected_tiles", 0),
        "candidate_tiles": row.get("candidate_tiles", 0),
        "tissue_area": row.get("tissue_area", 0),
    }


def _write_completion_marker(
    marker_path: Path,
    result: BatchSlideResult,
    statistics: Mapping[str, Any],
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "wsi_filename": result.wsi_filename,
        "slide_name": result.slide_name,
        "accepted_tiles": result.accepted_tiles,
        "rejected_tiles": result.rejected_tiles,
        "total_tiles": result.total_tiles,
        "tissue_area": result.tissue_area,
        "processing_time_seconds": result.processing_time_seconds,
        "completion_timestamp": result.completion_timestamp,
        "status": result.status,
        "statistics": dict(statistics),
    }
    with marker_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_config_snapshot(config: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _json_safe(config)
    if yaml is not None:
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(serializable, handle, sort_keys=False)
    else:
        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)
            handle.write("\n")


@contextlib.contextmanager
def _slide_file_logging(log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


class _SlideLogPrefixFilter(logging.Filter):
    def __init__(self, slide_name: str) -> None:
        super().__init__()
        self.slide_name = slide_name

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "_slide_name_prefixed", False):
            return True
        record.msg = f"[{self.slide_name}] {record.getMessage()}"
        record.args = ()
        record._slide_name_prefixed = True
        return True


@contextlib.contextmanager
def _slide_log_prefix(slide_name: str) -> Any:
    root_logger = logging.getLogger()
    log_filter = _SlideLogPrefixFilter(slide_name)
    handlers = list(root_logger.handlers)
    for handler in handlers:
        handler.addFilter(log_filter)
    try:
        yield
    finally:
        for handler in handlers:
            handler.removeFilter(log_filter)


def _write_failure_log(log_path: Path, exc: Exception) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_timestamp()} ERROR Failed: {exc}\n")


def _print_slide_summary(result: BatchSlideResult) -> None:
    print("-" * 52, flush=True)
    print(f"Finished:\n{result.slide_name}", flush=True)
    print(f"Accepted tiles : {result.accepted_tiles}", flush=True)
    print(f"Rejected tiles : {result.rejected_tiles}", flush=True)
    print(f"Total tiles    : {result.total_tiles}", flush=True)
    print(f"Elapsed time   : {_format_duration(result.processing_time_seconds)}", flush=True)
    print("-" * 52, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(description="Generate native level-0 WSI tiles.")
    parser.add_argument(
        "--config",
        default="tile_generator_config.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--slide", help="Override slide_path from the config.")
    parser.add_argument("--output-dir", help="Override output_dir from the config.")
    parser.add_argument("--input-dir", help="Batch input directory containing .tif/.tiff WSIs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing completed WSI outputs.")
    parser.add_argument("--workers", type=int, help="Number of parallel WSI workers for batch mode.")
    parser.add_argument(
        "--progress-every-tiles",
        type=int,
        help="Update terminal progress after this many saved/skipped tiles.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        help="Update terminal progress after this many seconds even if fewer tiles were processed.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars.")
    args = parser.parse_args(argv)

    cfg_for_mode = load_config(args.config)
    batch_input_dir = args.input_dir or cfg_for_mode.get("batch", {}).get("input_dir")
    if batch_input_dir:
        batch_output_dir = (
            args.output_dir
            or cfg_for_mode.get("batch", {}).get("output_dir")
            or cfg_for_mode.get("output_dir", "output")
        )
        batch_overrides: dict[str, Any] = {}
        if args.progress_every_tiles is not None:
            batch_overrides.setdefault("batch", {})["progress_update_every_tiles"] = args.progress_every_tiles
        if args.progress_interval_seconds is not None:
            batch_overrides.setdefault("batch", {})[
                "progress_update_interval_seconds"
            ] = args.progress_interval_seconds
        process_wsi_batch(
            input_dir=batch_input_dir,
            output_dir=batch_output_dir,
            config_path=args.config,
            config=batch_overrides or None,
            force=args.force,
            return_images=False,
            show_progress=not args.no_progress,
            workers=args.workers,
        )
        return 0

    overrides: dict[str, Any] = {}
    if args.slide:
        overrides["slide_path"] = args.slide
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    results = run_tile_generation(args.config, overrides)
    LOGGER.info(
        "Done. Accepted %s tiles. Metadata: %s. Visualization: %s.",
        len(results["accepted_tiles"]),
        results["metadata_path"],
        results["visualization_path"],
    )
    return 0


def _open_slide(slide_path: str | Path) -> tuple[Any, bool]:
    if openslide is None:
        raise TileGenerationError(
            "openslide-python is required to open WSI files. Install openslide-python "
            "and the OpenSlide shared library, or pass an OpenSlide-like object."
        )
    path = Path(slide_path)
    if not path.exists():
        raise TileGenerationError(f"Slide path does not exist: {path}")
    return openslide.OpenSlide(str(path)), True


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _safe_filename_token(value: str) -> str:
    token = str(value).strip().replace(" ", "-")
    if not token:
        raise TileGenerationError("Filename token cannot be empty.")
    if re.fullmatch(r"[A-Za-z0-9-]+", token) is None:
        raise TileGenerationError(f"Unsafe filename token: {value}")
    return token


def _coordinate_tile_id(x: int, y: int) -> str:
    return f"x{int(x)}_y{int(y)}"


def _timestamp() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _estimate_eta(elapsed: float, current: int, total: int) -> float | None:
    if current <= 0 or total <= 0 or current >= total:
        return None
    rate = elapsed / current
    return rate * (total - current)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _slide_dimensions(slide: Any) -> tuple[int, int]:
    try:
        width, height = slide.dimensions
    except AttributeError as exc:
        raise TileGenerationError("Slide object must expose OpenSlide-style dimensions.") from exc
    return int(width), int(height)


def _as_size_tuple(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise TileGenerationError(f"{name} must contain exactly two values.")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise TileGenerationError(f"{name} values must be positive.")
    return width, height


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _clip_roi(roi: ROI, slide_dimensions: Sequence[int]) -> ROI:
    slide_width, slide_height = _as_size_tuple(slide_dimensions, "slide_dimensions")
    return ROI(
        x1=_clip_int(roi.x1, 0, slide_width),
        y1=_clip_int(roi.y1, 0, slide_height),
        x2=_clip_int(roi.x2, 0, slide_width),
        y2=_clip_int(roi.y2, 0, slide_height),
    )


def _intersect_boxes(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    max_channel = arr.max(axis=-1)
    min_channel = arr.min(axis=-1)
    delta = max_channel - min_channel

    saturation = np.zeros_like(max_channel)
    np.divide(delta, max_channel, out=saturation, where=max_channel > 0)
    value = max_channel

    hue = np.zeros_like(max_channel)
    nonzero = delta > 1e-6
    red_max = (max_channel == r) & nonzero
    green_max = (max_channel == g) & nonzero
    blue_max = (max_channel == b) & nonzero
    hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6
    hue[green_max] = ((b[green_max] - r[green_max]) / delta[green_max]) + 2
    hue[blue_max] = ((r[blue_max] - g[blue_max]) / delta[blue_max]) + 4
    hue /= 6.0
    return hue, saturation, value


def _hsv_tissue_mask(rgb: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    _, saturation, value = _rgb_to_hsv(rgb)
    min_saturation = float(config.get("min_saturation", 0.08))
    min_value = float(config.get("min_value", 0.05))
    max_value = float(config.get("max_value", 0.93))
    return (saturation >= min_saturation) & (value >= min_value) & (value <= max_value)


def _otsu_tissue_mask(rgb: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    channel = str(config.get("channel", "saturation")).lower()
    if channel == "saturation":
        _, saturation, value = _rgb_to_hsv(rgb)
        scaled = np.clip(np.rint(saturation * 255), 0, 255).astype(np.uint8)
        threshold = _otsu_threshold(scaled)
        min_saturation = float(config.get("min_saturation", 0.04))
        max_value = float(config.get("max_value", 0.97))
        return (scaled >= threshold) & (saturation >= min_saturation) & (value <= max_value)
    if channel in {"gray", "grayscale", "luminance"}:
        gray = _rgb_to_luminance(rgb)
        threshold = _otsu_threshold(gray)
        return gray <= threshold
    if channel in {"optical_density", "od"}:
        gray = _rgb_to_luminance(rgb).astype(np.float32)
        optical_density = -np.log((gray + 1.0) / 256.0)
        scaled = np.clip(np.rint(255.0 * optical_density / optical_density.max()), 0, 255)
        threshold = _otsu_threshold(scaled.astype(np.uint8))
        return scaled >= threshold
    raise TileGenerationError(
        "tissue_detection.otsu.channel must be saturation, grayscale, or optical_density."
    )


def _lab_tissue_mask(rgb: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    l_channel, a_channel, b_channel = _rgb_to_lab(rgb)
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    max_l = float(config.get("max_l", 92.0))
    min_chroma = float(config.get("min_chroma", 5.0))
    return (l_channel <= max_l) & (chroma >= min_chroma)


def _rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    gray = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def _otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0
    bins = np.arange(256, dtype=np.float64)
    weight_background = np.cumsum(hist)
    weight_foreground = total - weight_background
    mean_background = np.cumsum(hist * bins)
    total_mean = mean_background[-1]

    valid = (weight_background > 0) & (weight_foreground > 0)
    variance = np.zeros(256, dtype=np.float64)
    numerator = (total_mean * weight_background - mean_background) ** 2
    denominator = weight_background * weight_foreground
    variance[valid] = numerator[valid] / denominator[valid]
    return int(np.argmax(variance))


def _rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = rgb.astype(np.float32) / 255.0
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    x = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    z = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041

    x /= 0.95047
    z /= 1.08883

    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    l_channel = 116.0 * fy - 16.0
    a_channel = 500.0 * (fx - fy)
    b_channel = 200.0 * (fy - fz)
    return l_channel, a_channel, b_channel


def _lab_f(value: np.ndarray) -> np.ndarray:
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return np.where(value > epsilon, np.cbrt(value), (kappa * value + 16.0) / 116.0)


def _require_scipy() -> None:
    if ndi is None:
        raise TileGenerationError(
            "scipy is required for mask cleaning and connected component analysis."
        )


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_area)
    keep[0] = False
    return keep[labels]


def _disk_structure(radius: int) -> np.ndarray:
    y_grid, x_grid = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x_grid * x_grid + y_grid * y_grid) <= radius * radius


def _thumbnail_box_to_level0(thumbnail: Thumbnail, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        _clip_int(math.floor(x1 * thumbnail.scale_x), 0, thumbnail.level0_width),
        _clip_int(math.floor(y1 * thumbnail.scale_y), 0, thumbnail.level0_height),
        _clip_int(math.ceil(x2 * thumbnail.scale_x), 0, thumbnail.level0_width),
        _clip_int(math.ceil(y2 * thumbnail.scale_y), 0, thumbnail.level0_height),
    )


def _level0_box_to_thumbnail(
    thumbnail: Thumbnail,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = _clip_int(math.floor(x / thumbnail.scale_x), 0, thumbnail.width)
    y1 = _clip_int(math.floor(y / thumbnail.scale_y), 0, thumbnail.height)
    x2 = _clip_int(math.ceil((x + width) / thumbnail.scale_x), 0, thumbnail.width)
    y2 = _clip_int(math.ceil((y + height) / thumbnail.scale_y), 0, thumbnail.height)
    if x2 <= x1 and x1 < thumbnail.width:
        x2 = x1 + 1
    if y2 <= y1 and y1 < thumbnail.height:
        y2 = y1 + 1
    return x1, y1, x2, y2


def _validate_label_image(label_image: np.ndarray, thumbnail: Thumbnail) -> np.ndarray:
    labels = np.asarray(label_image)
    if labels.shape != (thumbnail.height, thumbnail.width):
        raise TileGenerationError(
            f"Label image shape {labels.shape} does not match thumbnail "
            f"shape {(thumbnail.height, thumbnail.width)}."
        )
    return labels


def _resolve_stride(tile_size: int, overlap: int | None, stride: int | None) -> int:
    tile_size = int(tile_size)
    if stride is not None:
        stride = int(stride)
        if stride <= 0:
            raise TileGenerationError("tiling.stride must be positive.")
        if overlap is not None and int(overlap) > 0 and stride != tile_size - int(overlap):
            raise TileGenerationError(
                "tiling.stride must equal tiling.tile_size - tiling.overlap when both are set."
            )
        return stride
    if overlap is not None:
        overlap = int(overlap)
        if overlap < 0:
            raise TileGenerationError("tiling.overlap cannot be negative.")
        if overlap >= tile_size:
            raise TileGenerationError("tiling.overlap must be smaller than tiling.tile_size.")
        return tile_size - overlap
    return tile_size


def _validate_tile_grid(tile_size: int, stride: int) -> None:
    if tile_size <= 0:
        raise TileGenerationError("tiling.tile_size must be positive.")
    if stride <= 0:
        raise TileGenerationError("tiling.stride must be positive.")


def _grid_positions(
    start: int,
    stop: int,
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
) -> list[int]:
    last_start = stop - tile_size
    if last_start < start:
        return []
    positions = list(range(start, last_start + 1, stride))
    if include_edge_tiles and positions and positions[-1] != last_start:
        positions.append(last_start)
    elif include_edge_tiles and not positions:
        positions.append(last_start)
    return sorted(set(positions))


def _aligned_grid_positions_for_window(
    origin: int,
    roi_stop: int,
    window_start: int,
    window_stop: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    """Return ROI-origin-aligned tile starts whose tiles intersect a window."""

    last_valid_start = roi_stop - tile_size
    if last_valid_start < origin:
        return []
    first_intersecting_start = window_start - tile_size + 1
    last_intersecting_start = window_stop - 1
    start = max(origin, first_intersecting_start)
    stop = min(last_valid_start, last_intersecting_start)
    if stop < start:
        return []
    first_k = math.ceil((start - origin) / stride)
    last_k = math.floor((stop - origin) / stride)
    return [origin + k * stride for k in range(first_k, last_k + 1)]


def _tile_tissue_percentage(
    label_image: np.ndarray,
    thumbnail: Thumbnail,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[float, int]:
    tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(thumbnail, x, y, width, height)
    region = label_image[ty1:ty2, tx1:tx2]
    if region.size == 0:
        return 0.0, 0
    tissue = region > 0
    tissue_pixels = int(np.count_nonzero(tissue))
    if tissue_pixels == 0:
        return 0.0, 0
    positive_labels = region[tissue]
    counts = np.bincount(positive_labels.astype(np.int64))
    counts[0] = 0
    dominant_component = int(np.argmax(counts))
    return 100.0 * tissue_pixels / float(region.size), dominant_component


def _tile_white_percentage(
    thumbnail: Thumbnail,
    thumbnail_rgb: np.ndarray | None,
    x: int,
    y: int,
    width: int,
    height: int,
    min_value: float = 0.90,
    max_saturation: float = 0.12,
) -> float:
    tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(thumbnail, x, y, width, height)
    if thumbnail_rgb is None:
        thumbnail_rgb = _image_to_rgb_array(thumbnail.image)
    rgb = thumbnail_rgb[ty1:ty2, tx1:tx2]
    if rgb.size == 0:
        return 100.0
    _, saturation, value = _rgb_to_hsv(rgb)
    white = (value >= float(min_value)) & (saturation <= float(max_saturation))
    return 100.0 * float(np.count_nonzero(white)) / float(white.size)


def _normalize_percentage(value: float, label: str = "percentage") -> float:
    value = float(value)
    if value < 0:
        raise TileGenerationError(f"{label} threshold cannot be negative.")
    if value <= 1.0:
        return value * 100.0
    if value > 100.0:
        raise TileGenerationError(f"{label} threshold cannot exceed 100.")
    return value


def _image_extension(image_format: str) -> str:
    fmt = str(image_format).lower().strip(".")
    if fmt == "jpg":
        return "jpg"
    if fmt == "jpeg":
        return "jpeg"
    if fmt in {"png", "tif", "tiff", "webp"}:
        return fmt
    raise TileGenerationError(f"Unsupported image format: {image_format}")


def _assign_tile_paths(
    tiles: Sequence[TileRecord],
    output_dir: str | Path,
    image_format: str = "png",
    filename_prefix: str = "tile",
    include_component_id: bool = False,
) -> None:
    output_path = Path(output_dir)
    ext = _image_extension(image_format)
    for tile in tiles:
        tile.path = output_path / coordinate_to_filename(
            tile.x,
            tile.y,
            extension=ext,
            component_id=tile.component_id,
            prefix=filename_prefix,
            include_component_id=include_component_id,
        )


def _is_existing_tile_valid(path: Path, expected_size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            if image.size != expected_size:
                return False
            image.verify()
        return True
    except Exception:
        return False


def _save_image_atomic(
    image: Image.Image,
    path: Path,
    atomic_writes: bool,
    save_kwargs: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not atomic_writes:
        image.save(path, **dict(save_kwargs))
        return
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with contextlib.suppress(FileNotFoundError):
        tmp_path.unlink()
    image.save(tmp_path, **dict(save_kwargs))
    tmp_path.replace(path)


def _tile_metadata(tile: TileRecord) -> dict[str, Any]:
    return {
        "tile_filename": tile.path.name if tile.path else coordinate_to_filename(tile.x, tile.y),
        "tile_id": tile.tile_id,
        "component_id": tile.component_id,
        "level0_x": tile.x,
        "level0_y": tile.y,
        "tile_width": tile.width,
        "tile_height": tile.height,
        "thumbnail_x": tile.thumb_x,
        "thumbnail_y": tile.thumb_y,
        "tissue_percentage": tile.tissue_percentage,
        "white_percentage": tile.white_percentage,
        "tile_status": "accepted" if tile.accepted else "rejected",
        "rejection_reason": tile.rejection_reason,
        "path": str(tile.path) if tile.path else None,
        "skipped_existing": tile.skipped_existing,
    }


def _build_generated_tiles(
    slide: Any,
    accepted_tiles: Sequence[TileRecord],
    return_images: bool,
) -> list[GeneratedTile]:
    generated: list[GeneratedTile] = []
    for tile in accepted_tiles:
        image = None
        if return_images:
            image = slide.read_region((tile.x, tile.y), 0, (tile.width, tile.height)).convert("RGB")
            if image.size != (tile.width, tile.height):
                raise TileGenerationError(
                    f"Unexpected tile size {image.size} for {tile.tile_id}; "
                    f"expected {(tile.width, tile.height)}."
                )
        generated.append(
            GeneratedTile(
                tile_id=tile.tile_id,
                image=image,
                level0_coordinates=(tile.x, tile.y, tile.width, tile.height),
                thumbnail_coordinates=(tile.thumb_x, tile.thumb_y, tile.thumb_width, tile.thumb_height),
                component_id=tile.component_id,
                metadata=_tile_metadata(tile),
                path=tile.path,
                skipped_existing=tile.skipped_existing,
            )
        )
    return generated


def _build_statistics(
    slide_name: str,
    roi: ROI | None,
    components: Sequence[Component],
    candidates: Sequence[TileRecord],
    accepted: Sequence[TileRecord],
    rejected: Sequence[TileRecord],
    processing_time_seconds: float,
) -> dict[str, Any]:
    tissue_values = [tile.tissue_percentage for tile in accepted]
    white_values = [tile.white_percentage for tile in accepted]
    tissue_area = int(sum(component.area_pixels for component in components))
    return {
        "slide_name": slide_name,
        "ROI_width": roi.width if roi else 0,
        "ROI_height": roi.height if roi else 0,
        "num_components": len(components),
        "candidate_tiles": len(candidates),
        "accepted_tiles": len(accepted),
        "rejected_tiles": len(rejected),
        "tissue_area": tissue_area,
        "average_tissue_percent": float(np.mean(tissue_values)) if tissue_values else 0.0,
        "average_white_percent": float(np.mean(white_values)) if white_values else 0.0,
        "processing_time_seconds": float(processing_time_seconds),
    }


def _slide_name(config: Mapping[str, Any], slide: Any) -> str:
    slide_path = config.get("slide_path")
    if slide_path:
        return Path(slide_path).name
    filename = getattr(slide, "_filename", None) or getattr(slide, "filename", None)
    if filename:
        return Path(str(filename)).name
    return "in_memory_slide"


def _limit_tiles(tiles: Sequence[TileRecord], max_tiles: int | None) -> Sequence[TileRecord]:
    if max_tiles is None or max_tiles <= 0 or len(tiles) <= max_tiles:
        return tiles
    indices = np.linspace(0, len(tiles) - 1, max_tiles).astype(int)
    return [tiles[int(index)] for index in indices]


def _tile_label_budget(
    tile_count: int,
    draw_tile_labels: bool,
    max_tile_labels: int | None,
) -> bool:
    if not draw_tile_labels:
        return False
    if max_tile_labels is None or max_tile_labels <= 0:
        return True
    return tile_count <= max_tile_labels


def _draw_tile_boxes(
    ax: Any,
    thumbnail: Thumbnail,
    tiles: Sequence[TileRecord],
    color: str,
    rectangle_cls: Any,
    label_budget: bool,
) -> None:
    for tile in tiles:
        tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(
            thumbnail,
            tile.x,
            tile.y,
            tile.width,
            tile.height,
        )
        ax.add_patch(
            rectangle_cls(
                (tx1, ty1),
                max(1, tx2 - tx1),
                max(1, ty2 - ty1),
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.85,
            )
        )
        if label_budget:
            ax.text(
                tx1 + 1,
                ty1 + 1,
                f"{tile.x},{tile.y}",
                color=color,
                fontsize=5,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.2, "edgecolor": "none"},
            )


def _draw_roi_and_components(
    ax: Any,
    thumbnail: Thumbnail,
    components: Sequence[Component],
    roi: ROI | None,
    roi_mode: str,
    rectangle_cls: Any,
    draw_component_labels: bool = True,
) -> None:
    if roi is not None:
        tx1, ty1, tx2, ty2 = _level0_box_to_thumbnail(
            thumbnail,
            roi.x1,
            roi.y1,
            roi.width,
            roi.height,
        )
        ax.add_patch(
            rectangle_cls(
                (tx1, ty1),
                tx2 - tx1,
                ty2 - ty1,
                fill=False,
                edgecolor="yellow",
                linewidth=2.0,
            )
        )
        ax.text(
            tx1 + 2,
            ty1 + 10,
            f"ROI {roi_mode}",
            color="yellow",
            fontsize=7,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.3, "edgecolor": "none"},
        )

    for component in components:
        x1, y1, x2, y2 = component.bbox_thumbnail
        color = _component_color(component.component_id)
        ax.add_patch(
            rectangle_cls(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.9,
            )
        )
        if draw_component_labels:
            ax.text(
                x1 + max(1, (x2 - x1) // 2),
                y1 + max(1, (y2 - y1) // 2),
                f"C{component.component_id}",
                color=color,
                fontsize=7,
                weight="bold",
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 0.3, "edgecolor": "none"},
            )


def _component_color(component_id: int) -> tuple[float, float, float, float]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - visualization already guards this
        return (0.0, 1.0, 1.0, 1.0)
    cmap = plt.get_cmap("tab20")
    return cmap((component_id - 1) % 20)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
