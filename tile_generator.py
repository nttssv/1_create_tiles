"""OpenSlide-compatible WSI tile generation.

This module only generates candidate image tiles. It deliberately has no
dependency on downstream inference code such as YOLO, SAM, or CellSeg models.
Every saved tile is read from OpenSlide level 0 at the requested native pixel
size; thumbnail images are used only for tissue detection and tile filtering.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

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
        "filename_prefix": "tile",
        "include_component_id": False,
    },
    "metadata": {
        "filename": "tile_metadata.csv",
    },
    "visualization": {
        "filename": "patient_tile_map.png",
        "dpi": 160,
        "max_tiles_to_draw": None,
        "draw_tile_labels": True,
        "draw_component_labels": True,
        "max_tile_labels": 300,
    },
    "statistics": {
        "filename": "tile_statistics.json",
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
    filename_prefix: str = "tile",
    include_component_id: bool = False,
) -> list[Path]:
    """Crop and save accepted native-resolution level-0 tiles."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ext = _image_extension(image_format)
    saved_paths: list[Path] = []

    for tile in tiles:
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
        if path.exists() and resume:
            tile.skipped_existing = True
            saved_paths.append(path)
            continue
        if path.exists() and not overwrite:
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
        image.save(path, **save_kwargs)
        saved_paths.append(path)

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
        if save_to_disk:
            saved_tiles = save_tiles(
                slide_obj,
                accepted,
                accepted_tiles_dir,
                image_format=image_format,
                jpeg_quality=int(saving_cfg.get("jpeg_quality", 95)),
                overwrite=bool(saving_cfg.get("overwrite", False)),
                resume=bool(saving_cfg.get("resume", True)),
                filename_prefix=filename_prefix,
                include_component_id=include_component_id,
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
                    filename_prefix=filename_prefix,
                    include_component_id=include_component_id,
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
            "visualization_path": visualization_path,
            "statistics": statistics,
            "statistics_path": statistics_path,
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
    args = parser.parse_args(argv)

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
    return {
        "slide_name": slide_name,
        "ROI_width": roi.width if roi else 0,
        "ROI_height": roi.height if roi else 0,
        "num_components": len(components),
        "candidate_tiles": len(candidates),
        "accepted_tiles": len(accepted),
        "rejected_tiles": len(rejected),
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
