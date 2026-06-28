# WSI Tile Generator

This module redesigns only the tile creation stage. It is independent from YOLO,
SAM3.1, CellSeg1, and morphology analysis.

## Invariant

The inference input remains unchanged:

- 512x512 pixels by default
- cropped directly from the original WSI
- read from OpenSlide level 0
- never resized or downsampled before inference

The thumbnail is used only to detect tissue and estimate tissue coverage.

## Pipeline

1. Open the WSI with OpenSlide.
2. Create a low-resolution thumbnail.
3. Build a binary tissue mask with HSV, Otsu, LAB, or auto voting.
4. Clean the mask with small-object removal, hole filling, opening, and closing.
5. Label connected tissue components and assign component IDs.
6. Generate ROI-origin-aligned candidate level-0 tile coordinates.
7. Estimate tissue coverage from the thumbnail-derived component mask.
8. Reject tiles below the configured tissue threshold, default 70%, or above
   the configured white-background threshold.
9. Crop accepted and rejected tiles from OpenSlide level 0 as native 512x512 images.
10. Write CSV metadata, `patient_tile_map.png`, and `tile_statistics.json`.

## Deterministic Grid

Tiles are generated on a grid anchored to the effective ROI origin:

```text
x = ROI.x1 + k * stride
y = ROI.y1 + n * stride
```

`stride` is derived from overlap:

```text
stride = tile_size - overlap
```

Examples:

```text
tile_size=512, overlap=0   -> stride=512
tile_size=512, overlap=128 -> stride=384
tile_size=512, overlap=256 -> stride=256
```

Automatic mode uses the full slide as the grid ROI and evaluates only grid
tiles that intersect detected tissue components. Manual mode uses the manual
ROI, after optional padding and slide-boundary clipping. Candidate and accepted
tiles are always ordered top-to-bottom, then left-to-right.

## Coordinate Naming

The standard tile identifier is the level-0 top-left coordinate:

```text
x18432_y22528
```

The standard tile filename is:

```text
tile_x18432_y22528.png
```

Optionally, filenames can include component ID:

```text
tile_x18432_y22528_c1.png
```

The coordinate part is stable across future model outputs:

```text
tile_x18432_y22528_yolo.json
tile_x18432_y22528_sam.json
tile_x18432_y22528_cellseg.json
```

Use these utilities for QuPath export and downstream traceability:

```python
from tile_generator import (
    coordinate_to_filename,
    filename_to_coordinate,
    coordinate_to_rectangle,
)

filename = coordinate_to_filename(18432, 22528)
x, y = filename_to_coordinate(filename)
rectangle = coordinate_to_rectangle(x, y, 512, 512)
```

## Modes

Automatic mode:

```python
ROI_MODE = "auto"
```

Automatic mode tiles inside detected tissue components.

Manual mode:

```python
ROI_MODE = "manual"

MANUAL_ROI = {
    "x1": 18000,
    "y1": 12000,
    "x2": 34000,
    "y2": 28000,
}
```

Manual ROI coordinates are level-0 coordinates. The implementation sorts
reversed coordinates, clips the ROI to slide boundaries, computes width and
height, and prints:

```text
ROI
x1: ...
y1: ...
x2: ...
y2: ...
width: ...
height: ...
```

Set `manual_roi: null` in YAML to open the interactive thumbnail selector.
Matplotlib provides zoom and pan; after two clicks, the selected thumbnail
coordinates are converted back to level 0 and confirmation is required.

Manual ROI padding is configured with:

```yaml
tiling:
  roi_padding: 512
```

Padding is applied after coordinate sorting and before tile generation.

## Configuration

Edit `tile_generator_config.yaml`.

Important fields:

- `roi_mode`: `auto` or `manual`
- `manual_roi`: level-0 rectangle or `null`
- `thumbnail.max_size`: thumbnail size limit
- `tissue_detection.method`: `hsv`, `otsu`, `lab`, or `auto`
- `mask_cleaning.*`: morphology parameters
- `connected_components.min_area`: minimum component size in thumbnail pixels
- `tiling.tile_size`: keep `512` for the trained models
- `tiling.overlap`: overlap in level-0 pixels
- `tiling.stride`: optional legacy override; if set with nonzero overlap, it
  must equal `tile_size - overlap`
- `tiling.roi_padding`: manual ROI expansion in level-0 pixels
- `tiling.min_tissue_percentage`: default `70.0`
- `tiling.white_filter.max_white_percentage`: reject nearly empty white tiles
- `saving.resume`: skip existing deterministic tile filenames
- `saving.accepted_tiles_subdir`: accepted native tiles
- `saving.rejected_tiles_subdir`: rejected native tiles
- `saving.filename_prefix`: default `tile`
- `saving.include_component_id`: include `_c1` in filenames when needed
- `metadata.filename`: all-tile metadata CSV
- `visualization.filename`: overview image
- `statistics.filename`: run-level JSON statistics

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with the config:

```bash
python -m tile_generator --config tile_generator_config.yaml
```

Or from Python:

```python
from tile_generator import generate_tiles

results = generate_tiles(
    "tile_generator_config.yaml",
    return_images=True,
    save_to_disk=False,
)

for tile in results["tiles"]:
    image = tile.image
    level0_x, level0_y, width, height = tile.level0_coordinates
    component_id = tile.component_id
```

Override from the command line:

```bash
python -m tile_generator \
  --config tile_generator_config.yaml \
  --slide /data/patient_001.svs \
  --output-dir /data/patient_001_tiles
```

## Outputs

Accepted tile images:

```text
tile_output/accepted_tiles/
```

Rejected tile images:

```text
tile_output/rejected_tiles/
```

Metadata CSV:

```text
tile_output/tile_metadata.csv
tile_output/tile_coordinates.csv
tile_output/tile_summary.csv
```

CSV columns:

```text
tile_filename,tile_id,component_id,level0_x,level0_y,tile_width,tile_height,thumbnail_x,thumbnail_y,tissue_percentage,white_percentage,tile_status,rejection_reason,path,skipped_existing
```

Overview image:

```text
tile_output/patient_tile_map.png
tile_output/tissue_mask.png
```

Statistics JSON:

```text
tile_output/tile_statistics.json
```

The overview shows the thumbnail, tissue mask, connected components in distinct
colors, ROI, accepted tiles in green, rejected tiles in red, level-0 `x,y`
tile labels, and component IDs.

## Synthetic Visualization Example

Generate a documentation example without a real WSI:

```bash
python examples/create_visualization_example.py
```

The script writes:

```text
docs/patient_tile_map_example.png
```

## Batch Processing

Process an entire folder of `.tif` and `.tiff` WSIs in alphabetical order:

```bash
python -m tile_generator \
  --config tile_generator_config.yaml \
  --input-dir raw_images \
  --output-dir output \
  --workers 4
```

For each WSI, the batch runner creates:

```text
output/
  26RR000079-A-01-01/
    tiles/
      accepted_tiles/
      rejected_tiles/
    metadata/
      tile_metadata.csv
      tile_coordinates.csv
      tile_summary.csv
      tile_statistics.json
    visualization/
      patient_tile_map.png
      tissue_mask.png
    logs/
      processing.log
      configuration_snapshot.yaml
    _COMPLETED.json
```

The folder name is the WSI filename without extension. Existing completed WSI
folders are skipped automatically. Use `--force` to remove and regenerate a WSI
output folder.

If a job is interrupted before a WSI finishes, rerun the same command without
`--force`. The runner treats that WSI as partial output, recomputes the expected
tile coordinates, validates existing tile images, skips valid coordinate-named
tiles, regenerates missing or corrupt files, rewrites metadata/visualizations,
and then writes `_COMPLETED.json`.

Relevant resume settings:

```yaml
saving:
  resume: true
  validate_existing_tiles: true
  atomic_writes: true

batch:
  resume_partial: true
```

After all WSIs finish, the runner writes:

```text
output/batch_summary.csv
```

The batch summary has one row per WSI with filename, accepted tiles, rejected
tiles, total tiles, tissue area, processing time, completion timestamp, status,
output folder, and any error message.

Batch mode prints an overall WSI progress bar and a per-WSI tile-generation
progress bar while accepted/rejected tile files are written. With
`--workers > 1`, progress is reported at the WSI-completion level to avoid
multiple processes writing overlapping terminal bars; each slide still writes
its own `logs/processing.log`.

Parallelization is across WSIs, not within a single WSI. Each worker opens its
own OpenSlide handle and writes to a dedicated slide output folder.

## Multiprocessing Notes

The module is multiprocessing-ready because functions are stateless and do not
use global slide handles. For batch processing, parallelize at the WSI level:
each worker should open its own OpenSlide object, run `generate_tiles`, and
close the slide. Avoid sharing OpenSlide objects across processes.

## Backward Compatibility

Downstream inference code should continue consuming the generated image files as
ordinary 512x512 tiles. No YOLO, SAM3.1, CellSeg1, or morphology-analysis code
needs to import this module.

For direct model integration, use `generate_tiles(..., return_images=True,
save_to_disk=False)` and consume `results["tiles"]`. Each `GeneratedTile`
contains the tile image, level-0 coordinates, thumbnail coordinates, tile ID,
component ID, and metadata.
