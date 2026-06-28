# WSI Tile Generation

Standalone tile-generation stage for Whole Slide Image inference pipelines.

This code keeps downstream inference inputs unchanged: accepted tiles are
native OpenSlide level-0 crops, 512x512 pixels by default, with no resizing or
downsampling before inference.

## Files

- `tile_generator.py`: implementation and CLI
- `tile_generator_config.yaml`: configurable defaults
- `examples/run_tile_generator.py`: real-slide usage
- `examples/create_visualization_example.py`: synthetic visualization demo
- `docs/tile_generator.md`: detailed pipeline documentation
- `docs/patient_tile_map_example.png`: generated overview example

## Quick Start

```bash
pip install -r requirements.txt
python -m tile_generator --config tile_generator_config.yaml --slide /path/to/patient.svs
```

Slide reading defaults to `slide_backend: auto`: OpenSlide is tried first, then
`tiffslide` is used for generic TIFF/OME-TIFF files that OpenSlide reports as
unsupported.

Outputs are written under `tile_output/` by default:

- `accepted_tiles/`: accepted native 512x512 level-0 crops named `tile_x18432_y22528.png`
- `rejected_tiles/`: rejected native 512x512 level-0 crops
- `tile_metadata.csv`: one row per generated tile
- `patient_tile_map.png`: overview of mask, components, ROI, and tile filtering
- `tile_statistics.json`: run-level counts and timing

Batch mode:

```bash
python -m tile_generator \
  --config tile_generator_config.yaml \
  --input-dir raw_images \
  --output-dir output \
  --workers 4 \
  --slide-backend auto \
  --progress-every-tiles 100 \
  --progress-interval-seconds 2
```

Each WSI gets its own folder:

```text
output/26RR000079-A-01-01/
  tiles/
  metadata/
  visualization/
  logs/
```

Use `--force` to overwrite a completed WSI output. Otherwise, completed slides
are skipped automatically and `output/batch_summary.csv` is updated.
The batch summary includes each source WSI file size in bytes, MB, and GB.
Parallel workers run at the WSI level; each worker opens its own slide and
writes to a dedicated output folder.
The terminal progress display updates while tiles are saved or skipped,
including the last coordinate-named tile file handled by each active worker.

If a run stops mid-slide, run the same command again without `--force`. The
batch runner will re-enter the partial slide folder, validate existing tile
images, skip valid files, regenerate missing/corrupt files, and then write the
completion marker.

QC after batch:

```bash
python -m tile_generator \
  --qc-output-dir output \
  --qc-sample-size 25 \
  --qc-tile-size 512
```

This writes `output/batch_qc_summary.csv` and `output/batch_qc_report.json`.
QC checks completion markers, metadata/file count consistency, coordinate-based
filenames, duplicate tile IDs, required visualization/log files, and sampled
tile image sizes.

Rename unreadable input WSIs after QC:

```bash
python examples/rename_unreadable_wsi.py \
  --input-dir raw_images \
  --batch-summary output/batch_summary.csv

python examples/rename_unreadable_wsi.py \
  --input-dir raw_images \
  --batch-summary output/batch_summary.csv \
  --apply
```

The first command is a dry run. The second renames files unreadable by both
OpenSlide and tiffslide, for example `MTO107_CORRUPT.tiff.corrupt`, so batch
mode will skip them automatically.
If a file can be opened but still crashes during batch tiling, rename every
remaining `Failed` row from `batch_summary.csv`:

```bash
python examples/rename_unreadable_wsi.py \
  --input-dir raw_images \
  --batch-summary output/batch_summary.csv \
  --rename-all-batch-failed \
  --apply
```

## Direct API

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
```

The grid is deterministic and anchored to the effective ROI origin. Configure
overlap with `tiling.overlap`; the stride is `tile_size - overlap`.

Coordinate utilities:

```python
from tile_generator import coordinate_to_filename, filename_to_coordinate

name = coordinate_to_filename(18432, 22528)
x, y = filename_to_coordinate(name)
```
