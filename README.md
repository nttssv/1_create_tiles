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
  --workers 4
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
Parallel workers run at the WSI level; each worker opens its own slide and
writes to a dedicated output folder.

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
