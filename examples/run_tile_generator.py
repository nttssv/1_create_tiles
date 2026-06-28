"""Example: run standalone tile generation on a real OpenSlide WSI."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tile_generator import generate_tiles


CONFIG_PATH = REPO_ROOT / "tile_generator_config.yaml"


def main() -> None:
    results = generate_tiles(CONFIG_PATH, return_images=False, save_to_disk=True)
    print(f"accepted_tiles={len(results['accepted_tiles'])}")
    print(f"metadata={results['metadata_path']}")
    print(f"visualization={results['visualization_path']}")
    print(f"statistics={results['statistics_path']}")


if __name__ == "__main__":
    main()
