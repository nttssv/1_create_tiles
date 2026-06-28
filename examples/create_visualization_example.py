"""Create a synthetic tile-map example without requiring OpenSlide."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tile_generator import (
    ROI,
    clean_mask,
    create_thumbnail,
    create_tissue_mask,
    extract_connected_components,
    filter_tiles,
    generate_candidate_tiles,
    visualize_tiles,
)


class SyntheticSlide:
    """Small OpenSlide-like object for documentation and smoke testing."""

    def __init__(self) -> None:
        self.dimensions = (4096, 2048)
        image = Image.new("RGB", self.dimensions, "white")
        draw = ImageDraw.Draw(image)
        draw.ellipse((320, 260, 2180, 1720), fill=(204, 116, 172), outline=(110, 50, 120), width=12)
        draw.ellipse((2280, 520, 3660, 1680), fill=(160, 86, 150), outline=(90, 42, 110), width=12)
        draw.rectangle((1650, 900, 2600, 1300), fill=(230, 185, 210))
        self._image = image

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        thumb = self._image.copy()
        thumb.thumbnail(size)
        return thumb

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Image.Image:
        if level != 0:
            raise ValueError("SyntheticSlide only supports level 0.")
        x, y = location
        return self._image.crop((x, y, x + size[0], y + size[1])).convert("RGBA")


def main() -> None:
    output_path = REPO_ROOT / "docs" / "patient_tile_map_example.png"
    slide = SyntheticSlide()
    thumbnail = create_thumbnail(slide, max_size=(1024, 512))
    tissue_mask = create_tissue_mask(thumbnail.image)
    cleaned_mask = clean_mask(
        tissue_mask,
        {
            "min_object_area": 32,
            "fill_holes": True,
            "opening_radius": 1,
            "closing_radius": 2,
        },
    )
    label_image, components = extract_connected_components(
        cleaned_mask,
        thumbnail,
        min_area=128,
    )
    candidates = generate_candidate_tiles(
        slide_dimensions=slide.dimensions,
        thumbnail=thumbnail,
        label_image=label_image,
        components=components,
        tile_size=512,
        stride=512,
        roi_mode="auto",
    )
    accepted, rejected = filter_tiles(
        candidates,
        label_image=label_image,
        thumbnail=thumbnail,
        min_tissue_percentage=70.0,
    )
    visualize_tiles(
        thumbnail=thumbnail,
        tissue_mask=cleaned_mask,
        label_image=label_image,
        components=components,
        accepted_tiles=accepted,
        rejected_tiles=rejected,
        output_path=output_path,
        roi=ROI(0, 0, *slide.dimensions),
        roi_mode="auto",
    )
    print(output_path)


if __name__ == "__main__":
    main()
