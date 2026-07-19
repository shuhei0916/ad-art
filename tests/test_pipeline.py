import numpy as np

from ad_art.matcher import Tile
from ad_art.pipeline import generate_mosaic


def make_tile(tile_id: str, color: tuple[int, int, int]) -> Tile:
    return Tile(
        tile_id=tile_id,
        average_color=color,
        image_url=f"https://example.com/{tile_id}.jpg",
        link_url=f"https://example.com/dp/{tile_id}",
    )


class TestGenerateMosaic:
    def test_quadrant_colors_are_matched_to_nearest_tiles(self):
        # 4象限が異なる単色の画像 → それぞれ最も近い色の素材が選ばれる
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[0:8, 8:16] = (250, 0, 0)
        image[8:16, 0:8] = (0, 250, 0)
        image[8:16, 8:16] = (0, 0, 250)
        tiles = [
            make_tile("black", (0, 0, 0)),
            make_tile("red", (255, 0, 0)),
            make_tile("green", (0, 255, 0)),
            make_tile("blue", (0, 0, 255)),
        ]

        html = generate_mosaic(
            image, tiles, threshold=10.0, min_size=1, max_depth=8, scale=1.0
        )

        for tile_id in ["black", "red", "green", "blue"]:
            assert f"https://example.com/dp/{tile_id}" in html

    def test_output_contains_one_anchor_per_leaf(self):
        image = np.full((16, 16, 3), (100, 100, 100), dtype=np.uint8)  # 単色 → 葉1つ
        tiles = [make_tile("gray", (100, 100, 100))]

        html = generate_mosaic(
            image, tiles, threshold=10.0, min_size=1, max_depth=8, scale=1.0
        )

        assert html.count("<a href=") == 1
