import numpy as np

import pytest

from ad_art.matcher import Tile, TileMatcher, tile_from_image


def make_tile(tile_id: str, color: tuple[int, int, int]) -> Tile:
    return Tile(
        tile_id=tile_id,
        average_color=color,
        image_url=f"https://example.com/{tile_id}.jpg",
        link_url=f"https://example.com/dp/{tile_id}",
    )


def solid_image(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


class TestTileFromImage:
    def test_average_color_is_computed_from_image(self):
        image = solid_image(4, 4, (200, 100, 50))

        tile = tile_from_image(
            image,
            tile_id="B000TEST01",
            image_url="https://example.com/img.jpg",
            link_url="https://example.com/dp/B000TEST01?tag=mytag",
        )

        assert tile == Tile(
            tile_id="B000TEST01",
            average_color=(200, 100, 50),
            image_url="https://example.com/img.jpg",
            link_url="https://example.com/dp/B000TEST01?tag=mytag",
        )


class TestTileMatcher:
    def test_single_tile_always_matches(self):
        only = make_tile("only", (0, 0, 0))
        matcher = TileMatcher([only])

        assert matcher.match((255, 255, 255)) == only

    def test_nearest_color_wins(self):
        red = make_tile("red", (255, 0, 0))
        green = make_tile("green", (0, 255, 0))
        blue = make_tile("blue", (0, 0, 255))
        matcher = TileMatcher([red, green, blue])

        assert matcher.match((200, 30, 30)) == red
        assert matcher.match((10, 240, 10)) == green
        assert matcher.match((30, 30, 200)) == blue

    def test_tie_is_broken_deterministically_by_input_order(self):
        first = make_tile("first", (100, 0, 0))
        second = make_tile("second", (0, 0, 100))  # (50,0,50)から等距離
        matcher = TileMatcher([first, second])

        assert matcher.match((50, 0, 50)) == first

    def test_empty_tiles_raise_value_error(self):
        with pytest.raises(ValueError):
            TileMatcher([])


class TestMatchTopK:
    def test_k1_is_equivalent_to_nearest(self):
        red = make_tile("red", (255, 0, 0))
        blue = make_tile("blue", (0, 0, 255))
        matcher = TileMatcher([red, blue])
        rng = np.random.default_rng(0)

        assert matcher.match((250, 10, 10), k=1, rng=rng) == red

    def test_result_is_always_among_k_nearest(self):
        tiles = [
            make_tile("black", (0, 0, 0)),
            make_tile("dark", (30, 30, 30)),
            make_tile("gray", (60, 60, 60)),
            make_tile("white", (255, 255, 255)),
        ]
        matcher = TileMatcher(tiles)
        rng = np.random.default_rng(0)

        results = {matcher.match((10, 10, 10), k=3, rng=rng).tile_id for _ in range(50)}

        assert results <= {"black", "dark", "gray"}  # whiteは選ばれない
        assert len(results) > 1  # ランダム性でバリエーションが出る

    def test_same_seed_gives_same_sequence(self):
        tiles = [make_tile(f"t{i}", (i * 10, i * 10, i * 10)) for i in range(5)]
        matcher = TileMatcher(tiles)

        seq1 = [
            matcher.match((25, 25, 25), k=3, rng=np.random.default_rng(42)).tile_id
            for _ in range(10)
        ]
        seq2 = [
            matcher.match((25, 25, 25), k=3, rng=np.random.default_rng(42)).tile_id
            for _ in range(10)
        ]

        assert seq1 == seq2
