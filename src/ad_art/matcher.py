from dataclasses import dataclass

import numpy as np

from ad_art.quadtree import Region, average_color


@dataclass(frozen=True)
class Tile:
    tile_id: str
    average_color: tuple[int, int, int]
    image_url: str
    link_url: str


class TileMatcher:
    def __init__(self, tiles: list[Tile]):
        if not tiles:
            raise ValueError("tiles must not be empty")
        self._tiles = list(tiles)

    def match(
        self,
        color: tuple[int, int, int],
        k: int = 1,
        rng: np.random.Generator | None = None,
    ) -> Tile:
        """colorに最も近い素材を返す。k>1なら上位k件からランダムに選ぶ。"""
        if k <= 1 or rng is None:
            return min(self._tiles, key=lambda t: _distance_sq(t.average_color, color))
        nearest = sorted(
            self._tiles, key=lambda t: _distance_sq(t.average_color, color)
        )[:k]
        return nearest[int(rng.integers(len(nearest)))]


def _distance_sq(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def tile_from_image(
    image: np.ndarray, tile_id: str, image_url: str, link_url: str
) -> Tile:
    region = Region(x=0, y=0, width=image.shape[1], height=image.shape[0])
    return Tile(
        tile_id=tile_id,
        average_color=average_color(image, region),
        image_url=image_url,
        link_url=link_url,
    )
