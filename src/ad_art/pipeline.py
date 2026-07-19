import numpy as np

from ad_art.matcher import Tile, TileMatcher
from ad_art.quadtree import average_color, build_quadtree, leaf_regions
from ad_art.renderer import render_html


def generate_mosaic(
    image: np.ndarray,
    tiles: list[Tile],
    threshold: float,
    min_size: int,
    max_depth: int,
    scale: float = 1.0,
    k: int = 1,
    rng: np.random.Generator | None = None,
) -> str:
    """目標画像と素材カタログからモザイクHTMLを生成する。"""
    root = build_quadtree(image, threshold=threshold, min_size=min_size, max_depth=max_depth)
    matcher = TileMatcher(tiles)
    placements = [
        (leaf, matcher.match(average_color(image, leaf), k=k, rng=rng))
        for leaf in leaf_regions(root)
    ]
    return render_html(
        placements,
        canvas_width=image.shape[1],
        canvas_height=image.shape[0],
        scale=scale,
    )
