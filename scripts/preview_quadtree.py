"""四分木分割のプレビュー: 葉領域を平均色で塗った画像を出力する。

Usage: uv run python scripts/preview_quadtree.py <image> [threshold] [min_size] [max_depth]
"""

import sys

import numpy as np
from PIL import Image

from ad_art.quadtree import average_color, build_quadtree, leaf_regions


def render_preview(
    image: np.ndarray, threshold: float, min_size: int, max_depth: int
) -> tuple[Image.Image, int]:
    root = build_quadtree(image, threshold=threshold, min_size=min_size, max_depth=max_depth)
    leaves = leaf_regions(root)
    out = np.zeros_like(image)
    for leaf in leaves:
        out[leaf.y : leaf.y + leaf.height, leaf.x : leaf.x + leaf.width] = average_color(
            image, leaf
        )
        # 境界線でパッチワーク感を可視化
        out[leaf.y, leaf.x : leaf.x + leaf.width] = (255, 255, 255)
        out[leaf.y : leaf.y + leaf.height, leaf.x] = (255, 255, 255)
    return Image.fromarray(out), len(leaves)


def main() -> None:
    path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    min_size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    max_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    image = np.asarray(Image.open(path).convert("RGB"))

    preview, leaf_count = render_preview(image, threshold, min_size, max_depth)
    out_path = f"preview_t{threshold:g}_s{min_size}_d{max_depth}.png"
    preview.save(out_path)
    print(f"{out_path}: {leaf_count} leaves")


if __name__ == "__main__":
    main()
