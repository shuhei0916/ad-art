from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int


@dataclass
class QuadNode:
    region: Region
    color: tuple[int, int, int]
    children: list["QuadNode"] = field(default_factory=list)


def build_quadtree(
    image: np.ndarray,
    threshold: float,
    min_size: int,
    max_depth: int,
) -> QuadNode:
    height, width = image.shape[:2]
    root_region = Region(x=0, y=0, width=width, height=height)
    return _build_node(image, root_region, threshold, min_size, max_depth, depth=0)


def _build_node(
    image: np.ndarray,
    region: Region,
    threshold: float,
    min_size: int,
    max_depth: int,
    depth: int,
) -> QuadNode:
    node = QuadNode(region=region, color=average_color(image, region))
    can_split = (
        depth < max_depth
        and region.width // 2 >= min_size
        and region.height // 2 >= min_size
    )
    if can_split and color_variance(image, region) > threshold:
        node.children = [
            _build_node(image, child, threshold, min_size, max_depth, depth + 1)
            for child in subdivide(region)
        ]
    return node


def leaf_regions(node: QuadNode) -> list[Region]:
    if not node.children:
        return [node.region]
    return [leaf for child in node.children for leaf in leaf_regions(child)]


def _pixels(image: np.ndarray, region: Region) -> np.ndarray:
    return image[region.y : region.y + region.height, region.x : region.x + region.width]


def subdivide(region: Region) -> list[Region]:
    """領域を左上・右上・左下・右下の4つに分割する。奇数辺は右・下側が大きくなる。"""
    left_w = region.width // 2
    right_w = region.width - left_w
    top_h = region.height // 2
    bottom_h = region.height - top_h
    mid_x = region.x + left_w
    mid_y = region.y + top_h
    return [
        Region(region.x, region.y, left_w, top_h),
        Region(mid_x, region.y, right_w, top_h),
        Region(region.x, mid_y, left_w, bottom_h),
        Region(mid_x, mid_y, right_w, bottom_h),
    ]


def color_variance(image: np.ndarray, region: Region) -> float:
    """領域内の色分散。RGB各チャンネルの標準偏差の平均を返す。"""
    pixels = _pixels(image, region).reshape(-1, 3).astype(np.float64)
    return float(pixels.std(axis=0).mean())


def average_color(image: np.ndarray, region: Region) -> tuple[int, int, int]:
    mean = _pixels(image, region).reshape(-1, 3).mean(axis=0)
    return tuple(int(round(c)) for c in mean)
