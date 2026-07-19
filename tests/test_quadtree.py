import numpy as np

from ad_art.quadtree import (
    Region,
    average_color,
    build_quadtree,
    color_variance,
    leaf_regions,
    subdivide,
)


def solid_image(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


class TestAverageColor:
    def test_solid_color_region_returns_that_color(self):
        image = solid_image(8, 8, (200, 100, 50))
        region = Region(x=0, y=0, width=8, height=8)

        assert average_color(image, region) == (200, 100, 50)

    def test_region_covers_only_part_of_image(self):
        image = solid_image(8, 8, (0, 0, 0))
        image[0:4, 0:4] = (255, 255, 255)  # 左上だけ白
        region = Region(x=0, y=0, width=4, height=4)

        assert average_color(image, region) == (255, 255, 255)

    def test_mixed_colors_are_averaged(self):
        image = solid_image(2, 1, (0, 0, 0))
        image[0, 1] = (255, 255, 255)
        region = Region(x=0, y=0, width=2, height=1)

        assert average_color(image, region) == (128, 128, 128)


class TestColorVariance:
    def test_solid_region_has_zero_variance(self):
        image = solid_image(8, 8, (10, 20, 30))
        region = Region(x=0, y=0, width=8, height=8)

        assert color_variance(image, region) == 0.0

    def test_high_contrast_region_has_large_variance(self):
        image = solid_image(8, 8, (0, 0, 0))
        image[:, 4:] = (255, 255, 255)  # 右半分が白
        region = Region(x=0, y=0, width=8, height=8)

        assert color_variance(image, region) > 100.0


class TestSubdivide:
    def test_even_region_splits_into_four_quadrants(self):
        region = Region(x=0, y=0, width=8, height=6)

        assert subdivide(region) == [
            Region(x=0, y=0, width=4, height=3),  # 左上
            Region(x=4, y=0, width=4, height=3),  # 右上
            Region(x=0, y=3, width=4, height=3),  # 左下
            Region(x=4, y=3, width=4, height=3),  # 右下
        ]

    def test_odd_region_splits_without_gaps_or_overlaps(self):
        region = Region(x=2, y=1, width=5, height=7)

        children = subdivide(region)

        assert children == [
            Region(x=2, y=1, width=2, height=3),
            Region(x=4, y=1, width=3, height=3),
            Region(x=2, y=4, width=2, height=4),
            Region(x=4, y=4, width=3, height=4),
        ]
        assert sum(c.width * c.height for c in children) == 5 * 7


class TestBuildQuadtree:
    def test_solid_image_yields_single_leaf(self):
        image = solid_image(16, 16, (100, 150, 200))

        root = build_quadtree(image, threshold=10.0, min_size=1, max_depth=8)

        assert root.children == []
        assert root.region == Region(x=0, y=0, width=16, height=16)
        assert root.color == (100, 150, 200)

    def test_four_distinct_quadrants_yield_four_leaves(self):
        image = solid_image(16, 16, (0, 0, 0))
        image[0:8, 8:16] = (255, 0, 0)
        image[8:16, 0:8] = (0, 255, 0)
        image[8:16, 8:16] = (0, 0, 255)

        root = build_quadtree(image, threshold=10.0, min_size=1, max_depth=8)

        assert len(root.children) == 4
        assert all(child.children == [] for child in root.children)
        assert [c.color for c in root.children] == [
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]

    def test_min_size_stops_subdivision(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        root = build_quadtree(image, threshold=0.0, min_size=8, max_depth=8)

        # 8x8 の葉が4つで止まる(それ以上分割すると min_size を割る)
        leaves = leaf_regions(root)
        assert len(leaves) == 4
        assert all(leaf.width == 8 and leaf.height == 8 for leaf in leaves)

    def test_max_depth_stops_subdivision(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        root = build_quadtree(image, threshold=0.0, min_size=1, max_depth=1)

        leaves = leaf_regions(root)
        assert len(leaves) == 4

    def test_leaves_tile_the_whole_image(self):
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(20, 12, 3), dtype=np.uint8)

        root = build_quadtree(image, threshold=30.0, min_size=2, max_depth=6)

        leaves = leaf_regions(root)
        assert sum(leaf.width * leaf.height for leaf in leaves) == 20 * 12
