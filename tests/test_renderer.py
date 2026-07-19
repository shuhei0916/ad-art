from ad_art.matcher import Tile
from ad_art.quadtree import Region
from ad_art.renderer import render_html


def make_tile(tile_id: str) -> Tile:
    return Tile(
        tile_id=tile_id,
        average_color=(0, 0, 0),
        image_url=f"https://example.com/{tile_id}.jpg",
        link_url=f"https://example.com/dp/{tile_id}",
    )


class TestRenderHtml:
    def test_single_leaf_renders_linked_image_with_absolute_position(self):
        placements = [(Region(x=10, y=20, width=30, height=40), make_tile("t1"))]

        html = render_html(placements, canvas_width=100, canvas_height=200)

        assert '<a href="https://example.com/dp/t1"' in html
        assert '<img src="https://example.com/t1.jpg"' in html
        assert "left:10px" in html
        assert "top:20px" in html
        assert "width:30px" in html
        assert "height:40px" in html

    def test_scale_multiplies_all_dimensions(self):
        placements = [(Region(x=10, y=20, width=30, height=40), make_tile("t1"))]

        html = render_html(placements, canvas_width=100, canvas_height=200, scale=2.0)

        assert "left:20px" in html
        assert "top:40px" in html
        assert "width:60px" in html
        assert "height:80px" in html
        assert "width:200px" in html  # コンテナ
        assert "height:400px" in html

    def test_multiple_leaves_render_one_anchor_each(self):
        placements = [
            (Region(x=0, y=0, width=10, height=10), make_tile("t1")),
            (Region(x=10, y=0, width=10, height=10), make_tile("t2")),
            (Region(x=0, y=10, width=20, height=10), make_tile("t3")),
        ]

        html = render_html(placements, canvas_width=20, canvas_height=20)

        assert html.count("<a href=") == 3

    def test_special_characters_in_urls_are_escaped(self):
        tile = Tile(
            tile_id="t1",
            average_color=(0, 0, 0),
            image_url="https://example.com/img.jpg?a=1&b=2",
            link_url="https://example.com/dp/t1?tag=x&id=<y>",
        )
        placements = [(Region(x=0, y=0, width=10, height=10), tile)]

        html = render_html(placements, canvas_width=10, canvas_height=10)

        assert "a=1&amp;b=2" in html
        assert "tag=x&amp;id=&lt;y&gt;" in html
        assert "id=<y>" not in html

    def test_output_is_complete_html_document(self):
        placements = [(Region(x=0, y=0, width=10, height=10), make_tile("t1"))]

        html = render_html(placements, canvas_width=10, canvas_height=10)

        assert html.startswith("<!doctype html>")
        assert "</html>" in html
