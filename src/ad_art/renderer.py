from html import escape

from ad_art.matcher import Tile
from ad_art.quadtree import Region


def render_html(
    placements: list[tuple[Region, Tile]],
    canvas_width: int,
    canvas_height: int,
    scale: float = 1.0,
) -> str:
    anchors = "\n".join(_render_anchor(region, tile, scale) for region, tile in placements)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ad-art mosaic</title>
<style>
.mosaic {{ position: relative; }}
.mosaic a {{ position: absolute; display: block; }}
.mosaic img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
</style>
</head>
<body>
<div class="mosaic" style="width:{round(canvas_width * scale)}px;height:{round(canvas_height * scale)}px">
{anchors}
</div>
</body>
</html>
"""


def _render_anchor(region: Region, tile: Tile, scale: float) -> str:
    style = (
        f"left:{round(region.x * scale)}px;"
        f"top:{round(region.y * scale)}px;"
        f"width:{round(region.width * scale)}px;"
        f"height:{round(region.height * scale)}px"
    )
    return (
        f'<a href="{escape(tile.link_url)}" style="{style}" target="_blank" rel="nofollow noopener">'
        f'<img src="{escape(tile.image_url)}" alt="{escape(tile.tile_id)}" loading="lazy">'
        f"</a>"
    )
