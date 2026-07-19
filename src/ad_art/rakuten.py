import dataclasses
import io
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from ad_art.matcher import Tile
from ad_art.quadtree import Region, average_color

# 2026年2月のインフラ刷新後の新エンドポイント(旧 app.rakuten.co.jp は2026-05-14に停止)
API_URL = "https://openapi.rakuten.co.jp/ichibams/api/IchibaItem/Search/20220601"
# Webアプリケーション型はReferer/Originが「許可されたWebサイト」と一致する必要がある
REFERER = "https://shuhei0916.github.io/"


def _http_fetch(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"Referer": REFERER, "Origin": REFERER.rstrip("/")},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


class RakutenClient:
    def __init__(
        self,
        app_id: str,
        access_key: str,
        affiliate_id: str,
        fetch: Callable[[dict], dict] = _http_fetch,
    ):
        self._app_id = app_id
        self._access_key = access_key
        self._affiliate_id = affiliate_id
        self._fetch = fetch

    def search(self, keyword: str, max_pages: int = 1) -> list[Tile]:
        tiles: dict[str, Tile] = {}
        page = 1
        while page <= max_pages:
            response = self._fetch(
                {
                    "applicationId": self._app_id,
                    "accessKey": self._access_key,
                    "affiliateId": self._affiliate_id,
                    "keyword": keyword,
                    "hits": 30,
                    "page": page,
                }
            )
            for tile in tiles_from_response(response):
                tiles.setdefault(tile.tile_id, tile)
            if page >= response.get("pageCount", 1):
                break
            page += 1
        return list(tiles.values())

# 平均色は画像ダウンロード後に catalog 側で計算して埋める
_PLACEHOLDER_COLOR = (0, 0, 0)


def tiles_from_response(response: dict) -> list[Tile]:
    """楽天市場商品検索APIのレスポンスを Tile のリストに変換する。画像なし商品は除外。"""
    tiles = []
    for entry in response.get("Items", []):
        item = entry["Item"]
        images = item.get("mediumImageUrls", [])
        if not images:
            continue
        tiles.append(
            Tile(
                tile_id=item["itemCode"],
                average_color=_PLACEHOLDER_COLOR,
                image_url=images[0]["imageUrl"],
                link_url=item.get("affiliateUrl") or item["itemUrl"],
            )
        )
    return tiles


def _http_download_image(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as response:
        return np.asarray(Image.open(io.BytesIO(response.read())).convert("RGB"))


def build_catalog(
    tiles: list[Tile],
    download: Callable[[str], np.ndarray] = _http_download_image,
) -> list[Tile]:
    """各素材の画像をダウンロードして平均色を埋める。取得失敗した素材は除外。"""
    catalog = []
    for tile in tiles:
        try:
            image = download(tile.image_url)
        except OSError:
            continue
        region = Region(x=0, y=0, width=image.shape[1], height=image.shape[0])
        catalog.append(dataclasses.replace(tile, average_color=average_color(image, region)))
    return catalog


def save_catalog(tiles: list[Tile], path: Path) -> None:
    data = [dataclasses.asdict(tile) for tile in tiles]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def load_catalog(path: Path) -> list[Tile]:
    return [
        Tile(
            tile_id=entry["tile_id"],
            average_color=tuple(entry["average_color"]),
            image_url=entry["image_url"],
            link_url=entry["link_url"],
        )
        for entry in json.loads(path.read_text())
    ]


def with_image_size(url: str, size: int) -> str:
    """楽天サムネイルURLの _ex=WxH パラメータを指定サイズに揃える。"""
    base = re.sub(r"([?&])_ex=\d+x\d+", r"\1", url).rstrip("?&")
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}_ex={size}x{size}"
