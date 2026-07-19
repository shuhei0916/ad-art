import numpy as np

from ad_art.matcher import Tile
from ad_art.rakuten import (
    RakutenClient,
    build_catalog,
    load_catalog,
    save_catalog,
    tiles_from_response,
    with_image_size,
)


def item(**overrides) -> dict:
    base = {
        "itemCode": "shop:10000001",
        "itemName": "テスト商品",
        "itemUrl": "https://item.rakuten.co.jp/shop/10000001/",
        "affiliateUrl": "https://hb.afl.rakuten.co.jp/hgc/xxxx/?pc=https%3A%2F%2Fitem...",
        "mediumImageUrls": [
            {"imageUrl": "https://thumbnail.image.rakuten.co.jp/@0_mall/shop/img/a.jpg?_ex=128x128"}
        ],
    }
    base.update(overrides)
    return {"Item": base}


class TestTilesFromResponse:
    def test_items_are_converted_to_tiles(self):
        response = {"Items": [item()]}

        tiles = tiles_from_response(response)

        assert len(tiles) == 1
        assert tiles[0].tile_id == "shop:10000001"
        assert tiles[0].image_url.startswith(
            "https://thumbnail.image.rakuten.co.jp/@0_mall/shop/img/a.jpg"
        )
        assert tiles[0].link_url.startswith("https://hb.afl.rakuten.co.jp/")

    def test_missing_affiliate_url_falls_back_to_item_url(self):
        response = {"Items": [item(affiliateUrl="")]}

        tiles = tiles_from_response(response)

        assert tiles[0].link_url == "https://item.rakuten.co.jp/shop/10000001/"

    def test_item_without_image_is_skipped(self):
        response = {"Items": [item(mediumImageUrls=[]), item(itemCode="shop:2")]}

        tiles = tiles_from_response(response)

        assert [t.tile_id for t in tiles] == ["shop:2"]


class TestRakutenClient:
    def test_search_sends_credentials_and_keyword(self):
        calls = []

        def fake_fetch(params: dict) -> dict:
            calls.append(params)
            return {"Items": [item()], "pageCount": 1}

        client = RakutenClient(
            app_id="APP", access_key="pk_KEY", affiliate_id="AFF", fetch=fake_fetch
        )
        tiles = client.search("コーヒー", max_pages=1)

        assert len(tiles) == 1
        assert calls[0]["applicationId"] == "APP"
        assert calls[0]["accessKey"] == "pk_KEY"
        assert calls[0]["affiliateId"] == "AFF"
        assert calls[0]["keyword"] == "コーヒー"
        assert calls[0]["page"] == 1

    def test_search_aggregates_multiple_pages(self):
        def fake_fetch(params: dict) -> dict:
            code = f"shop:{params['page']}"
            return {"Items": [item(itemCode=code)], "pageCount": 3}

        client = RakutenClient(
            app_id="APP", access_key="pk_KEY", affiliate_id="AFF", fetch=fake_fetch
        )
        tiles = client.search("本", max_pages=3)

        assert [t.tile_id for t in tiles] == ["shop:1", "shop:2", "shop:3"]

    def test_search_stops_at_page_count(self):
        def fake_fetch(params: dict) -> dict:
            return {"Items": [item(itemCode=f"shop:{params['page']}")], "pageCount": 2}

        client = RakutenClient(
            app_id="APP", access_key="pk_KEY", affiliate_id="AFF", fetch=fake_fetch
        )
        tiles = client.search("本", max_pages=10)

        assert len(tiles) == 2

    def test_duplicate_item_codes_are_deduplicated(self):
        def fake_fetch(params: dict) -> dict:
            return {"Items": [item(), item()], "pageCount": 1}

        client = RakutenClient(
            app_id="APP", access_key="pk_KEY", affiliate_id="AFF", fetch=fake_fetch
        )
        tiles = client.search("本", max_pages=1)

        assert len(tiles) == 1


class TestBuildCatalog:
    def test_average_color_is_filled_from_downloaded_image(self):
        tile = Tile(
            tile_id="shop:1",
            average_color=(0, 0, 0),
            image_url="https://example.com/a.jpg",
            link_url="https://example.com/dp/1",
        )

        def fake_download(url: str) -> np.ndarray:
            return np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)

        catalog = build_catalog([tile], download=fake_download)

        assert catalog[0].average_color == (10, 20, 30)

    def test_failed_download_skips_tile(self):
        tiles = [
            Tile("shop:1", (0, 0, 0), "https://example.com/a.jpg", "https://example.com/1"),
            Tile("shop:2", (0, 0, 0), "https://example.com/b.jpg", "https://example.com/2"),
        ]

        def fake_download(url: str) -> np.ndarray:
            if "a.jpg" in url:
                raise OSError("download failed")
            return np.full((4, 4, 3), (1, 2, 3), dtype=np.uint8)

        catalog = build_catalog(tiles, download=fake_download)

        assert [t.tile_id for t in catalog] == ["shop:2"]


class TestCatalogPersistence:
    def test_catalog_roundtrips_through_json(self, tmp_path):
        tiles = [
            Tile("shop:1", (10, 20, 30), "https://example.com/a.jpg", "https://example.com/1"),
            Tile("shop:2", (40, 50, 60), "https://example.com/b.jpg", "https://example.com/2"),
        ]
        path = tmp_path / "catalog.json"

        save_catalog(tiles, path)

        assert load_catalog(path) == tiles


class TestWithImageSize:
    def test_ex_parameter_is_replaced(self):
        url = "https://thumbnail.image.rakuten.co.jp/img/a.jpg?_ex=128x128"

        assert with_image_size(url, 64) == (
            "https://thumbnail.image.rakuten.co.jp/img/a.jpg?_ex=64x64"
        )

    def test_url_without_ex_parameter_gets_one(self):
        url = "https://thumbnail.image.rakuten.co.jp/img/a.jpg"

        assert with_image_size(url, 64) == (
            "https://thumbnail.image.rakuten.co.jp/img/a.jpg?_ex=64x64"
        )
