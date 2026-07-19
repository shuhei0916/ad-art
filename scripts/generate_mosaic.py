"""楽天APIで素材を集めてモザイクHTMLを生成する。

Usage:
  uv run python scripts/generate_mosaic.py <image> <keyword...> [--out mosaic.html]

.env に RAKUTEN_APP_ID / RAKUTEN_AFFILIATE_ID が必要。
カタログは catalog.json にキャッシュされ、2回目以降はAPIを呼ばない。
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ad_art.pipeline import generate_mosaic
from ad_art.rakuten import (
    RakutenClient,
    _http_fetch,
    build_catalog,
    load_catalog,
    save_catalog,
)


def load_env(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def rate_limited_fetch(params: dict) -> dict:
    time.sleep(1.0)  # 楽天ウェブサービスのQPS制限(1req/s)対策
    return _http_fetch(params)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("keywords", nargs="+")
    parser.add_argument("--out", default="mosaic.html")
    parser.add_argument("--catalog", default="catalog.json")
    parser.add_argument("--threshold", type=float, default=15.0)
    parser.add_argument("--min-size", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if catalog_path.exists():
        tiles = load_catalog(catalog_path)
        print(f"カタログをキャッシュから読み込み: {len(tiles)}素材")
    else:
        env = load_env(Path(".env"))
        client = RakutenClient(
            app_id=env["RAKUTEN_APP_ID"],
            affiliate_id=env["RAKUTEN_AFFILIATE_ID"],
            fetch=rate_limited_fetch,
        )
        raw_tiles = []
        for keyword in args.keywords:
            found = client.search(keyword, max_pages=args.max_pages)
            print(f"検索 '{keyword}': {len(found)}素材")
            raw_tiles.extend(found)
        tiles = build_catalog(raw_tiles)
        save_catalog(tiles, catalog_path)
        print(f"カタログ構築完了: {len(tiles)}素材 → {catalog_path}")

    image = np.asarray(Image.open(args.image).convert("RGB"))
    html = generate_mosaic(
        image,
        tiles,
        threshold=args.threshold,
        min_size=args.min_size,
        max_depth=args.max_depth,
        scale=args.scale,
        k=args.k,
        rng=np.random.default_rng(0),
    )
    Path(args.out).write_text(html)
    print(f"生成完了: {args.out}")


if __name__ == "__main__":
    main()
