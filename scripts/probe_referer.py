"""許可されたRefererを特定するための接続プローブ。.envから認証情報を読む。"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_mosaic import load_env

from ad_art.rakuten import API_URL

REFERERS = [
    ("https://localhost/", "https://localhost"),
    ("https://github.com/", "https://github.com"),
    ("https://shuhei0916.github.io/", "https://shuhei0916.github.io"),
    ("https://www.rakuten.co.jp/", "https://www.rakuten.co.jp"),
]


def main() -> None:
    env = load_env(Path(".env"))
    params = {
        "applicationId": env["RAKUTEN_APP_ID"],
        "accessKey": env["RAKUTEN_ACCESS_KEY"],
        "affiliateId": env["RAKUTEN_AFFILIATE_ID"],
        "keyword": "コーヒー",
        "hits": 1,
        "page": 1,
        "format": "json",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    for referer, origin in REFERERS:
        headers = {"Referer": referer, "Origin": origin}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                data = json.load(response)
                print(f"OK   referer={referer} count={data.get('count')}")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:120]
            print(f"NG   referer={referer} HTTP {e.code} {body}")
        time.sleep(1.6)


if __name__ == "__main__":
    main()
