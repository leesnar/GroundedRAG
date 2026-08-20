"""Download the source PDFs listed in SOURCES.md into data/raw/.

The PDFs themselves aren't committed to the repo (92MB, and they're FAO/CGIAR
publications we don't want to redistribute) -- this script re-fetches them from
their original open-access URLs so `python src/ingest.py` can be reproduced.
"""

from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parent / "raw"

SOURCES = {
    "fao_smallscale_rainbow_trout.pdf": "https://www.fao.org/4/i2125e/i2125e.pdf",
    "fao_water_quality_fish_health.pdf": "https://openknowledge.fao.org/server/api/core/bitstreams/185abd2a-fe7d-49dc-86ff-a6a1174566c7/content",
    "fao_tilapia_pond_farming_ghana.pdf": "https://cgspace.cgiar.org/bitstreams/3e09f24e-2ee3-448c-96ec-4d0e6b392416/download",
    "fao_smallscale_aquaponics.pdf": "https://openknowledge.fao.org/server/api/core/bitstreams/2ca21047-390f-42cd-bd1d-0c2ebc9c1df2/content",
    "fao_sustainable_aquaculture_training.pdf": "https://openknowledge.fao.org/server/api/core/bitstreams/dee7e1c8-11fa-4365-9e7d-ac8bbc551265/content",
    "fao_tilapia_feed_management_africa.pdf": "https://www.fao.org/fishery/docs/CDrom/T583/root/14.pdf",
    "fao_onfarm_feed_management_tilapia.pdf": "https://www.fao.org/fishery/docs/CDrom/T583/root/03.pdf",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCES.items():
        dest = RAW_DIR / filename
        print(f"Fetching {filename} ...")
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            print(f"  WARNING: {filename} does not look like a PDF, skipping")
            continue
        dest.write_bytes(response.content)
        print(f"  saved {dest} ({len(response.content) / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
