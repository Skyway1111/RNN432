"""The social card a shared link shows: the hero at 1200x630, without its reading notes, to web/card.jpg.

python -m web.card
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
    page.goto((HERE / "index.html").as_uri() + "?final")
    page.add_style_tag(content="#hero .live { visibility: hidden; }")
    page.wait_for_timeout(3500)  # the hero's bloom
    page.screenshot(path=str(HERE / "card.jpg"), type="jpeg", quality=90)
    browser.close()
