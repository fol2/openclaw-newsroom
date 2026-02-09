from __future__ import annotations

from newsroom.image_fetch import extract_og_image_url


def test_extract_og_image_url_prefers_og() -> None:
    html = (
        "<html><head>"
        '<meta property="og:image" content="/img/og.jpg" />'
        '<meta name="twitter:image" content="/img/tw.jpg" />'
        "</head></html>"
    )
    assert extract_og_image_url(html, base_url="https://example.com/a/b") == "https://example.com/img/og.jpg"


def test_extract_og_image_url_falls_back_to_twitter() -> None:
    html = "<html><head>" '<meta name="twitter:image" content="https://cdn.example.com/tw.png" />' "</head></html>"
    assert extract_og_image_url(html, base_url="https://example.com") == "https://cdn.example.com/tw.png"


def test_extract_og_image_url_none_when_missing() -> None:
    assert extract_og_image_url("<html><head></head><body>x</body></html>", base_url="https://example.com") is None

