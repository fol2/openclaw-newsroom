from __future__ import annotations

import hashlib
import mimetypes
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)


class _MetaImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.og: str | None = None
        self.twitter: str | None = None
        self.image_src: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = (tag or "").lower().strip()
        if t not in {"meta", "link"}:
            return
        d: dict[str, str] = {}
        for k, v in attrs:
            if not k or v is None:
                continue
            d[str(k).lower().strip()] = str(v).strip()

        if t == "meta":
            key = (d.get("property") or d.get("name") or "").lower().strip()
            content = (d.get("content") or "").strip()
            if not content:
                return
            if key in {"og:image", "og:image:url"} and not self.og:
                self.og = content
            elif key in {"twitter:image", "twitter:image:src"} and not self.twitter:
                self.twitter = content
            return

        if t == "link":
            rel = (d.get("rel") or "").lower().strip()
            href = (d.get("href") or "").strip()
            if rel == "image_src" and href and not self.image_src:
                self.image_src = href


def extract_og_image_url(html_text: str, *, base_url: str | None = None) -> str | None:
    if not isinstance(html_text, str) or not html_text.strip():
        return None
    # Keep this bounded: we only need the head.
    head = html_text[:200_000]
    p = _MetaImageParser()
    try:
        p.feed(head)
    except Exception:
        return None

    cand = p.og or p.twitter or p.image_src
    if not cand:
        return None

    cand = cand.strip()
    if not cand:
        return None
    if base_url:
        try:
            cand = urljoin(base_url, cand)
        except Exception:
            pass
    return cand


def fetch_og_image_url(
    session: requests.Session,
    page_url: str,
    *,
    timeout_seconds: int = 8,
    max_bytes: int = 1_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str | None:
    page_url = str(page_url or "").strip()
    if not page_url.startswith("http"):
        return None
    try:
        resp = session.get(
            page_url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-HK,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            },
            timeout=timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
    except Exception:
        return None
    if resp.status_code < 200 or resp.status_code >= 400:
        return None
    ctype = str(resp.headers.get("content-type") or "").lower()
    if "html" not in ctype and "xml" not in ctype:
        # Some sites mislabel, still allow if it's not obviously binary.
        if any(x in ctype for x in ("image/", "video/", "application/pdf")):
            return None

    buf = bytearray()
    try:
        for chunk in resp.iter_content(chunk_size=16_384):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) >= int(max_bytes):
                break
    except Exception:
        return None
    enc = resp.encoding or "utf-8"
    try:
        html_text = bytes(buf).decode(enc, errors="ignore")
    except Exception:
        html_text = bytes(buf).decode("utf-8", errors="ignore")

    final_url = str(getattr(resp, "url", page_url) or page_url)
    return extract_og_image_url(html_text, base_url=final_url)


def _ext_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    ct = str(content_type).split(";")[0].strip().lower()
    ext = mimetypes.guess_extension(ct)
    if ext:
        return ext
    if ct == "image/jpg":
        return ".jpg"
    return None


def _ext_from_url(url: str) -> str | None:
    try:
        p = urlparse(url)
        path = p.path or ""
    except Exception:
        path = ""
    m = re.search(r"\.(jpg|jpeg|png|webp|gif)(?:$|\\?)", path, flags=re.IGNORECASE)
    if not m:
        return None
    return "." + m.group(1).lower()


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    path: str | None
    final_url: str | None
    content_type: str | None
    bytes_written: int
    error: str | None


def download_image(
    session: requests.Session,
    image_url: str,
    *,
    dest_dir: Path,
    timeout_seconds: int = 10,
    max_bytes: int = 8_000_000,
    user_agent: str = DEFAULT_USER_AGENT,
) -> DownloadResult:
    image_url = str(image_url or "").strip()
    if not image_url.startswith("http"):
        return DownloadResult(ok=False, path=None, final_url=None, content_type=None, bytes_written=0, error="invalid_url")

    digest = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:20]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        resp = session.get(
            image_url,
            headers={"User-Agent": user_agent, "Accept": "image/*,*/*;q=0.8"},
            timeout=timeout_seconds,
            allow_redirects=True,
            stream=True,
        )
    except Exception as e:
        return DownloadResult(ok=False, path=None, final_url=None, content_type=None, bytes_written=0, error=str(e))

    if resp.status_code < 200 or resp.status_code >= 400:
        return DownloadResult(ok=False, path=None, final_url=str(getattr(resp, "url", image_url)), content_type=None, bytes_written=0, error=f"http_status:{resp.status_code}")

    ctype = str(resp.headers.get("content-type") or "").lower()
    if ctype and not ctype.startswith("image/"):
        return DownloadResult(ok=False, path=None, final_url=str(getattr(resp, "url", image_url)), content_type=ctype, bytes_written=0, error="not_image")

    ext = _ext_from_content_type(ctype) or _ext_from_url(str(getattr(resp, "url", image_url))) or ".jpg"
    path = dest_dir / f"og_{digest}{ext}"
    if path.exists() and path.stat().st_size > 1_000:
        return DownloadResult(ok=True, path=str(path), final_url=str(getattr(resp, "url", image_url)), content_type=ctype or None, bytes_written=int(path.stat().st_size), error=None)

    written = 0
    try:
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64_000):
                if not chunk:
                    continue
                written += len(chunk)
                if written > int(max_bytes):
                    return DownloadResult(ok=False, path=str(path), final_url=str(getattr(resp, "url", image_url)), content_type=ctype or None, bytes_written=written, error="too_large")
                f.write(chunk)
    except Exception as e:
        return DownloadResult(ok=False, path=str(path), final_url=str(getattr(resp, "url", image_url)), content_type=ctype or None, bytes_written=written, error=str(e))

    if written < 4_000:
        # Too small: likely a pixel/placeholder.
        return DownloadResult(ok=False, path=str(path), final_url=str(getattr(resp, "url", image_url)), content_type=ctype or None, bytes_written=written, error="too_small")

    return DownloadResult(ok=True, path=str(path), final_url=str(getattr(resp, "url", image_url)), content_type=ctype or None, bytes_written=written, error=None)
