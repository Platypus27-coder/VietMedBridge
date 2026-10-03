"""Extract known article containers while preserving source-derived block order."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


_BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "figcaption", "blockquote", "pre"}


def _inline_text(node: Tag) -> str:
    # Joining inline nodes with extra spaces can corrupt e.g. 1<span>.5</span>mg.
    pieces = []
    for item in node.descendants:
        if isinstance(item, Comment):
            continue
        if isinstance(item, NavigableString):
            pieces.append(str(item))
        elif isinstance(item, Tag) and item.name == "br":
            pieces.append(" ")
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _blocks(root: Tag) -> list[str]:
    pieces = []

    def walk(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            value = re.sub(r"\s+", " ", str(node)).strip()
            if value:
                pieces.append(value)
            return
        if not isinstance(node, Tag):
            return
        if node.name == "table":
            caption = node.find("caption", recursive=False)
            if caption:
                value = _inline_text(caption)
                if value:
                    pieces.append(value)
            rows = []
            for row in node.find_all("tr"):
                cells = row.find_all(["th", "td"], recursive=False)
                if cells:
                    rows.append(" | ".join(_inline_text(cell) for cell in cells))
            if rows:
                pieces.append("\n".join(rows))
            return
        if node.name in _BLOCKS and not node.find(list(_BLOCKS | {"table"})):
            value = _inline_text(node)
            if value:
                pieces.append(value)
            return
        for child in node.children:
            walk(child)

    walk(root)
    return pieces


def longchau_article(soup: BeautifulSoup, source_url: str) -> dict | None:
    """Use only the audited Long Châu article template, never UI-wide fallback.

    Other hosts/paths return None so generic extraction remains available.
    A changed known article layout raises rather than silently ingesting menus.
    """
    requested = urlsplit(source_url)
    if requested.hostname != "nhathuoclongchau.com.vn" or not requested.path.startswith("/bai-viet/"):
        return None
    canonical = soup.select("link[rel=canonical]")
    if len(canonical) != 1:
        raise ValueError("longchau_missing_or_ambiguous_canonical")
    reported = urlsplit(str(canonical[0].get("href", "")))
    if reported.hostname != requested.hostname or reported.path != requested.path:
        raise ValueError("longchau_canonical_url_mismatch")
    containers = soup.select('[data-lcpr="prr-id-articles-content"]')
    if len(containers) != 1:
        raise ValueError("longchau_article_layout_changed")
    container = containers[0]
    headings = container.select("h1")
    bodies = container.select('[data-theme-element="article"]')
    if len(headings) != 1 or len(bodies) != 1:
        raise ValueError("longchau_article_layout_changed")
    title, body = _inline_text(headings[0]), bodies[0]
    if not title or body.parent is not headings[0].parent:
        raise ValueError("longchau_article_layout_changed")
    # The audited template puts the article's standfirst immediately before
    # the body, outside its main-content element. Keep it as source content.
    intro = body.find_previous_sibling()
    if not isinstance(intro, Tag) or not intro.find("p") or intro.find(["button", "h1", "h2", "h3"]):
        raise ValueError("longchau_article_intro_layout_changed")
    for root in (intro, body):
        for node in list(root.select("script,style,svg,nav,aside,form,button,[hidden],[aria-hidden=true]")):
            node.decompose()
    intro_parts = _blocks(intro)
    body_parts = _blocks(body)
    if not body_parts:
        raise ValueError("longchau_empty_article_body")
    source = "\n\n".join([title, *intro_parts, *body_parts])
    return {"title": title, "source_text": source,
            "heading_hints": [title, *[_inline_text(h) for h in body.find_all(re.compile(r"^h[1-6]$"))]],
            "raw_has_table": body.find("table") is not None,
            "parser": "longchau-article-dom-v1"}
