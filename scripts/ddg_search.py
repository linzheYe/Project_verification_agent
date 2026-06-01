from __future__ import annotations

from typing import Any


def ensure_ddgs_dependency() -> None:
    """Fail fast when DuckDuckGo dependency is missing.

    Why this check exists:
    - Without `ddgs`, retrieval always returns empty evidence and the run is misleading.
    - We prefer immediate, explicit failure before entering long LLM loops.
    """
    try:
        import ddgs  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "missing dependency `ddgs` for DuckDuckGo retrieval. "
            "Please install it in current Python env (e.g. `pip install ddgs`) "
            "or run with an environment that already has it "
            "(e.g. `/home/an/.venvs/ddg_test/bin/python`)."
        ) from e


def search_snippets_with_ddg(
    search_query: str,
    *,
    max_results: int = 8,
) -> list[dict[str, str]]:
    """Run one DuckDuckGo search and return normalized snippets.

    Why this module is standalone:
    - Retrieval backend can change later without touching pipeline logic.
    - Pipeline only needs url/title/content, so we normalize output here.
    """
    query = str(search_query or "").strip()
    if not query:
        return []

    ensure_ddgs_dependency()
    try:
        from ddgs import DDGS
    except Exception as e:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(f"ddgs import failed: {e}")

    rows: list[dict[str, str]] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for item in results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            content = str(item.get("body") or item.get("snippet") or "").strip()
            if not content:
                continue
            rows.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                }
            )
    return rows
