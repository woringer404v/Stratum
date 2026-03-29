"""
Stratum — API Utilities

Reusable HTTP client layer with retry logic, rate-limit awareness, pagination
helpers, and XML parsing for all four data sources.
"""

import json
import time
import logging
import xml.etree.ElementTree as ET

import requests

from config import RETRY_CONFIG, API_ENDPOINTS

logger = logging.getLogger("stratum.api")


# ---------------------------------------------------------------------------
# Core HTTP — GET with Retry
# ---------------------------------------------------------------------------
def fetch_with_retry(url, params=None, headers=None, max_retries=None,
                     backoff_base=None, timeout=None):
    """Make an HTTP GET request with exponential backoff retry.

    Retries on HTTP 429 (rate limit), 5xx (server error), and ConnectionError.
    Respects Retry-After header when present on 429 responses.

    Args:
        url: The URL to fetch.
        params: Optional query parameters dict.
        headers: Optional HTTP headers dict.
        max_retries: Max retry attempts (default from RETRY_CONFIG).
        backoff_base: Base for exponential backoff in seconds (default from RETRY_CONFIG).
        timeout: Request timeout in seconds (default from RETRY_CONFIG).

    Returns:
        requests.Response object.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    max_retries = max_retries or RETRY_CONFIG["max_retries"]
    backoff_base = backoff_base or RETRY_CONFIG["backoff_base"]
    timeout = timeout or RETRY_CONFIG["timeout"]

    retryable_status_codes = {429, 500, 502, 503, 504}
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)

            if response.status_code == 200:
                return response

            if response.status_code in retryable_status_codes:
                wait_time = _get_wait_time(response, attempt, backoff_base)
                logger.warning(
                    "Retryable HTTP %d from %s — waiting %.1fs (attempt %d/%d)",
                    response.status_code, url, wait_time, attempt + 1, max_retries,
                )
                time.sleep(wait_time)
                continue

            # Non-retryable error
            response.raise_for_status()

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exception = exc
            wait_time = backoff_base ** attempt
            logger.warning(
                "Connection error fetching %s — waiting %.1fs (attempt %d/%d): %s",
                url, wait_time, attempt + 1, max_retries, exc,
            )
            time.sleep(wait_time)
            continue

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise requests.HTTPError(
        f"All {max_retries} retries exhausted for {url} "
        f"(last status: {response.status_code})"
    )


def post_with_retry(url, json_body, headers=None, max_retries=None,
                    backoff_base=None, timeout=None):
    """Make an HTTP POST request with exponential backoff retry.

    Same retry semantics as fetch_with_retry but for POST requests.
    Used for LLM API calls (OpenAI ChatCompletion).

    Args:
        url: The URL to post to.
        json_body: Dict to send as JSON body.
        headers: Optional HTTP headers dict.
        max_retries: Max retry attempts (default from RETRY_CONFIG).
        backoff_base: Base for exponential backoff in seconds.
        timeout: Request timeout in seconds.

    Returns:
        requests.Response object.

    Raises:
        requests.HTTPError: If all retries are exhausted.
    """
    max_retries = max_retries or RETRY_CONFIG["max_retries"]
    backoff_base = backoff_base or RETRY_CONFIG["backoff_base"]
    timeout = timeout or RETRY_CONFIG["timeout"]

    retryable_status_codes = {429, 500, 502, 503, 504}
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url, json=json_body, headers=headers, timeout=timeout
            )

            if response.status_code == 200:
                return response

            if response.status_code in retryable_status_codes:
                wait_time = _get_wait_time(response, attempt, backoff_base)
                logger.warning(
                    "Retryable HTTP %d from POST %s — waiting %.1fs (attempt %d/%d)",
                    response.status_code, url, wait_time, attempt + 1, max_retries,
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exception = exc
            wait_time = backoff_base ** attempt
            logger.warning(
                "Connection error posting to %s — waiting %.1fs (attempt %d/%d): %s",
                url, wait_time, attempt + 1, max_retries, exc,
            )
            time.sleep(wait_time)
            continue

    if last_exception:
        raise last_exception
    raise requests.HTTPError(
        f"All {max_retries} retries exhausted for POST {url} "
        f"(last status: {response.status_code})"
    )


def _get_wait_time(response, attempt, backoff_base):
    """Calculate wait time respecting Retry-After header if present.

    Args:
        response: The HTTP response.
        attempt: Current attempt number (0-indexed).
        backoff_base: Base for exponential backoff.

    Returns:
        Wait time in seconds as float.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return min(backoff_base ** attempt, RETRY_CONFIG["backoff_max"])


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------
def fetch_json(url, params=None, headers=None):
    """Fetch a URL and return parsed JSON.

    Args:
        url: The URL to fetch.
        params: Optional query parameters.
        headers: Optional HTTP headers.

    Returns:
        Parsed JSON as dict or list.

    Raises:
        ValueError: If response is not valid JSON.
    """
    response = fetch_with_retry(url, params=params, headers=headers)
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Non-JSON response from {url}: {response.text[:200]}") from exc


def fetch_xml(url, params=None):
    """Fetch a URL and return parsed XML ElementTree root.

    Args:
        url: The URL to fetch.
        params: Optional query parameters.

    Returns:
        xml.etree.ElementTree.Element root node.
    """
    response = fetch_with_retry(url, params=params)
    return ET.fromstring(response.content)


# ---------------------------------------------------------------------------
# GitHub Pagination
# WHY: GitHub has no trending endpoint. We query the search API with
# created:>DATE sort:stars to find recently-created repos gaining traction.
# Rate limit for unauthenticated search: 10 req/min — we sleep between pages.
# ---------------------------------------------------------------------------
def paginate_github(query, max_pages=3, per_page=30):
    """Fetch multiple pages of GitHub repository search results.

    Uses the GitHub Search API (no auth required for public read within rate limits).
    Sleeps 6 seconds between pages to respect the 10 req/min unauthenticated limit.

    Args:
        query: The search query string (e.g. "machine learning created:>2026-03-22").
        max_pages: Maximum number of pages to fetch (default 3).
        per_page: Results per page, max 100 (default 30).

    Returns:
        Flat list of repository item dicts.
    """
    base_url = f"{API_ENDPOINTS['github']}/search/repositories"
    all_items = []

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }

        logger.info("GitHub search page %d/%d for query: %s", page, max_pages, query)

        try:
            data = fetch_json(base_url, params=params)
        except Exception as exc:
            logger.error("GitHub search failed on page %d: %s", page, exc)
            break

        items = data.get("items", [])
        if not items:
            logger.info("No more GitHub results on page %d", page)
            break

        all_items.extend(items)
        logger.info("Fetched %d repos (total: %d)", len(items), len(all_items))

        # WHY: Unauthenticated GitHub search allows 10 requests/minute.
        # Sleeping 6s between pages keeps us well under the limit.
        if page < max_pages:
            time.sleep(6)

    return all_items


# ---------------------------------------------------------------------------
# Stack Overflow Pagination
# WHY: SO API returns compressed responses and has a 300 req/day quota for
# unauthenticated callers. We monitor quota_remaining and stop early.
# ---------------------------------------------------------------------------
def paginate_stackoverflow(tagged, max_pages=3, pagesize=100):
    """Fetch multiple pages of Stack Overflow questions for a given tag.

    Stops early if quota_remaining drops below 10 to avoid exhausting the
    daily allocation.

    Args:
        tagged: Semicolon-separated tags (e.g. "python;spark").
        max_pages: Maximum pages to fetch (default 3).
        pagesize: Results per page, max 100 (default 100).

    Returns:
        Flat list of question item dicts.
    """
    base_url = f"{API_ENDPOINTS['stackoverflow']}/questions"
    all_items = []

    for page in range(1, max_pages + 1):
        params = {
            "order": "desc",
            "sort": "activity",
            "tagged": tagged,
            "site": "stackoverflow",
            "pagesize": pagesize,
            "page": page,
            "filter": "withbody",
        }

        logger.info("SO questions page %d/%d for tags: %s", page, max_pages, tagged)

        try:
            data = fetch_json(base_url, params=params)
        except Exception as exc:
            logger.error("SO fetch failed on page %d: %s", page, exc)
            break

        items = data.get("items", [])
        all_items.extend(items)

        quota_remaining = data.get("quota_remaining", 999)
        has_more = data.get("has_more", False)

        logger.info(
            "Fetched %d questions (total: %d, quota_remaining: %d)",
            len(items), len(all_items), quota_remaining,
        )

        if not has_more:
            logger.info("No more SO results available")
            break

        # WHY: Unauthenticated SO quota is 300/day. Stop early to leave
        # headroom for manual exploration or reruns.
        if quota_remaining < 10:
            logger.warning("SO quota nearly exhausted (%d remaining) — stopping", quota_remaining)
            break

    return all_items


# ---------------------------------------------------------------------------
# arXiv Pagination
# WHY: arXiv uses offset-based pagination with XML/Atom responses.
# Rate limit is ~3 seconds between requests — enforced with sleep.
# ---------------------------------------------------------------------------
ARXIV_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


def _parse_arxiv_entry(entry):
    """Parse a single arXiv Atom entry element into a flat dict.

    Args:
        entry: An XML Element representing an <entry> in the Atom feed.

    Returns:
        Dict with keys: arxiv_id, title, summary, authors, published,
        updated, categories, link.
    """
    def text(tag, ns="atom"):
        """Extract text from a namespaced child element."""
        el = entry.find(f"{ns}:{tag}", ARXIV_ATOM_NS) if ns else entry.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    # Authors — extract all <author><name> elements as a JSON array string
    authors = [
        name_el.text.strip()
        for author_el in entry.findall("atom:author", ARXIV_ATOM_NS)
        for name_el in author_el.findall("atom:name", ARXIV_ATOM_NS)
        if name_el.text
    ]

    # Categories — all <category term="..."> attributes
    categories = [
        cat_el.get("term", "")
        for cat_el in entry.findall("atom:category", ARXIV_ATOM_NS)
        if cat_el.get("term")
    ]

    # Link — prefer the abstract page link (rel="alternate")
    link = ""
    for link_el in entry.findall("atom:link", ARXIV_ATOM_NS):
        if link_el.get("rel") == "alternate":
            link = link_el.get("href", "")
            break
    if not link:
        links = entry.findall("atom:link", ARXIV_ATOM_NS)
        if links:
            link = links[0].get("href", "")

    # arXiv ID — extract from the <id> element (e.g. "http://arxiv.org/abs/2301.12345v2")
    raw_id = text("id")
    arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(text("title").split()),  # collapse whitespace
        "summary": " ".join(text("summary").split()),
        "authors": json.dumps(authors),
        "published": text("published"),
        "updated": text("updated"),
        "categories": json.dumps(categories),
        "link": link,
    }


def paginate_arxiv(search_query, max_results=200, batch_size=50):
    """Fetch arXiv results in batches via offset pagination.

    Parses Atom XML entries into flat dicts suitable for DataFrame creation.
    Sleeps 3 seconds between requests per arXiv's rate limit policy.

    Args:
        search_query: arXiv query string (e.g. "cat:cs.AI" or "cat:cs.AI+OR+cat:cs.LG").
        max_results: Total maximum results to fetch (default 200).
        batch_size: Results per API call (default 50).

    Returns:
        List of parsed entry dicts.
    """
    base_url = f"{API_ENDPOINTS['arxiv']}/query"
    all_entries = []
    start = 0

    while start < max_results:
        current_batch = min(batch_size, max_results - start)
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": current_batch,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        logger.info(
            "arXiv fetch: start=%d, batch_size=%d, query=%s",
            start, current_batch, search_query,
        )

        try:
            root = fetch_xml(base_url, params=params)
        except Exception as exc:
            logger.error("arXiv fetch failed at start=%d: %s", start, exc)
            break

        entries = root.findall("atom:entry", ARXIV_ATOM_NS)
        if not entries:
            logger.info("No more arXiv entries at start=%d", start)
            break

        for entry in entries:
            parsed = _parse_arxiv_entry(entry)
            if parsed["arxiv_id"]:
                all_entries.append(parsed)

        logger.info("Parsed %d entries (total: %d)", len(entries), len(all_entries))

        start += current_batch

        # WHY: arXiv recommends >= 3 seconds between requests
        if start < max_results:
            time.sleep(3)

    return all_entries
