# Usage: python -m src.etl.scrape_descriptions --max-books 50000 --concurrency 20
# Fetches missing OpenLibrary work descriptions for tier-2 books and appends JSONL results.

"""
Scrape missing OpenLibrary descriptions for tier-2 books in the processed catalog.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

from src.etl.clean import parse_description

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "books_tier1-2_500k.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "scraped_descriptions.jsonl"
OPENLIBRARY_WORK_URL = "https://openlibrary.org{work_id}.json"
DEFAULT_USER_AGENT = "BookSearch/1.0 (portfolio project; contact@example.com)"
REQUESTS_PER_SECOND = 50
REQUEST_TIMEOUT = httpx.Timeout(10.0)
MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 30.0


@dataclass(slots=True)
class CandidateBook:
    work_id: str
    title: str
    edition_count: int | None
    order: int


@dataclass(slots=True)
class ScrapeStats:
    attempted: int = 0
    found: int = 0
    missing: int = 0


class PerSecondRateLimiter:
    """Allow up to N acquisitions in any rolling one-second window."""

    def __init__(self, rate: int) -> None:
        self._semaphore = asyncio.Semaphore(rate)
        self._release_tasks: set[asyncio.Task[None]] = set()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        task = asyncio.create_task(self._release_later())
        self._release_tasks.add(task)
        task.add_done_callback(self._release_tasks.discard)

    async def _release_later(self) -> None:
        try:
            await asyncio.sleep(1.0)
        finally:
            self._semaphore.release()

    async def close(self) -> None:
        if self._release_tasks:
            await asyncio.gather(*self._release_tasks, return_exceptions=True)


class JsonlWriter:
    """Append JSONL records safely from concurrent async tasks."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def extract_api_description(value: object) -> str | None:
    """Normalize the API's description field into a clean string."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    if not isinstance(value, str):
        return None
    return parse_description(value)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(seconds, 0.0)


def load_completed_ids(output_path: Path) -> set[str]:
    """Read existing output records so interrupted runs can resume."""
    if not output_path.exists():
        return set()

    completed_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: skipping malformed output line {line_number} in {output_path}"
                )
                continue
            work_id = record.get("work_id")
            if isinstance(work_id, str) and work_id:
                completed_ids.add(work_id)
    return completed_ids


def load_candidates(input_path: Path) -> list[CandidateBook]:
    """Read tier-2 books with missing descriptions from the processed JSONL dump."""
    candidates: list[CandidateBook] = []
    seen_work_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            work_id = record.get("work_id")
            title = record.get("title")
            tier = record.get("tier")
            description = record.get("description")
            subjects = record.get("subjects") or []

            if not isinstance(work_id, str) or not work_id.startswith("/works/"):
                continue
            if work_id in seen_work_ids:
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            if description:
                continue

            if tier is not None:
                if tier != 2:
                    continue
            elif not isinstance(subjects, list) or not subjects:
                continue

            edition_count = record.get("edition_count")
            if not isinstance(edition_count, int):
                edition_count = None

            seen_work_ids.add(work_id)
            candidates.append(
                CandidateBook(
                    work_id=work_id,
                    title=title.strip(),
                    edition_count=edition_count,
                    order=order,
                )
            )

    return candidates


def select_books(candidates: list[CandidateBook], max_books: int) -> tuple[list[CandidateBook], bool]:
    """Select the working set, preferring books with higher edition_count when present."""
    has_popularity = any(book.edition_count is not None for book in candidates)
    if has_popularity:
        ordered = sorted(
            candidates,
            key=lambda book: (book.edition_count is None, -(book.edition_count or 0), book.order),
        )
    else:
        ordered = candidates
    return ordered[:max_books], has_popularity


async def fetch_description(
    client: httpx.AsyncClient,
    work_id: str,
    rate_limiter: PerSecondRateLimiter,
) -> str | None:
    """Fetch and normalize a work description from the OpenLibrary Works API."""
    url = OPENLIBRARY_WORK_URL.format(work_id=work_id)
    backoff = 1.0

    for attempt in range(1, MAX_RETRIES + 1):
        await rate_limiter.acquire()

        try:
            response = await client.get(url)
        except (httpx.TimeoutException, httpx.RequestError):
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                return None
            return extract_api_description(payload.get("description"))

        if response.status_code == 404:
            return None

        if response.status_code == 429:
            retry_after = parse_retry_after(response.headers.get("Retry-After"))
            await asyncio.sleep(retry_after or min(backoff, MAX_BACKOFF_SECONDS))
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        if 500 <= response.status_code < 600:
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep(min(backoff, MAX_BACKOFF_SECONDS))
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
            continue

        return None

    return None


async def worker(
    queue: asyncio.Queue[CandidateBook | None],
    client: httpx.AsyncClient,
    rate_limiter: PerSecondRateLimiter,
    writer: JsonlWriter,
    progress: tqdm,
    stats: ScrapeStats,
) -> None:
    """Process books from the queue until a sentinel is received."""
    while True:
        book = await queue.get()
        if book is None:
            queue.task_done()
            return

        try:
            description = await fetch_description(client, book.work_id, rate_limiter)
            await writer.write(
                {
                    "work_id": book.work_id,
                    "title": book.title,
                    "description": description,
                    "source": "openlibrary_api",
                }
            )
            stats.attempted += 1
            if description:
                stats.found += 1
            else:
                stats.missing += 1
        finally:
            progress.update(1)
            queue.task_done()


async def run(args: argparse.Namespace) -> None:
    input_path = INPUT_PATH
    output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    completed_ids = load_completed_ids(output_path)
    candidates = load_candidates(input_path)
    selected_books, used_popularity = select_books(candidates, args.max_books)
    pending_books = [book for book in selected_books if book.work_id not in completed_ids]

    print(f"Loaded {len(candidates):,} tier-2 candidates from {input_path}")
    if used_popularity:
        print("Sorting by edition_count (descending) before selection")
    else:
        print("No edition_count field found; using file order for selection")
    print(f"Selected top {len(selected_books):,} books (max_books={args.max_books:,})")
    print(f"Skipping {len(selected_books) - len(pending_books):,} already fetched work IDs")

    if not pending_books:
        print(f"Nothing to do. Output already covers the selected work IDs in {output_path}")
        return

    queue: asyncio.Queue[CandidateBook | None] = asyncio.Queue()
    for book in pending_books:
        await queue.put(book)

    rate_limiter = PerSecondRateLimiter(REQUESTS_PER_SECOND)
    writer = JsonlWriter(output_path)
    progress = tqdm(total=len(pending_books), desc="Fetching descriptions", unit="book")
    stats = ScrapeStats()

    try:
        limits = httpx.Limits(
            max_connections=max(args.concurrency, 20),
            max_keepalive_connections=args.concurrency,
        )
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        async with httpx.AsyncClient(
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            limits=limits,
        ) as client:
            async with asyncio.TaskGroup() as task_group:
                for _ in range(args.concurrency):
                    task_group.create_task(
                        worker(queue, client, rate_limiter, writer, progress, stats)
                    )

                await queue.join()

                for _ in range(args.concurrency):
                    await queue.put(None)
    finally:
        progress.close()
        writer.close()
        await rate_limiter.close()

    print(
        f"Finished: attempted={stats.attempted:,}, "
        f"descriptions_found={stats.found:,}, missing={stats.missing:,}"
    )
    print(f"Results appended to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch missing OpenLibrary descriptions for tier-2 books."
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=50_000,
        help="Max number of tier-2 books to select from the input file.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of concurrent worker tasks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL file to append scraped descriptions to.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_books <= 0:
        parser.error("--max-books must be greater than 0")
    if args.concurrency <= 0:
        parser.error("--concurrency must be greater than 0")

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress already written to the output file.")


if __name__ == "__main__":
    main()
