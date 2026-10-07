from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import httpx


@dataclass(frozen=True)
class ProbeResult:
    request_index: int
    key_index: int
    page: int
    status_code: int | None
    organic_count: int
    elapsed_seconds: float
    error: str | None = None


def load_keys(path: Path) -> list[str]:
    keys = [line.strip() for line in path.expanduser().read_text().splitlines()]
    keys = [key for key in keys if key and not key.startswith("#")]
    if not keys:
        raise RuntimeError(f"no Serper keys found in {path}")
    return keys


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize(results: Sequence[ProbeResult], wall_seconds: float) -> dict[str, object]:
    completed = [item for item in results if item.status_code == 200 and item.error is None]
    useful = [item for item in completed if item.organic_count > 0]
    latencies = [item.elapsed_seconds for item in completed]
    errors: dict[str, int] = {}
    for item in results:
        if item.error:
            errors[item.error] = errors.get(item.error, 0) + 1
        elif item.status_code != 200:
            label = f"HTTP_{item.status_code}"
            errors[label] = errors.get(label, 0) + 1
        elif item.organic_count == 0:
            errors["EMPTY_ORGANIC"] = errors.get("EMPTY_ORGANIC", 0) + 1
    return {
        "requests": len(results),
        "http_200": len(completed),
        "useful_responses": len(useful),
        "success_ratio": round(len(useful) / len(results), 6) if results else 0.0,
        "organic_min": min((item.organic_count for item in completed), default=0),
        "organic_mean": round(
            statistics.fmean(item.organic_count for item in completed), 3
        ) if completed else 0.0,
        "latency_mean_seconds": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "latency_p95_seconds": round(percentile(latencies, 0.95), 3),
        "wall_seconds": round(wall_seconds, 3),
        "errors": errors,
    }


async def run_probe(
    *,
    keys: Sequence[str],
    proxy_url: str,
    search_url: str,
    query: str,
    request_count: int,
    concurrency: int,
    per_key_concurrency: int,
    pages: int,
    timeout_seconds: float,
) -> tuple[list[ProbeResult], float]:
    if request_count < 1 or concurrency < 1 or per_key_concurrency < 1 or pages < 1:
        raise ValueError("request_count, concurrency, per_key_concurrency, and pages must be positive")
    if per_key_concurrency > 5:
        raise ValueError("per_key_concurrency must not exceed the Serper limit of 5")

    global_limit = asyncio.Semaphore(concurrency)
    key_limits = [asyncio.Semaphore(per_key_concurrency) for _ in keys]
    limits = httpx.Limits(
        max_connections=max(concurrency, len(keys)),
        max_keepalive_connections=max(concurrency, len(keys)),
    )
    started_wall = time.perf_counter()
    async with httpx.AsyncClient(
        proxy=proxy_url,
        follow_redirects=True,
        limits=limits,
        timeout=timeout_seconds,
    ) as client:
        async def one(index: int) -> ProbeResult:
            key_index = index % len(keys)
            page = index % pages + 1
            started = time.perf_counter()
            try:
                async with global_limit, key_limits[key_index]:
                    response = await client.post(
                        search_url,
                        headers={
                            "X-API-KEY": keys[key_index],
                            "Content-Type": "application/json",
                        },
                        json={"q": query, "num": 10, "page": page},
                    )
                organic_count = 0
                if response.status_code == 200:
                    payload = response.json()
                    organic_count = len(payload.get("organic", []))
                return ProbeResult(
                    request_index=index,
                    key_index=key_index,
                    page=page,
                    status_code=response.status_code,
                    organic_count=organic_count,
                    elapsed_seconds=time.perf_counter() - started,
                )
            except Exception as exc:
                return ProbeResult(
                    request_index=index,
                    key_index=key_index,
                    page=page,
                    status_code=None,
                    organic_count=0,
                    elapsed_seconds=time.perf_counter() - started,
                    error=type(exc).__name__,
                )

        results = await asyncio.gather(*(one(index) for index in range(request_count)))
    return list(results), time.perf_counter() - started_wall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Serper proxy preflight. API keys and response bodies are never printed."
    )
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=Path("secrets/serper.keys"),
    )
    parser.add_argument("--proxy-url", required=True)
    parser.add_argument("--search-url", default="https://google.serper.dev/search")
    parser.add_argument("--query", default="GLM-4.7-Flash long context inference")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--per-key-concurrency", type=int, default=5)
    parser.add_argument("--pages", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--min-success-ratio", type=float, default=1.0)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    keys = load_keys(args.keys_file)
    results, wall_seconds = await run_probe(
        keys=keys,
        proxy_url=args.proxy_url,
        search_url=args.search_url,
        query=args.query,
        request_count=args.requests,
        concurrency=args.concurrency,
        per_key_concurrency=args.per_key_concurrency,
        pages=args.pages,
        timeout_seconds=args.timeout_seconds,
    )
    report = summarize(results, wall_seconds)
    report["keys_loaded"] = len(keys)
    report["proxy_url"] = args.proxy_url
    report["per_key_concurrency"] = args.per_key_concurrency
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if float(report["success_ratio"]) >= args.min_success_ratio else 1


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
