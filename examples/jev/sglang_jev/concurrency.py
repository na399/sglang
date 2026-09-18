# SPDX-License-Identifier: Apache-2.0
"""Bounded, work-conserving async execution with scoped failure cleanup."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


async def gather_scoped(*awaitables):
    tasks = [asyncio.create_task(a) for a in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        # gather alone does not stop siblings after an ordinary exception.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def map_bounded(
    items: Sequence[T], operation: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    """Keep workers occupied, but return results in input order.

    At most min(concurrency, len(items)) tasks exist. Claiming an item contains
    no await, so the iterator is safe to share within this single event loop.
    An unexpected failure or caller cancellation drains all active workers
    before returning; it never turns a programming error into a benchmark row.
    """
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    pending = iter(enumerate(items))
    results: list[Any] = [None] * len(items)

    async def worker():
        for index, item in pending:
            results[index] = await operation(item)

    await gather_scoped(*(worker() for _ in range(min(concurrency, len(items)))))
    return cast(list[R], results)
