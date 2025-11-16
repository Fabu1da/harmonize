from typing import Callable, Generator, Iterable, TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], max_count: int, max_size: int, size_fn: Callable[[T], int] = len) -> Generator[list[T], None, None]:
    batch: list[T] = []
    batch_size = 0
    for item in iterable:
        item_size: int = size_fn(item)
        if item_size > max_size:
            raise ValueError(f"Item size {item_size} exceeds max_size {max_size}")
        if batch_size + item_size > max_size:
            yield batch
            batch = []
            batch_size = 0
        batch.append(item)
        batch_size += item_size
        if len(batch) == max_count:
            yield batch
            batch = []
    if batch:
        yield batch
