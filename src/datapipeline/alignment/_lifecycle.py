from collections.abc import Iterable, Iterator

from datapipeline.domain.record import TemporalRecord


def close_alignment_inputs(
    inputs: Iterable[Iterator[TemporalRecord]],
) -> BaseException | None:
    """Close every alignment input and return the first cleanup failure."""

    first_error: BaseException | None = None
    for records in inputs:
        try:
            close = getattr(records, "close", None)
            if callable(close):
                close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error
