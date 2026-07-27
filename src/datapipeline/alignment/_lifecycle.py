from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from datapipeline.domain.record import TemporalRecord


@contextmanager
def closing_alignment_inputs(
    inputs: Iterable[Iterator[TemporalRecord]],
) -> Iterator[None]:
    """Close every input while preserving alignment failure precedence."""

    processing_failed = False
    try:
        yield
    except GeneratorExit:
        raise
    except BaseException:
        processing_failed = True
        raise
    finally:
        first_error: BaseException | None = None
        for records in inputs:
            try:
                close = getattr(records, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if not processing_failed and first_error is not None:
            raise first_error
