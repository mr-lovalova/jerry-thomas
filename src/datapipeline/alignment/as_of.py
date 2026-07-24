from collections.abc import Generator, Iterator
from datetime import datetime, timedelta

from datapipeline.domain.record import TemporalRecord
from datapipeline.transforms.utils import partition_key


_CanonicalKey = tuple[tuple[object, ...], datetime]


def as_of_stream(
    primary: Iterator[TemporalRecord],
    lookup: Iterator[TemporalRecord],
    partition_by: tuple[str, ...],
    max_age: timedelta | None = None,
    require_match: bool = True,
) -> Generator[tuple[TemporalRecord, TemporalRecord | None], None, None]:
    """Pair primary records with the latest eligible lookup in the same partition."""
    processing_failed = False

    try:
        if max_age is not None and not isinstance(max_age, timedelta):
            raise TypeError("As-of max_age must be a timedelta or None")
        if max_age is not None and max_age <= timedelta(0):
            raise ValueError("As-of max_age must be a positive timedelta")
        if type(require_match) is not bool:
            raise TypeError("As-of require_match must be a boolean")

        previous_primary_key: _CanonicalKey | None = None
        primary_partition_types: tuple[type[object], ...] | None = None

        lookup_record: TemporalRecord | None = None
        lookup_key: _CanonicalKey | None = None
        lookup_exhausted = False
        lookup_partition_types: tuple[type[object], ...] | None = None

        def advance_lookup(
            expected_primary_types: tuple[type[object], ...] | None,
        ) -> None:
            nonlocal lookup_exhausted
            nonlocal lookup_key
            nonlocal lookup_partition_types
            nonlocal lookup_record

            try:
                record = next(lookup)
            except StopIteration:
                lookup_exhausted = True
                lookup_key = None
                lookup_record = None
                return

            partition = partition_key(record, partition_by)
            if lookup_partition_types is None:
                lookup_partition_types = tuple(type(value) for value in partition)
                if expected_primary_types is not None:
                    for field, primary_type, lookup_type in zip(
                        partition_by,
                        expected_primary_types,
                        lookup_partition_types,
                        strict=True,
                    ):
                        if lookup_type is not primary_type:
                            raise TypeError(
                                f"As-of lookup partition field {field!r} uses "
                                f"{lookup_type.__name__}; primary uses "
                                f"{primary_type.__name__}."
                            )

            key = partition, record.time
            if key == lookup_key:
                raise ValueError(
                    "As-of lookup has duplicate canonical key "
                    f"partition={partition!r}, time={record.time.isoformat()}"
                )
            if lookup_key is not None and key < lookup_key:
                raise ValueError(
                    f"As-of lookup is not ordered: key {key!r} follows {lookup_key!r}"
                )

            lookup_record = record
            lookup_key = key

        candidate: TemporalRecord | None = None
        candidate_partition: tuple[object, ...] | None = None

        for primary_record in primary:
            primary_partition = partition_key(primary_record, partition_by)
            if primary_partition_types is None:
                primary_partition_types = tuple(
                    type(value) for value in primary_partition
                )

            primary_key = primary_partition, primary_record.time
            if primary_key == previous_primary_key:
                raise ValueError(
                    "As-of primary has duplicate canonical key "
                    f"partition={primary_partition!r}, "
                    f"time={primary_record.time.isoformat()}"
                )
            if previous_primary_key is not None and primary_key < previous_primary_key:
                raise ValueError(
                    "As-of primary is not ordered: "
                    f"key {primary_key!r} follows {previous_primary_key!r}"
                )
            previous_primary_key = primary_key

            if not lookup_exhausted and lookup_record is None:
                advance_lookup(primary_partition_types)

            if candidate_partition != primary_partition:
                candidate = None
                candidate_partition = primary_partition

            while lookup_key is not None and lookup_key[0] < primary_partition:
                advance_lookup(primary_partition_types)

            while (
                lookup_key is not None
                and lookup_key[0] == primary_partition
                and lookup_key[1] <= primary_record.time
            ):
                candidate = lookup_record
                advance_lookup(primary_partition_types)

            match = candidate
            if (
                match is not None
                and max_age is not None
                and primary_record.time - match.time > max_age
            ):
                match = None

            if match is None and require_match:
                raise ValueError(
                    "As-of lookup has no eligible record for primary "
                    f"partition={primary_partition!r}, "
                    f"time={primary_record.time.isoformat()}"
                )

            yield primary_record, match

        # A future lookahead can hide a later out-of-order eligible record.
        while not lookup_exhausted:
            advance_lookup(primary_partition_types)
    except GeneratorExit:
        raise
    except BaseException:
        processing_failed = True
        raise
    finally:
        try:
            primary_close = getattr(primary, "close", None)
            if callable(primary_close):
                primary_close()
        except BaseException:
            if not processing_failed:
                processing_failed = True
                raise
        finally:
            try:
                lookup_close = getattr(lookup, "close", None)
                if callable(lookup_close):
                    lookup_close()
            except BaseException:
                if not processing_failed:
                    raise
