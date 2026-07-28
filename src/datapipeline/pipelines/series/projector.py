from collections.abc import Iterator
from typing import Any

from datapipeline.config.dataset.series import SeriesConfig
from datapipeline.domain.series import SeriesRecord
from datapipeline.domain.series_id import (
    SERIES_ID_COMPONENT_SEPARATOR,
    encode_series_id_component,
    make_partitioned_series_id,
)
from datapipeline.domain.sample_key import SampleKeyContract
from datapipeline.domain.value import normalize_series_value
from datapipeline.transforms.utils import (
    get_field,
    partition_key,
    record_establishes_domain,
)


class SeriesProjector:
    def __init__(
        self,
        partition_by: tuple[str, ...],
        sample_keys: SampleKeyContract,
        configs: tuple[SeriesConfig, ...],
    ) -> None:
        self._sample_keys = sample_keys
        self._configs = configs
        sample_key_fields = set(self._sample_keys.fields)
        self._series_id_fields = tuple(
            field for field in partition_by if field not in sample_key_fields
        )

    def project(
        self,
        record: Any,
    ) -> Iterator[SeriesRecord]:
        entity_key = partition_key(record, self._sample_keys.fields)
        self._sample_keys.validate(entity_key)
        suffix = None
        if self._series_id_fields:
            suffix = SERIES_ID_COMPONENT_SEPARATOR.join(
                encode_series_id_component(field, getattr(record, field))
                for field in self._series_id_fields
            )
        establishes_domain = record_establishes_domain(record)
        for config in self._configs:
            series_id = (
                config.id
                if suffix is None
                else make_partitioned_series_id(config.id, suffix)
            )
            yield SeriesRecord(
                id=series_id,
                time=record.time,
                value=normalize_series_value(get_field(record, config.field)),
                entity_key=entity_key,
                _establishes_domain=establishes_domain,
            )
