from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
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


@dataclass
class SeriesProjector:
    partition_by: tuple[str, ...]
    sample_keys: SampleKeyContract
    series_id_fields: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        sample_keys = set(self.sample_keys.fields)
        self.series_id_fields = tuple(
            partition_field
            for partition_field in self.partition_by
            if partition_field not in sample_keys
        )

    def project(
        self,
        record: Any,
        configs: Sequence[SeriesConfig],
    ) -> Iterator[SeriesRecord]:
        entity_key = partition_key(record, self.sample_keys.fields)
        self.sample_keys.validate(entity_key)
        suffix = None
        if self.series_id_fields:
            suffix = SERIES_ID_COMPONENT_SEPARATOR.join(
                encode_series_id_component(field, getattr(record, field))
                for field in self.series_id_fields
            )
        establishes_domain = record_establishes_domain(record)

        for config in configs:
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
