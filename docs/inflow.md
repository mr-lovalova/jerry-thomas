# Inflow Walkthrough

`jerry inflow create` is an interactive wizard that scaffolds an end-to-end
source-backed stream: source YAML + DTO/parser + domain + mapper + stream.

## Quick Example

```bash
jerry inflow create
```

Example prompt flow (your choices may differ):

```
Provider: noaa
Dataset: weather
Source: create (default)
Loader: fs
Format: jsonl
Parser: create
DTO class name: WeatherNoaaDTO
Parser class name: WeatherNoaaDTOParser
Domain: weather
Mapper: create
Mapper input: DTO
Mapper name: map_weather_noaa_dto_to_weather
```

This produces (paths may vary):

- `src/<package>/dtos/weather_noaa_dto.py`
- `src/<package>/parsers/weather_noaa_dto_parser.py`
- `src/<package>/domains/weather/model.py`
- `src/<package>/mappers/map_weather_noaa_dto_to_weather.py`
- `your-dataset/sources/noaa.weather.yaml`
- `your-dataset/streams/weather.weather.yaml`

## Identity vs Custom

### Parser

Choose **Identity parser** when your loader already emits dicts/objects that match
your DTO shape and no type conversion is needed. Choose **Create parser** when you
need to parse timestamps, coerce types, rename fields, or drop/validate rows.

For persisted domain-record-like JSON/JSONL rows, choose **Temporal record
rehydration** (`core.temporal_record`) to rehydrate canonical records without
creating a DTO or custom mapper.

### Mapper

Choose **Identity mapper** only when your DTO is already the final domain record
shape (timezone‑aware `time` plus any identity fields). Otherwise choose
**Create mapper** to map DTO → domain record and add light derived fields.

## After Scaffolding

1. Fill placeholders in `sources/*.yaml` (paths/URLs/auth/etc.).
2. Reference your stream id in `dataset.yaml` under `stream: <stream_id>`
   and select a `field` for each feature/target.
3. From the plugin root, reinstall it if entry points were added:

```bash
python -m pip install -e .
```

## Troubleshooting

- **Entry points not found**: reinstall the plugin (`pip install -e ...`).
- **Import errors with hyphenated plugin names**: import paths use underscores
  (e.g. `my_datapipeline`), even if the folder is `my-datapipeline`.
- **Empty output**: check source YAML paths and parser/mapper logic.
