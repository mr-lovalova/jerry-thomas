import base64
import html
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jerrythomas.artifacts.models import (
    ListVectorMetadataEntry,
    VectorMetadataEntry,
)
from jerrythomas.transforms.utils import is_missing


Status = Literal["absent", "present", "null"]
_ABSENT = object()
_STATUS_CODE = {"absent": 0, "present": 1, "null": 2}


@dataclass(frozen=True)
class MatrixCell:
    status: Status
    elements: tuple[Status, ...] = ()


@dataclass(frozen=True)
class MatrixRow:
    group: str
    features: tuple[MatrixCell, ...]
    targets: tuple[MatrixCell, ...]


@dataclass(frozen=True)
class AvailabilityMatrix:
    feature_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    rows: tuple[MatrixRow, ...]

    def output_rows(self) -> Iterator[dict[str, Any]]:
        for row in self.rows:
            for identifier, cell in zip(self.feature_ids, row.features):
                output: dict[str, Any] = {
                    "vector": "feature",
                    "identifier": identifier,
                    "group": row.group,
                    "status": cell.status,
                }
                if cell.elements:
                    output["elements"] = list(cell.elements)
                yield output
            for identifier, cell in zip(self.target_ids, row.targets):
                output = {
                    "vector": "target",
                    "identifier": identifier,
                    "group": row.group,
                    "status": cell.status,
                }
                if cell.elements:
                    output["elements"] = list(cell.elements)
                yield output


class MatrixBuilder:
    """Build one explicitly bounded availability matrix."""

    def __init__(
        self,
        feature_entries: Sequence[VectorMetadataEntry],
        target_entries: Sequence[VectorMetadataEntry],
        max_cells: int,
    ) -> None:
        self._feature_entries = tuple(feature_entries)
        self._target_entries = tuple(target_entries)
        self._feature_ids = {entry.id for entry in feature_entries}
        self._target_ids = {entry.id for entry in target_entries}
        self._max_cells = max_cells
        self._row_width = sum(
            _entry_width(entry)
            for entry in (*self._feature_entries, *self._target_entries)
        )
        self._rows: list[MatrixRow] = []

    def add(
        self,
        group_key: object,
        features: Mapping[str, Any],
        targets: Mapping[str, Any],
    ) -> None:
        unknown_features = set(features) - self._feature_ids
        if unknown_features:
            identifier = min(unknown_features)
            raise ValueError(
                f"Feature vector contains ID {identifier!r} missing from metadata. "
                "Rebuild vector metadata."
            )
        unknown_targets = set(targets) - self._target_ids
        if unknown_targets:
            identifier = min(unknown_targets)
            raise ValueError(
                f"Target vector contains ID {identifier!r} missing from metadata. "
                "Rebuild vector metadata."
            )
        if not self._row_width:
            return

        next_cells = (len(self._rows) + 1) * self._row_width
        if next_cells > self._max_cells:
            raise ValueError(
                f"Availability matrix exceeds max_cells={self._max_cells} at "
                f"sample {len(self._rows) + 1}. Increase matrix options.max_cells "
                "or inspect a smaller dataset window."
            )

        self._rows.append(
            MatrixRow(
                group=_format_group_key(group_key),
                features=_cells(self._feature_entries, features),
                targets=_cells(self._target_entries, targets),
            )
        )

    def finish(self) -> AvailabilityMatrix:
        return AvailabilityMatrix(
            feature_ids=tuple(entry.id for entry in self._feature_entries),
            target_ids=tuple(entry.id for entry in self._target_entries),
            rows=tuple(self._rows),
        )


def _entry_width(entry: VectorMetadataEntry) -> int:
    if isinstance(entry, ListVectorMetadataEntry):
        return entry.length
    return 1


def _cells(
    entries: Sequence[VectorMetadataEntry],
    values: Mapping[str, Any],
) -> tuple[MatrixCell, ...]:
    return tuple(_cell(entry, values.get(entry.id, _ABSENT)) for entry in entries)


def _cell(entry: VectorMetadataEntry, value: object) -> MatrixCell:
    if value is _ABSENT:
        return MatrixCell("absent")
    if isinstance(entry, ListVectorMetadataEntry):
        if is_missing(value):
            null: Status = "null"
            elements = (null,) * entry.length
            return MatrixCell("null", elements)
        if not isinstance(value, list):
            raise ValueError(f"List vector {entry.id!r} contains a scalar value.")
        expected = entry.length
        if len(value) != expected:
            raise ValueError(
                f"List vector {entry.id!r} has length {len(value)}; expected {expected}."
            )
        elements = tuple(
            "null" if is_missing(element) else "present" for element in value
        )
        status: Status = "present" if "present" in elements else "null"
        return MatrixCell(status, elements)
    if isinstance(value, list):
        raise ValueError(f"Scalar vector {entry.id!r} contains a list value.")
    return MatrixCell("null" if is_missing(value) else "present")


def _format_group_key(group_key: object) -> str:
    if isinstance(group_key, tuple):
        return ", ".join(str(part) for part in group_key)
    return str(group_key)


def render_matrix_html(matrix: AvailabilityMatrix) -> str:
    row_labels = tuple(row.group for row in matrix.rows)
    sections = [
        _html_table(
            "Feature Availability",
            "features",
            matrix.feature_ids,
            row_labels,
            tuple(row.features for row in matrix.rows),
        ),
        _html_table(
            "Target Availability",
            "targets",
            matrix.target_ids,
            row_labels,
            tuple(row.targets for row in matrix.rows),
        ),
    ]
    return (
        "<html><head><meta charset='utf-8'>"
        f"<style>{_STYLE}</style>"
        "<title>Vector Availability</title></head><body>"
        "<main><header><h1>Availability Matrix</h1>"
        "<div class='legend'><span class='present'>Present</span>"
        "<span class='null'>Null</span><span class='absent'>Absent</span></div>"
        "</header>"
        f"<script>{_SCRIPT}</script>"
        f"{''.join(sections)}"
        "</main></body></html>"
    )


def _html_table(
    title: str,
    table_id: str,
    identifiers: tuple[str, ...],
    row_labels: tuple[str, ...],
    rows: tuple[tuple[MatrixCell, ...], ...],
) -> str:
    if not identifiers:
        return f"<section><h2>{html.escape(title)}</h2><p>No data.</p></section>"

    codes = bytearray()
    sub: dict[int, list[int]] = {}
    for row_cells in rows:
        for cell in row_cells:
            cell_index = len(codes)
            codes.append(_STATUS_CODE[cell.status])
            if cell.elements:
                sub[cell_index] = [_STATUS_CODE[status] for status in cell.elements]

    payload = {
        "rows": row_labels,
        "encoded": base64.b64encode(bytes(codes)).decode("ascii"),
        "sub": sub,
        "columns": len(identifiers),
    }
    encoded_payload = json.dumps(payload).replace("<", "\\u003c")
    headers = "".join(
        f"<th scope='col'>{html.escape(identifier)}</th>" for identifier in identifiers
    )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<div class='table' id='{table_id}-container'><table><thead><tr>"
        f"<th scope='col' class='group'>Group</th>{headers}</tr></thead>"
        f"<tbody id='{table_id}-body' data-colspan='{len(identifiers) + 1}'></tbody>"
        "</table></div></section>"
        f"<script>setupMatrix('{table_id}', {encoded_payload});</script>"
    )


_STYLE = """
* { box-sizing: border-box; }
body { margin: 24px; background: #f7f7f8; color: #222; font-family: sans-serif; }
main { overflow: hidden; border: 1px solid #ccc; border-radius: 8px; background: white; }
header { display: flex; align-items: center; gap: 24px; padding: 16px 20px; }
h1, h2 { margin: 0; }
h1 { font-size: 22px; }
h2 { padding: 16px 20px 10px; font-size: 18px; }
.legend { display: flex; gap: 16px; font-size: 13px; }
.legend span::before { content: ''; display: inline-block; width: 12px; height: 12px;
  margin-right: 6px; border-radius: 2px; vertical-align: -1px; background: #bbb; }
.legend .present::before, td.present, .sub .present { background: #2ecc71; }
.legend .null::before, td.null, .sub .null { background: #f1c40f; }
.legend .absent::before, td.absent, .sub .absent { background: #e74c3c; }
section { border-top: 1px solid #eee; }
.table { overflow: auto; max-height: 55vh; margin: 0 20px 20px; border: 1px solid #ccc; }
table { min-width: 100%; border-collapse: collapse; }
th, td { min-width: 28px; height: 28px; border: 1px solid #ddd; text-align: center; }
thead th { position: sticky; top: 0; z-index: 2; background: #f2f2f4; }
th.group { position: sticky; left: 0; z-index: 3; min-width: 160px; padding: 0 6px; text-align: left; }
tbody th.group { z-index: 1; background: white; font-weight: normal; }
.sub { display: flex; gap: 2px; height: 24px; padding: 2px; }
.sub span { flex: 1; min-width: 8px; border-radius: 2px; }
"""


_SCRIPT = """
function setupMatrix(id, payload) {
  const container = document.getElementById(id + '-container');
  const body = document.getElementById(id + '-body');
  const raw = atob(payload.encoded);
  const data = Uint8Array.from(raw, c => c.charCodeAt(0));
  const classes = ['absent', 'present', 'null'];
  const rowHeight = 28;
  let previousStart = -1;

  function render() {
    const buffer = 20;
    const start = Math.max(0, Math.floor(container.scrollTop / rowHeight) - buffer);
    const count = Math.ceil(container.clientHeight / rowHeight) + buffer * 2;
    const end = Math.min(payload.rows.length, start + count);
    if (start === previousStart) return;
    previousStart = start;
    let output = '';
    if (start) output += spacer(start * rowHeight);
    for (let row = start; row < end; row++) output += matrixRow(row);
    if (end < payload.rows.length) output += spacer((payload.rows.length - end) * rowHeight);
    body.innerHTML = output;
  }

  function spacer(height) {
    return `<tr><td colspan="${body.dataset.colspan}" style="height:${height}px;border:0"></td></tr>`;
  }

  function matrixRow(row) {
    let cells = '';
    for (let column = 0; column < payload.columns; column++) {
      const index = row * payload.columns + column;
      const status = classes[data[index]] || 'absent';
      const elements = payload.sub[index];
      if (elements && elements.length) {
        const spans = elements.map(code => `<span class="${classes[code] || 'absent'}"></span>`).join('');
        cells += `<td title="${status}"><div class="sub">${spans}</div></td>`;
      } else {
        cells += `<td class="${status}" title="${status}"></td>`;
      }
    }
    return `<tr><th scope="row" class="group">${escapeHtml(payload.rows[row])}</th>${cells}</tr>`;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    })[character]);
  }

  container.addEventListener('scroll', () => requestAnimationFrame(render));
  render();
}
"""
