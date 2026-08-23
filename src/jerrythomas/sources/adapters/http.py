from collections.abc import Iterator, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from jerrythomas.sources.ports import SourceResource, SourceTransport


class HttpTransport(SourceTransport):
    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        chunk_size: int = 64 * 1024,
        timeout_seconds: float | None = None,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.params = dict(params or {})
        self.chunk_size = chunk_size
        self.timeout_seconds = timeout_seconds

    def _build_url(self) -> str:
        if not self.params:
            return self.url
        parsed = urlparse(self.url)
        existing = parse_qsl(parsed.query, keep_blank_values=True)
        merged = existing + list(self.params.items())
        query = urlencode(merged, doseq=True)
        return urlunparse(parsed._replace(query=query))

    def resources(self) -> Iterator[SourceResource]:
        req_url = self._build_url()
        req = Request(req_url, headers=self.headers)

        def byte_stream() -> Iterator[bytes]:
            try:
                resp = urlopen(req, timeout=self.timeout_seconds)
            except (URLError, HTTPError) as exc:
                raise RuntimeError(f"failed to fetch {self.url}: {exc}") from exc
            with resp:
                expected_bytes = getattr(resp, "length", None)
                received_bytes = 0
                while True:
                    chunk = resp.read(self.chunk_size)
                    if not chunk:
                        if (
                            expected_bytes is not None
                            and received_bytes != expected_bytes
                        ):
                            raise RuntimeError(
                                f"failed to fetch {self.url}: response ended after "
                                f"{received_bytes} of {expected_bytes} bytes"
                            )
                        break
                    received_bytes += len(chunk)
                    yield chunk

        yield SourceResource(uri=req_url, stream=byte_stream())
