# ruff: noqa: INP001

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import Mock

import requests

from boltz.data.msa import mmseqs2

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _msa_archive() -> bytes:
    buffer = io.BytesIO()
    contents = b">101\nACDE\n"
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"):
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return buffer.getvalue()


def _download_response(
    content: bytes,
    *,
    status_code: int = 200,
    error: requests.HTTPError | None = None,
) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.content = content
    response.raise_for_status.side_effect = error
    return response


def test_run_mmseqs2_retries_http_errors_and_invalid_archives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Only a successful, valid MSA archive is cached after transient failures."""
    submit_response = Mock(spec=requests.Response)
    submit_response.status_code = 200
    submit_response.json.return_value = {"status": "COMPLETE", "id": "test-ticket"}

    valid_archive = _msa_archive()
    download_responses = [
        _download_response(
            b"service unavailable",
            status_code=503,
            error=requests.HTTPError("503 Server Error"),
        ),
        _download_response(b"not a tar archive"),
        _download_response(valid_archive),
    ]
    response_iterator = iter(download_responses)
    output_path = tmp_path / "query_env" / "out.tar.gz"

    def next_download(_url: str, **_kwargs: object) -> Mock:
        assert not output_path.exists()
        return next(response_iterator)

    get = Mock(side_effect=next_download)
    monkeypatch.setattr(mmseqs2.requests, "post", Mock(return_value=submit_response))
    monkeypatch.setattr(mmseqs2.requests, "get", get)
    monkeypatch.setattr(mmseqs2.time, "sleep", lambda _seconds: None)

    alignments = mmseqs2.run_mmseqs2("ACDE", prefix=str(tmp_path / "query"))

    assert get.call_count == len(download_responses)
    assert alignments == [">101\nACDE\n>101\nACDE\n"]
    assert output_path.read_bytes() == valid_archive
