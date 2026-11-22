# ruff: noqa: INP001

from __future__ import annotations

import io
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import requests

from boltz.data.msa import mmseqs2

if TYPE_CHECKING:
    from pathlib import Path


def _msa_archive(contents: bytes = b">101\nACDE\n", *, paired: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        names = (
            ("pair.a3m",)
            if paired
            else ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
        )
        for name in names:
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


def test_pairing_identical_sequences_needs_no_server(monkeypatch, tmp_path):
    """Copies of one sequence have no inter-sequence pairing to request."""

    def unexpected_request(*args, **kwargs):
        raise AssertionError("Identical sequences must not request pairing")

    monkeypatch.setattr(mmseqs2.requests, "post", unexpected_request)
    for sequences in ("ACDE", ["ACDE"], ["ACDE", "ACDE"]):
        expected_count = 1 if isinstance(sequences, str) else len(sequences)
        assert (
            mmseqs2.run_mmseqs2(
                sequences, prefix=str(tmp_path / "query"), use_pairing=True
            )
            == [""] * expected_count
        )
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(params=["native", "reference"])
def input_processor(request):
    """Exercise both CLI data pipelines without modifying the pinned reference."""
    if request.param == "native":
        from boltz.main import process_input
    else:
        from boltz.upstream.bootstrap import _load_source
        from boltz.upstream.contract import engine_root

        reference = _load_source(
            "boltz.upstream._msa_test_reference_cli",
            engine_root().parent / "reference/boltz/main.py",
        )
        process_input = reference.process_input
    return process_input


@pytest.mark.parametrize(
    "chemistry",
    [
        {"modifications": [{"position": 2, "ccd": "ALA"}]},
        {"cyclic": True},
    ],
)
@pytest.mark.parametrize("heteromer", [False, True])
def test_chemistry_groups_share_sequence_msa_content(
    monkeypatch, tmp_path, input_processor, chemistry, heteromer
):
    """Chemistry splitting retains unpaired homomers and normal heteromer pairing."""
    import numpy as np
    import yaml

    from boltz.data.types import MSA, Record
    from tests.data.parse.test_schema import _mock_residue_mol

    monkeypatch.setattr("boltz.data.parse.schema.get_mol", _mock_residue_mol)
    schema = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "ACDE"}},
            {"protein": {"id": "B", "sequence": "ACDE", **chemistry}},
        ]
    }
    if heteromer:
        schema["sequences"].append({"protein": {"id": "C", "sequence": "AAAA"}})
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(schema))

    submissions = []

    def submit(url, *, data, **kwargs):
        submissions.append((url, data["q"]))
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {"status": "COMPLETE", "id": "test-ticket"}
        return response

    def download(url, **kwargs):
        endpoint, query = submissions[-1]
        contents = b"\x00".join(
            f">{101 + idx}\n{seq}\n".encode()
            for idx, seq in enumerate(query.splitlines()[1::2])
        )
        return _download_response(
            _msa_archive(contents, paired=endpoint.endswith("/pair"))
        )

    monkeypatch.setattr(mmseqs2.requests, "post", submit)
    monkeypatch.setattr(mmseqs2.requests, "get", download)
    monkeypatch.setattr(mmseqs2.time, "sleep", lambda _seconds: None)
    directories = {}
    for name in (
        "msa_dir",
        "mol_dir",
        "processed_msa_dir",
        "processed_constraints_dir",
        "processed_templates_dir",
        "processed_mols_dir",
        "structure_dir",
        "records_dir",
    ):
        directories[name] = tmp_path / name
        directories[name].mkdir()

    assert (
        input_processor(
            path=path,
            ccd={},
            boltz2=True,
            use_msa_server=True,
            msa_server_url="https://msa.invalid",
            msa_pairing_strategy="greedy",
            msa_server_username=None,
            msa_server_password=None,
            api_key_header=None,
            api_key_value=None,
            max_msa_seqs=16,
            **directories,
        )
        is None
    )
    expected_endpoints = ["pair", "msa"] if heteromer else ["msa"]
    assert [url.rsplit("/", 1)[-1] for url, _ in submissions] == expected_endpoints
    expected_sequences = ["ACDE", "AAAA"] if heteromer else ["ACDE"]
    assert all(
        query.splitlines()[1::2] == expected_sequences for _, query in submissions
    )

    record = Record.load(directories["records_dir"] / "input.json")
    assert record.chains[0].entity_id != record.chains[1].entity_id
    alignments = [
        MSA.load(directories["processed_msa_dir"] / f"{c.msa_id}.npz")
        for c in record.chains[:2]
    ]
    for field in ("sequences", "residues", "deletions"):
        np.testing.assert_array_equal(
            getattr(alignments[0], field), getattr(alignments[1], field)
        )
    if not heteromer:
        assert set(alignments[0].sequences["taxonomy"]) == {-1}
