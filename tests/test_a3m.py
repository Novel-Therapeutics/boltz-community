"""Tests for A3M parsing, especially UniRef-based pairing keys (#627)."""

import io

import pytest

# The import chain a3m → types → rdkit may not be available locally.
pytest.importorskip("rdkit", reason="rdkit not installed")
from boltz.data.parse.a3m import _parse_a3m, parse_a3m


class TestPairingKeysFromUniRef:
    """When taxonomy=None, UniRef headers should produce deterministic pairing keys."""

    SAMPLE_A3M = """\
>query_sequence
ACDEF
>UniRef100_A0A001 some description
ACDEG
>UniRef90_B0B002 another description
ACD-F
>UniRef50_C0C003
ACDEY
>unknown_header
ACDEK
"""

    def _parse(self, text, taxonomy=None, max_seqs=None):
        return _parse_a3m(io.StringIO(text), taxonomy=taxonomy, max_seqs=max_seqs)

    def test_uniref_headers_get_positive_pairing_keys(self):
        """UniRef sequences should get positive pairing keys (not -1)."""
        msa = self._parse(self.SAMPLE_A3M)
        # 5 unique sequences: query + 3 UniRef + 1 unknown
        assert len(msa.sequences) == 5
        # Sequences 1,2,3 are UniRef — should have positive taxonomy
        for i in [1, 2, 3]:
            tax = msa.sequences[i]["taxonomy"]
            assert tax > 0, f"Sequence {i} should have positive pairing key, got {tax}"

    def test_non_uniref_header_gets_minus_one(self):
        """Non-UniRef headers (including query) should get taxonomy=-1."""
        msa = self._parse(self.SAMPLE_A3M)
        # query (index 0) has >query_sequence — not UniRef
        assert msa.sequences[0]["taxonomy"] == -1
        # unknown_header (index 4) — not UniRef
        assert msa.sequences[4]["taxonomy"] == -1

    def test_pairing_keys_are_deterministic(self):
        """Same UniRef ID should always produce the same pairing key (uses crc32, not hash)."""
        msa1 = self._parse(self.SAMPLE_A3M)
        msa2 = self._parse(self.SAMPLE_A3M)
        for i in range(len(msa1.sequences)):
            assert msa1.sequences[i]["taxonomy"] == msa2.sequences[i]["taxonomy"]

    def test_different_uniref_ids_get_different_keys(self):
        """Different UniRef IDs should (almost certainly) get different keys."""
        msa = self._parse(self.SAMPLE_A3M)
        keys = {msa.sequences[i]["taxonomy"] for i in [1, 2, 3]}
        assert len(keys) == 3, f"Expected 3 distinct keys, got {keys}"

    def test_same_uniref_across_chains_pairs_correctly(self):
        """The same UniRef ID in two different chain A3Ms should get the same key."""
        chain_a = """\
>query_A
ACDEF
>UniRef90_X0X999
ACDEG
"""
        chain_b = """\
>query_B
KLMNO
>UniRef90_X0X999
KLMNP
"""
        msa_a = self._parse(chain_a)
        msa_b = self._parse(chain_b)
        # Both chains' second sequence shares UniRef90_X0X999
        key_a = msa_a.sequences[1]["taxonomy"]
        key_b = msa_b.sequences[1]["taxonomy"]
        assert key_a == key_b, f"Same UniRef ID should produce same key: {key_a} vs {key_b}"
        assert key_a > 0

    def test_taxonomy_db_path_still_works(self):
        """When taxonomy dict is provided, the original lookup path is used."""
        taxonomy = {"A0A001": 12345}
        a3m = """\
>UniRef100_A0A001
ACDEF
>UniRef100_UNKNOWN
ACDEG
"""
        msa = self._parse(a3m, taxonomy=taxonomy)
        assert msa.sequences[0]["taxonomy"] == 12345
        # Unknown UniRef100 ID gets -1 via taxonomy lookup miss
        assert msa.sequences[1]["taxonomy"] == -1

    def test_pairing_key_fits_int32(self):
        """Pairing keys must fit in int32 (the taxonomy field dtype)."""
        msa = self._parse(self.SAMPLE_A3M)
        for seq in msa.sequences:
            tax = int(seq["taxonomy"])
            assert -(2**31) <= tax <= 2**31 - 1, f"Key {tax} doesn't fit int32"

    def test_max_seqs_respected(self):
        """max_seqs should limit the number of sequences parsed."""
        msa = self._parse(self.SAMPLE_A3M, max_seqs=2)
        assert len(msa.sequences) == 2


class TestParseA3mFile:
    """Test the file-level parse_a3m function."""

    def test_plain_a3m(self, tmp_path):
        f = tmp_path / "test.a3m"
        f.write_text(">query\nACDEF\n>UniRef90_Z0Z\nACDEG\n")
        msa = parse_a3m(f, taxonomy=None)
        assert len(msa.sequences) == 2
        assert msa.sequences[1]["taxonomy"] > 0

    def test_gzipped_a3m(self, tmp_path):
        import gzip

        f = tmp_path / "test.a3m.gz"
        with gzip.open(str(f), "wt") as gz:
            gz.write(">query\nACDEF\n>UniRef90_Z0Z\nACDEG\n")
        msa = parse_a3m(f, taxonomy=None)
        assert len(msa.sequences) == 2
        assert msa.sequences[1]["taxonomy"] > 0
