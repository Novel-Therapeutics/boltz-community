"""Tests for boltz.data.parse.a3m — parse_a3m."""

from pathlib import Path

import numpy as np
import pytest

from boltz.data.parse.a3m import parse_a3m


EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "msa"


class TestParseA3m:
    """Tests for parse_a3m."""

    def test_parse_real_file(self):
        """Parse examples/msa/seq1.a3m, verify MSA structure and token IDs."""
        path = EXAMPLES_DIR / "seq1.a3m"
        if not path.exists():
            pytest.skip("seq1.a3m not found in examples/msa/")

        msa = parse_a3m(path, taxonomy=None, max_seqs=None)

        assert len(msa.sequences) > 0
        assert len(msa.residues) > 0
        # First sequence should have contiguous residues
        seq0 = msa.sequences[0]
        assert seq0["seq_idx"] == 0
        assert seq0["res_end"] > seq0["res_start"]
        # All residue tokens should be valid (non-negative)
        assert (msa.residues["res_type"] >= 0).all()

    def test_max_seqs(self):
        """max_seqs=3 returns exactly 3 sequences."""
        path = EXAMPLES_DIR / "seq1.a3m"
        if not path.exists():
            pytest.skip("seq1.a3m not found in examples/msa/")

        msa = parse_a3m(path, taxonomy=None, max_seqs=3)
        assert len(msa.sequences) == 3

    def test_deletions_tracked(self, synthetic_a3m_path):
        """Lowercase insertions → correct deletion counts."""
        msa = parse_a3m(synthetic_a3m_path, taxonomy=None, max_seqs=None)

        # "ACdDEF" has insertion 'd' before D → deletion count 1 at some res_idx
        # "aCDEF" has insertion 'a' before C → deletion count 1
        assert len(msa.deletions) > 0

        # Check that deletion values are positive
        for d in msa.deletions:
            assert d["deletion"] > 0

    def test_duplicates_skipped(self, synthetic_a3m_path):
        """Identical sequences are deduplicated."""
        msa = parse_a3m(synthetic_a3m_path, taxonomy=None, max_seqs=None)

        # "ACDEF" appears twice (query and seq4), should be deduplicated
        # So we expect fewer sequences than 5
        assert len(msa.sequences) < 5

    def test_gap_handling(self, synthetic_a3m_path):
        """'-' chars map to gap token."""
        from boltz.data import const

        msa = parse_a3m(synthetic_a3m_path, taxonomy=None, max_seqs=None)

        # seq3 is "A-DEF" — the '-' should map to gap token
        gap_token = const.token_ids[const.prot_letter_to_token["-"]]

        # Find seq3 (index 2 after dedup) and check for gap
        found_gap = False
        for seq in msa.sequences:
            res_start = seq["res_start"]
            res_end = seq["res_end"]
            residues = msa.residues[res_start:res_end]
            if gap_token in residues["res_type"]:
                found_gap = True
                break
        assert found_gap

    def test_gzipped(self, synthetic_a3m_gz_path):
        """Gzipped A3M parses identically."""
        msa = parse_a3m(synthetic_a3m_gz_path, taxonomy=None, max_seqs=None)

        assert len(msa.sequences) > 0
        assert len(msa.residues) > 0


class TestA3mNullBytes:
    """A3M parser must handle null bytes without crashing.

    Some MSA server responses contain \\x00 bytes which would cause KeyError
    in prot_letter_to_token lookup if not stripped.
    """

    def test_null_bytes_stripped(self, tmp_path):
        """A3M file with null bytes parses without error."""
        content = ">query\nACDEF\x00\n>seq2\nAC\x00DEF\n"
        p = tmp_path / "null.a3m"
        p.write_text(content)

        msa = parse_a3m(p, taxonomy=None, max_seqs=None)
        assert len(msa.sequences) > 0
        assert len(msa.residues) > 0

    def test_null_only_line_skipped(self, tmp_path):
        """A line containing only null bytes is skipped."""
        content = ">query\nACDEF\n\x00\x00\x00\n>seq2\nACDEF\n"
        p = tmp_path / "null2.a3m"
        p.write_text(content)

        msa = parse_a3m(p, taxonomy=None, max_seqs=None)
        assert len(msa.sequences) >= 1
