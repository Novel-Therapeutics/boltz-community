"""Tests for MSA trimming when query is longer than input sequence."""

import numpy as np
import pytest

from boltz.data.feature.featurizerv2 import _trim_msa_to_input, dummy_msa
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def _make_msa(res_types: list[list[int]], deletions=None) -> MSA:
    """Build an MSA from a list of sequences (each a list of res_type ints)."""
    all_residues = []
    all_sequences = []
    all_deletions = [] if deletions is None else deletions
    for seq_idx, types in enumerate(res_types):
        start = len(all_residues)
        all_residues.extend(types)
        end = len(all_residues)
        all_sequences.append((seq_idx, -1, start, end, 0, 0))
    return MSA(
        residues=np.array([(t,) for t in all_residues], dtype=MSAResidue),
        deletions=np.array(all_deletions, dtype=MSADeletion),
        sequences=np.array(all_sequences, dtype=MSASequence),
    )


def _make_input_residues(types: list[int]) -> np.ndarray:
    """Build input residues array matching featurizer format."""
    dt = np.dtype([("res_type", "i1")])
    return np.array([(t,) for t in types], dtype=dt)


class TestTrimMsaToInput:
    """Tests for _trim_msa_to_input."""

    def test_exact_subsequence_at_start(self):
        """Input matches start of MSA query."""
        # MSA query: [1, 2, 3, 4, 5], input: [1, 2, 3]
        msa = _make_msa([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        input_res = _make_input_residues([1, 2, 3])
        input_types = np.array([1, 2, 3])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        assert len(result.sequences) == 2
        first = result.sequences[0]
        trimmed = result.residues[first["res_start"]:first["res_end"]]
        assert len(trimmed) == 3
        assert list(trimmed["res_type"]) == [1, 2, 3]

    def test_exact_subsequence_in_middle(self):
        """Input matches middle of MSA query."""
        # MSA query: [1, 2, 3, 4, 5], input: [2, 3, 4]
        msa = _make_msa([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
        input_res = _make_input_residues([2, 3, 4])
        input_types = np.array([2, 3, 4])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        assert len(result.sequences) == 2
        # Check second sequence was trimmed at same offset
        second = result.sequences[1]
        trimmed = result.residues[second["res_start"]:second["res_end"]]
        assert len(trimmed) == 3
        assert list(trimmed["res_type"]) == [12, 13, 14]

    def test_exact_subsequence_at_end(self):
        """Input matches end of MSA query."""
        # MSA query: [1, 2, 3, 4, 5], input: [3, 4, 5]
        msa = _make_msa([[1, 2, 3, 4, 5]])
        input_res = _make_input_residues([3, 4, 5])
        input_types = np.array([3, 4, 5])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        first = result.sequences[0]
        trimmed = result.residues[first["res_start"]:first["res_end"]]
        assert list(trimmed["res_type"]) == [3, 4, 5]

    def test_no_match_returns_dummy(self):
        """Input doesn't match MSA query at all — returns dummy."""
        msa = _make_msa([[1, 2, 3, 4, 5]])
        input_res = _make_input_residues([9, 8, 7])
        input_types = np.array([9, 8, 7])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        # Dummy MSA has exactly 1 sequence
        assert len(result.sequences) == 1
        first = result.sequences[0]
        trimmed = result.residues[first["res_start"]:first["res_end"]]
        assert len(trimmed) == 3

    def test_minor_mismatch_accepted(self):
        """Small number of mismatches (< 5%) should still trim."""
        # 100 residue sequence with 1 mismatch at position 50
        query = list(range(1, 121))  # 120 residues
        input_seq = list(range(11, 111))  # 100 residues, offset 10
        input_seq[50] = 99  # 1 mismatch out of 100 = 1%

        msa = _make_msa([query])
        input_res = _make_input_residues(input_seq)
        input_types = np.array(input_seq)
        msa_query_types = np.array(query)

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        # Should succeed (1% < 5% threshold)
        assert len(result.sequences) == 1
        first = result.sequences[0]
        trimmed = result.residues[first["res_start"]:first["res_end"]]
        assert len(trimmed) == 100
        # Query residues should be overwritten with input types
        assert trimmed["res_type"][50] == 99

    def test_too_many_mismatches_returns_dummy(self):
        """Too many mismatches (> 5%) should return dummy."""
        query = [1] * 20
        input_seq = [1] * 8 + [9] * 2  # 10 residues, 2 mismatches = 20%

        msa = _make_msa([query])
        input_res = _make_input_residues(input_seq)
        input_types = np.array(input_seq)
        msa_query_types = np.array(query)

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        # Should be dummy (1 sequence only)
        assert len(result.sequences) == 1

    def test_deletions_trimmed(self):
        """Deletions within the trimmed region are kept and re-indexed."""
        msa = _make_msa([[1, 2, 3, 4, 5]])
        # Add deletions: one inside trim region, one outside
        msa = MSA(
            residues=msa.residues,
            deletions=np.array([(1, 3), (3, 2), (4, 1)], dtype=MSADeletion),
            sequences=np.array([(0, -1, 0, 5, 0, 3)], dtype=MSASequence),
        )
        input_res = _make_input_residues([2, 3, 4])
        input_types = np.array([2, 3, 4])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        # Deletions at original idx 1,3 are in range [1,4), should be kept
        # idx 4 is out of range [1,4)
        first = result.sequences[0]
        dels = result.deletions[first["del_start"]:first["del_end"]]
        assert len(dels) == 2
        # Re-indexed: original 1 -> 0, original 3 -> 2
        assert dels[0]["res_idx"] == 0
        assert dels[0]["deletion"] == 3
        assert dels[1]["res_idx"] == 2
        assert dels[1]["deletion"] == 2

    def test_multiple_sequences_preserved(self):
        """All MSA sequences of correct length are trimmed."""
        seqs = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [10, 20, 30, 40, 50]]
        msa = _make_msa(seqs)
        input_res = _make_input_residues([2, 3, 4])
        input_types = np.array([2, 3, 4])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        assert len(result.sequences) == 3
        for i, expected in enumerate([[2, 3, 4], [4, 3, 2], [20, 30, 40]]):
            seq = result.sequences[i]
            trimmed = result.residues[seq["res_start"]:seq["res_end"]]
            assert list(trimmed["res_type"]) == expected

    def test_skips_sequences_with_wrong_length(self):
        """Sequences that don't match query length are skipped."""
        msa = _make_msa([[1, 2, 3, 4, 5], [1, 2, 3]])  # second is shorter
        input_res = _make_input_residues([2, 3, 4])
        input_types = np.array([2, 3, 4])
        msa_query_types = np.array([1, 2, 3, 4, 5])

        result = _trim_msa_to_input(
            msa, input_res, input_types, msa_query_types, 0, "test", "warn"
        )

        # Only the first (length-5) sequence should be kept
        assert len(result.sequences) == 1
