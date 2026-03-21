"""Tests for performance-refactored featurizer functions.

Covers:
  - prepare_msa_arrays: vectorized pairing fill and msa_sequences slot assignment
  - process_atom_features: pre-allocated atom_to_token and ref_space_uid arrays
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("numba")

import numba
import torch
from numba import types

from boltz.data import const
from boltz.data.feature.featurizer import prepare_msa_arrays, process_atom_features
from boltz.data.types import (
    MSA,
    Atom,
    Bond,
    Chain,
    Connection,
    Interface,
    MSADeletion,
    MSAResidue,
    MSASequence,
    Residue,
    Structure,
    Token,
    TokenBond,
    Tokenized,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msa(res_types: list[int]) -> MSA:
    """Build a single-sequence MSA from a list of residue type ints."""
    residues = np.array([(t,) for t in res_types], dtype=MSAResidue)
    sequences = np.array([(0, -1, 0, len(res_types), 0, 0)], dtype=MSASequence)
    return MSA(
        residues=residues,
        sequences=sequences,
        deletions=np.array([], dtype=MSADeletion),
    )


def _empty_msa() -> MSA:
    return MSA(
        residues=np.array([], dtype=MSAResidue),
        sequences=np.array([], dtype=MSASequence),
        deletions=np.array([], dtype=MSADeletion),
    )


def _empty_deletions():
    return numba.typed.Dict.empty(
        key_type=types.UniTuple(types.int64, 3),
        value_type=types.int64,
    )


def _make_minimal_tokenized(n_tokens: int = 3, atoms_per_token: int = 2) -> Tokenized:
    """Build a minimal Tokenized with n_tokens protein tokens, atoms_per_token each.

    Setting atoms_per_token < 3 keeps frame computation on the trivial branch.
    Tokens 0 and 2 share residue index 0; token 1 has residue index 1.
    This lets us test ref_space_uid deduplication.
    """
    n_atoms = n_tokens * atoms_per_token

    atoms = np.zeros(n_atoms, dtype=Atom)
    for i in range(n_atoms):
        atoms[i]["name"] = [33, 33, 0, 0]      # "CA"
        atoms[i]["element"] = 6                 # Carbon
        atoms[i]["charge"] = 0
        atoms[i]["coords"] = [float(i), 0.0, 0.0]
        atoms[i]["conformer"] = [float(i), 0.0, 0.0]
        atoms[i]["is_present"] = True
        atoms[i]["chirality"] = 0

    # res_idx pattern: [0, 1, 0] so tokens 0 and 2 share the same residue
    res_idx_map = [0, 1, 0] if n_tokens == 3 else list(range(n_tokens))

    residues = np.zeros(n_tokens, dtype=Residue)
    for i in range(n_tokens):
        residues[i]["name"] = "ALA"
        residues[i]["res_type"] = 0
        residues[i]["res_idx"] = i
        residues[i]["atom_idx"] = i * atoms_per_token
        residues[i]["atom_num"] = atoms_per_token

    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = "A"
    chains[0]["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_atoms
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = n_tokens

    structure = Structure(
        atoms=atoms,
        bonds=np.array([], dtype=Bond),
        residues=residues,
        chains=chains,
        connections=np.array([], dtype=Connection),
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(1, dtype=bool),
    )

    tokens = np.zeros(n_tokens, dtype=Token)
    for i in range(n_tokens):
        abs_atom_start = i * atoms_per_token
        tokens[i]["token_idx"] = i
        tokens[i]["atom_idx"] = abs_atom_start
        tokens[i]["atom_num"] = atoms_per_token
        tokens[i]["res_idx"] = res_idx_map[i] if n_tokens == 3 else i
        tokens[i]["res_type"] = 0
        tokens[i]["asym_id"] = 0
        tokens[i]["entity_id"] = 0
        tokens[i]["mol_type"] = const.chain_type_ids["PROTEIN"]
        tokens[i]["center_idx"] = abs_atom_start
        tokens[i]["disto_idx"] = abs_atom_start
        tokens[i]["resolved_mask"] = True
        tokens[i]["disto_mask"] = True

    return Tokenized(
        tokens=tokens,
        bonds=np.array([], dtype=TokenBond),
        structure=structure,
        msa={},
    )


# ---------------------------------------------------------------------------
# prepare_msa_arrays — vectorized fill
# ---------------------------------------------------------------------------

class TestPrepareMsaArrays:
    """Tests for the vectorized pairing/msa_sequences fill in prepare_msa_arrays."""

    def _tokens_for_chains(self, chain_ids: list[int]) -> np.ndarray:
        """One token per chain, each with res_idx=0."""
        tokens = np.zeros(len(chain_ids), dtype=Token)
        for i, cid in enumerate(chain_ids):
            tokens[i]["asym_id"] = cid
            tokens[i]["res_idx"] = 0
        return tokens

    def test_residue_types_correct_for_nonsequential_chain_ids(self):
        """Each chain's residue type should appear in msa_data at the right token row.

        Using chain IDs 0 and 5 (non-sequential) to expose any column-mapping bugs.
        Chain 0 has res_type=7; chain 5 has res_type=3.
        """
        chain_ids = [0, 5]
        tokens = self._tokens_for_chains(chain_ids)
        msa = {0: _make_msa([7]), 5: _make_msa([3])}
        # Both chains paired at seq_idx 0 in row 0
        pairing = [{0: 0, 5: 0}]
        is_paired = [{0: 1, 5: 1}]

        msa_data, _, _ = prepare_msa_arrays(
            tokens, pairing, is_paired, _empty_deletions(), msa
        )

        assert msa_data.shape == (2, 1)
        assert msa_data[0, 0] == 7, "chain 0's residue type should be 7"
        assert msa_data[1, 0] == 3, "chain 5's residue type should be 3"

    def test_multiple_pairing_rows_with_distinct_seq_indices(self):
        """With two MSA rows and two chains, each row maps to the right sequence."""
        # Chain 0: seq 0 → res_type=10, seq 1 → res_type=20
        # Chain 3: seq 0 → res_type=30, seq 1 → res_type=40
        chain0_msa = _make_msa([10])
        # Add a second sequence manually
        residues = np.array([(10,), (20,)], dtype=MSAResidue)
        sequences = np.array(
            [(0, -1, 0, 1, 0, 0), (1, -1, 1, 2, 0, 0)], dtype=MSASequence
        )
        chain0_msa = MSA(
            residues=residues,
            sequences=sequences,
            deletions=np.array([], dtype=MSADeletion),
        )

        residues3 = np.array([(30,), (40,)], dtype=MSAResidue)
        sequences3 = np.array(
            [(0, -1, 0, 1, 0, 0), (1, -1, 1, 2, 0, 0)], dtype=MSASequence
        )
        chain3_msa = MSA(
            residues=residues3,
            sequences=sequences3,
            deletions=np.array([], dtype=MSADeletion),
        )

        msa = {0: chain0_msa, 3: chain3_msa}
        tokens = self._tokens_for_chains([0, 3])

        # Row 0: chain 0 seq 0, chain 3 seq 1
        # Row 1: chain 0 seq 1, chain 3 seq 0
        pairing = [{0: 0, 3: 1}, {0: 1, 3: 0}]
        is_paired = [{0: 1, 3: 1}, {0: 1, 3: 1}]

        msa_data, _, _ = prepare_msa_arrays(
            tokens, pairing, is_paired, _empty_deletions(), msa
        )

        assert msa_data.shape == (2, 2)
        # Token 0 = chain 0:
        assert msa_data[0, 0] == 10   # row 0, chain 0, seq 0 → res_type 10
        assert msa_data[0, 1] == 20   # row 1, chain 0, seq 1 → res_type 20
        # Token 1 = chain 3:
        assert msa_data[1, 0] == 40   # row 0, chain 3, seq 1 → res_type 40
        assert msa_data[1, 1] == 30   # row 1, chain 3, seq 0 → res_type 30

    def test_empty_msa_chain_leaves_gap_tokens(self):
        """A chain with 0 MSA sequences produces gap tokens in msa_data (seq_idx stays -1 sentinel)."""
        chain_ids = [0, 2]
        tokens = self._tokens_for_chains(chain_ids)
        # Chain 2 has no sequences
        msa = {0: _make_msa([7]), 2: _empty_msa()}

        # For the empty chain, seq_idx should be -1 (no pairing) so msa_data stays at gap
        pairing = [{0: 0, 2: -1}]
        is_paired = [{0: 1, 2: 0}]

        msa_data, _, _ = prepare_msa_arrays(
            tokens, pairing, is_paired, _empty_deletions(), msa
        )

        gap = const.token_ids["-"]
        assert msa_data[1, 0] == gap, "empty-MSA chain should produce gap token"


# ---------------------------------------------------------------------------
# process_atom_features — pre-allocated arrays
# ---------------------------------------------------------------------------

class TestProcessAtomFeaturesPreAllocation:
    """Tests for the pre-allocated atom_to_token and ref_space_uid arrays."""

    def test_atom_to_token_mapping(self):
        """Each atom should map to the token that owns it.

        3 tokens × 2 atoms = 6 real atoms (padded to 32 by the function).
        Expected for first 6: atoms 0-1 → token 0, atoms 2-3 → token 1, atoms 4-5 → token 2.
        """
        n_tokens, apa = 3, 2
        total_atoms = n_tokens * apa
        data = _make_minimal_tokenized(n_tokens=n_tokens, atoms_per_token=apa)
        out = process_atom_features(data)

        # atom_to_token is returned as one-hot (padded_atoms, n_tokens); slice real atoms
        atom_to_token = out["atom_to_token"][:total_atoms].argmax(dim=-1)
        expected = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.equal(atom_to_token, expected), (
            f"atom_to_token mismatch: got {atom_to_token.tolist()}, "
            f"expected {expected.tolist()}"
        )

    def test_ref_space_uid_same_residue_same_uid(self):
        """Atoms belonging to tokens that share (asym_id, res_idx) must get the same uid.

        In the minimal tokenized fixture, token 0 and token 2 both have res_idx=0,
        so their atoms should share a uid distinct from token 1's atoms (res_idx=1).
        """
        data = _make_minimal_tokenized(n_tokens=3, atoms_per_token=2)
        out = process_atom_features(data)

        # ref_space_uid is a 1-D long tensor (integer UID per atom, then padded)
        uid = out["ref_space_uid"]
        # atoms 0-1 (token 0, res_idx=0) and atoms 4-5 (token 2, res_idx=0) same uid
        assert uid[0] == uid[1], "atoms within same token should share uid"
        assert uid[0] == uid[4], "tokens sharing (asym_id, res_idx) should share uid"
        assert uid[0] == uid[5], "tokens sharing (asym_id, res_idx) should share uid"
        # atoms 2-3 (token 1, res_idx=1) should have a different uid
        assert uid[2] != uid[0], "different residue should get different uid"
        assert uid[2] == uid[3], "atoms within same token should share uid"

    def test_output_shapes_padded_to_window_multiple(self):
        """Atom-level tensors are padded to a multiple of atoms_per_window_queries=32."""
        n_tokens, apa = 3, 2
        total_atoms = n_tokens * apa
        atoms_per_window = 32
        expected_padded = ((total_atoms - 1) // atoms_per_window + 1) * atoms_per_window

        data = _make_minimal_tokenized(n_tokens=n_tokens, atoms_per_token=apa)
        out = process_atom_features(data)

        assert out["ref_pos"].shape[0] == expected_padded
        assert out["ref_element"].shape[0] == expected_padded
        assert out["ref_space_uid"].shape[0] == expected_padded
        assert out["atom_to_token"].shape[0] == expected_padded

    def test_pad_mask_marks_real_vs_padding_atoms(self):
        """atom_pad_mask should be 1 for real atoms and 0 for padding."""
        n_tokens, apa = 3, 2
        total_atoms = n_tokens * apa
        data = _make_minimal_tokenized(n_tokens=n_tokens, atoms_per_token=apa)
        out = process_atom_features(data)

        pad_mask = out["atom_pad_mask"]
        assert pad_mask[:total_atoms].all(), "real atoms should have pad_mask=1"
        assert pad_mask[total_atoms:].sum() == 0, "padding atoms should have pad_mask=0"
