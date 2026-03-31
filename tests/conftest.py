"""Shared test fixtures for boltz-community tests."""

import gzip
import tempfile
from pathlib import Path

import pytest

try:
    import numpy as np
    import torch

    from boltz.data import const
    from boltz.data.types import (
        Atom,
        Bond,
        Chain,
        Connection,
        Interface,
        Residue,
        Structure,
    )
except ImportError:
    pass


@pytest.fixture
def minimal_structure():
    """Create a minimal Structure with 1 protein chain, 3 residues (ALA/GLY/VAL), ~15 atoms."""
    pytest.importorskip("torch")
    # --- atoms: 5 atoms per residue = 15 total ---
    coords = np.array(
        [
            # ALA (res 0): N, CA, C, O, CB
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [1.5, 2.4, 0.0],
            [2.0, -1.0, 0.5],
            # GLY (res 1): N, CA, C, O, HA
            [3.5, 1.4, 0.0],
            [5.0, 1.4, 0.0],
            [5.5, 2.8, 0.0],
            [5.0, 3.8, 0.0],
            [5.5, 0.5, 0.5],
            # VAL (res 2): N, CA, C, O, CB
            [7.0, 2.8, 0.0],
            [8.5, 2.8, 0.0],
            [9.0, 4.2, 0.0],
            [8.5, 5.2, 0.0],
            [9.0, 1.8, 0.5],
        ],
        dtype=np.float32,
    )

    n_atoms = len(coords)
    atoms = np.zeros(n_atoms, dtype=Atom)
    # atom names encoded as 4xi1 (shifted ASCII: uppercase char - 32)
    atom_names = [
        [46, 0, 0, 0],   # N
        [33, 33, 0, 0],   # CA
        [35, 0, 0, 0],    # C
        [47, 0, 0, 0],    # O
        [34, 34, 0, 0],   # CB
    ] * 3

    for i in range(n_atoms):
        atoms[i]["name"] = atom_names[i]
        atoms[i]["element"] = 7 if i % 5 == 0 else 6  # N or C
        atoms[i]["charge"] = 0
        atoms[i]["coords"] = coords[i]
        atoms[i]["conformer"] = coords[i]
        atoms[i]["is_present"] = True
        atoms[i]["chirality"] = 0

    # --- residues ---
    res_names = ["ALA", "GLY", "VAL"]
    res_types = [
        const.token_ids.get("ALA", 0),
        const.token_ids.get("GLY", 0),
        const.token_ids.get("VAL", 0),
    ]
    residues = np.zeros(3, dtype=Residue)
    for i in range(3):
        residues[i]["name"] = res_names[i]
        residues[i]["res_type"] = res_types[i]
        residues[i]["res_idx"] = i
        residues[i]["atom_idx"] = i * 5
        residues[i]["atom_num"] = 5
        residues[i]["atom_center"] = i * 5 + 1  # CA
        residues[i]["atom_disto"] = i * 5 + 4    # CB
        residues[i]["is_standard"] = True
        residues[i]["is_present"] = True

    # --- chain ---
    chains = np.zeros(1, dtype=Chain)
    chains[0]["name"] = "A"
    chains[0]["mol_type"] = const.chain_type_ids["PROTEIN"]
    chains[0]["entity_id"] = 0
    chains[0]["sym_id"] = 0
    chains[0]["asym_id"] = 0
    chains[0]["atom_idx"] = 0
    chains[0]["atom_num"] = n_atoms
    chains[0]["res_idx"] = 0
    chains[0]["res_num"] = 3
    chains[0]["cyclic_period"] = 0

    # --- bonds, connections, interfaces ---
    bonds = np.array([], dtype=Bond)
    connections = np.array([], dtype=Connection)
    interfaces = np.array([], dtype=Interface)
    mask = np.ones(1, dtype=bool)

    return Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
    )


@pytest.fixture
def minimal_structure_with_ligand(minimal_structure):
    """Extend minimal_structure with 1 NONPOLYMER chain (5 atoms)."""
    pytest.importorskip("torch")
    # Ligand atoms
    lig_coords = np.array(
        [
            [12.0, 0.0, 0.0],
            [13.0, 1.0, 0.0],
            [14.0, 0.0, 0.0],
            [13.0, -1.0, 0.0],
            [13.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    n_lig = len(lig_coords)
    lig_atoms = np.zeros(n_lig, dtype=Atom)
    for i in range(n_lig):
        lig_atoms[i]["name"] = [35, 0, 0, 0]  # C
        lig_atoms[i]["element"] = 6  # Carbon
        lig_atoms[i]["charge"] = 0
        lig_atoms[i]["coords"] = lig_coords[i]
        lig_atoms[i]["conformer"] = lig_coords[i]
        lig_atoms[i]["is_present"] = True
        lig_atoms[i]["chirality"] = 0

    all_atoms = np.concatenate([minimal_structure.atoms, lig_atoms])

    # Ligand residue
    lig_res = np.zeros(1, dtype=Residue)
    lig_res[0]["name"] = "LIG"
    lig_res[0]["res_type"] = 0
    lig_res[0]["res_idx"] = 3
    lig_res[0]["atom_idx"] = 15
    lig_res[0]["atom_num"] = 5
    lig_res[0]["atom_center"] = 15
    lig_res[0]["atom_disto"] = 15
    lig_res[0]["is_standard"] = False
    lig_res[0]["is_present"] = True
    all_residues = np.concatenate([minimal_structure.residues, lig_res])

    # Ligand chain
    lig_chain = np.zeros(1, dtype=Chain)
    lig_chain[0]["name"] = "B"
    lig_chain[0]["mol_type"] = const.chain_type_ids["NONPOLYMER"]
    lig_chain[0]["entity_id"] = 1
    lig_chain[0]["sym_id"] = 0
    lig_chain[0]["asym_id"] = 1
    lig_chain[0]["atom_idx"] = 15
    lig_chain[0]["atom_num"] = 5
    lig_chain[0]["res_idx"] = 3
    lig_chain[0]["res_num"] = 1
    lig_chain[0]["cyclic_period"] = 0
    all_chains = np.concatenate([minimal_structure.chains, lig_chain])

    mask = np.ones(len(all_chains), dtype=bool)

    return Structure(
        atoms=all_atoms,
        bonds=minimal_structure.bonds,
        residues=all_residues,
        chains=all_chains,
        connections=minimal_structure.connections,
        interfaces=minimal_structure.interfaces,
        mask=mask,
    )


@pytest.fixture
def synthetic_a3m_path(tmp_path):
    """Create a temporary A3M file with 5 sequences including insertions/gaps."""
    content = """\
>query
ACDEF
>seq2
ACdDEF
>seq3
A-DEF
>seq4
ACDEF
>seq5
aCDEF
"""
    p = tmp_path / "test.a3m"
    p.write_text(content)
    return p


@pytest.fixture
def synthetic_a3m_gz_path(tmp_path):
    """Create a gzipped A3M file."""
    content = b"""\
>query
ACDEF
>seq2
ACdDEF
>seq3
A-DEF
"""
    p = tmp_path / "test.a3m.gz"
    with gzip.open(str(p), "wb") as f:
        f.write(content)
    return p


@pytest.fixture
def small_feats_dict():
    """Minimal feature dict (B=1, N_tokens=8, N_atoms=16) for loss functions."""
    pytest.importorskip("torch")
    B, N_tok, N_atom = 1, 8, 16

    # token_to_rep_atom: (B, N_tok, N_atom) - picks one atom per token
    token_to_rep_atom = torch.zeros(B, N_tok, N_atom)
    for i in range(N_tok):
        token_to_rep_atom[0, i, i * 2] = 1.0

    # atom_to_token: (B, N_atom, N_tok)
    atom_to_token = torch.zeros(B, N_atom, N_tok)
    for i in range(N_atom):
        atom_to_token[0, i, i // 2] = 1.0

    # mol_type: all protein
    mol_type = torch.zeros(B, N_tok, dtype=torch.long)

    return {
        "token_to_rep_atom": token_to_rep_atom,
        "atom_to_token": atom_to_token,
        "token_pad_mask": torch.ones(B, N_tok),
        "atom_pad_mask": torch.ones(B, N_atom),
        "mol_type": mol_type,
        "asym_id": torch.zeros(B, N_tok, dtype=torch.long),
        "residue_index": torch.arange(N_tok).unsqueeze(0),
        "entity_id": torch.zeros(B, N_tok, dtype=torch.long),
        "token_disto_mask": torch.ones(B, N_tok),
        "bfactor": torch.rand(B, N_atom) * 80,
        "r_set_to_rep_atom": token_to_rep_atom.clone(),
    }


@pytest.fixture
def expected_sample_chunk_sizes():
    """Build the expected per-call chunk sizes for a sampled denoising loop."""
    def _expected_sample_chunk_sizes(
        multiplicity: int, max_parallel_samples: int, num_steps: int
    ) -> list[int]:
        per_step = [
            min(max_parallel_samples, multiplicity - start)
            for start in range(0, multiplicity, max_parallel_samples)
        ]
        return per_step * num_steps

    return _expected_sample_chunk_sizes


@pytest.fixture
def v1_atom_diffusion_factory():
    """Build a lightweight Boltz-1 diffusion module that records sampled batch sizes."""
    from boltz.model.modules.diffusion import AtomDiffusion

    def _make_v1_atom_diffusion(call_sizes: list[int]):
        class RecordingV1ScoreModel(torch.nn.Module):
            def __init__(self, token_s: int):
                super().__init__()
                self.coord_proj = torch.nn.Linear(3, 3, bias=False)
                self.token_proj = torch.nn.Linear(1, 2 * token_s, bias=False)
                torch.nn.init.zeros_(self.coord_proj.weight)
                torch.nn.init.zeros_(self.token_proj.weight)

            def forward(self, r_noisy, times, **network_condition_kwargs):
                assert network_condition_kwargs["multiplicity"] == r_noisy.shape[0]
                call_sizes.append(r_noisy.shape[0])
                num_tokens = network_condition_kwargs["feats"]["token_index"].shape[1]
                token_a = self.token_proj(times.to(r_noisy.dtype).unsqueeze(-1))
                token_a = token_a.unsqueeze(1).expand(-1, num_tokens, -1)
                return {
                    "r_update": self.coord_proj(r_noisy),
                    "token_a": token_a,
                }

        diffusion = AtomDiffusion.__new__(AtomDiffusion)
        torch.nn.Module.__init__(diffusion)
        diffusion.token_s = 2
        diffusion.score_model = RecordingV1ScoreModel(diffusion.token_s)
        diffusion.sigma_min = 0.5
        diffusion.sigma_max = 1.0
        diffusion.sigma_data = 1.0
        diffusion.rho = 1.0
        diffusion.num_sampling_steps = 2
        diffusion.gamma_0 = 0.0
        diffusion.gamma_min = 999.0
        diffusion.noise_scale = 1.0
        diffusion.step_scale = 1.0
        diffusion.use_inference_model_cache = False
        diffusion.accumulate_token_repr = False
        diffusion.alignment_reverse_diff = False
        diffusion.eval()
        return diffusion

    return _make_v1_atom_diffusion


@pytest.fixture
def v2_atom_diffusion_factory():
    """Build a lightweight Boltz-2 diffusion module that records sampled batch sizes."""
    from boltz.model.modules.diffusionv2 import AtomDiffusion

    def _make_v2_atom_diffusion(call_sizes: list[int]):
        class RecordingV2ScoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coord_proj = torch.nn.Linear(3, 3, bias=False)
                torch.nn.init.zeros_(self.coord_proj.weight)

            def forward(self, r_noisy, times, **network_condition_kwargs):
                del times
                assert network_condition_kwargs["multiplicity"] == r_noisy.shape[0]
                call_sizes.append(r_noisy.shape[0])
                return self.coord_proj(r_noisy)

        diffusion = AtomDiffusion.__new__(AtomDiffusion)
        torch.nn.Module.__init__(diffusion)
        diffusion.score_model = RecordingV2ScoreModel()
        diffusion.sigma_min = 0.5
        diffusion.sigma_max = 1.0
        diffusion.sigma_data = 1.0
        diffusion.rho = 1.0
        diffusion.num_sampling_steps = 2
        diffusion.gamma_0 = 0.0
        diffusion.gamma_min = 999.0
        diffusion.noise_scale = 1.0
        diffusion.step_scale = 1.0
        diffusion.step_scale_random = None
        diffusion.alignment_reverse_diff = False
        diffusion.eval()
        return diffusion

    return _make_v2_atom_diffusion
