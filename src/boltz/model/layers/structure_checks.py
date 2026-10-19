"""Checks of predicted stereochemistry and bond geometry.

Without --use_potentials, stereocentres are often predicted inverted or
flattened, and confidence scores do not reveal it. With it, rigid fused rings
occasionally come out with a bond pulled apart. These checks measure both on
every sample, using the same constraints the potentials enforce
(ChiralAtomPotential and PoseBustersPotential in
boltz.model.potentials.potentials) and the same dihedral convention. The
constraints exist for ligands and modified residues, not for standard residues.
"""

import torch

# A stereocentre is "flattened" when its improper dihedral is within this many
# radians (10 degrees) of planar. On ideal RDKit geometry the dihedral of real
# stereocentres ranges from about 29 to 40 degrees, so the steering potential's
# own 30-degree buffer is a target, not a correctness boundary: it would flag
# ideal centres in glucose and steroids as flattened.
FLAT_DIHEDRAL = 0.17453

# A bond is "out of range" when its length is more than this fraction outside
# its RDKit bounds. This catches bonds pulled apart (1.98-3.19 A for C-C in
# failed predictions, 30-110% too long) while ignoring ordinary variation, such
# as the two interchangeable oxygens of a carboxylic acid swapping lengths,
# which the steering potential's 12.5% buffer would flag.
BOND_TOLERANCE = 0.25

# Per-sample counts returned by check_structures and written to confidence JSON.
CHECK_KEYS = (
    "chiral_centers_checked",
    "chiral_centers_inverted",
    "chiral_centers_flattened",
    "bonds_checked",
    "bonds_out_of_range",
)


def dihedral(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Signed dihedral of four atoms, as DihedralPotential.compute_variable."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])
    n_ijk = torch.cross(r_ij, r_kj, dim=-1)
    n_jkl = torch.cross(r_kj, r_kl, dim=-1)
    sign = torch.sign((r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(-1))
    cos = (n_ijk * n_jkl).sum(-1) / (
        torch.linalg.norm(n_ijk, dim=-1) * torch.linalg.norm(n_jkl, dim=-1)
    )
    return sign * torch.arccos(torch.clamp(cos, -1 + 1e-8, 1 - 1e-8))


def check_structures(
    coords: torch.Tensor, feats: dict[str, torch.Tensor], multiplicity: int
) -> dict[str, torch.Tensor]:
    """Count stereochemistry and bond-length failures in each sample.

    Parameters
    ----------
    coords : torch.Tensor
        Predicted atom coordinates, shape (B * multiplicity, N_atoms, 3).
    feats : dict[str, torch.Tensor]
        Input features, batched with leading dimension B.
    multiplicity : int
        Samples per input.

    Returns
    -------
    dict[str, torch.Tensor]
        Integer counts per sample, shape (B * multiplicity,):
        chiral_centers_checked, chiral_centers_inverted, chiral_centers_flattened,
        bonds_checked, bonds_out_of_range.

    """
    n_samples = coords.shape[0]
    device = coords.device
    out = {
        key: torch.zeros(n_samples, dtype=torch.long, device=device)
        for key in CHECK_KEYS
    }
    coords = coords.float()

    for sample in range(n_samples):
        b = sample // multiplicity
        x = coords[sample]

        # One reference constraint per stereocentre; padding is never a reference.
        if "chiral_atom_index" in feats:
            ref = feats["chiral_reference_mask"][b].bool()
            if ref.any():
                index = feats["chiral_atom_index"][b][:, ref]
                is_r = feats["chiral_atom_orientations"][b][ref].bool()
                phi = dihedral(x, index)
                wanted = torch.where(is_r, phi, -phi)
                out["chiral_centers_checked"][sample] = ref.sum()
                out["chiral_centers_inverted"][sample] = (wanted <= -FLAT_DIHEDRAL).sum()
                out["chiral_centers_flattened"][sample] = (
                    wanted.abs() < FLAT_DIHEDRAL
                ).sum()

        # Bonds only (not angles); padding is never a bond.
        if "rdkit_bounds_index" in feats:
            bond = (
                feats["rdkit_bounds_bond_mask"][b].bool()
                & ~feats["rdkit_bounds_angle_mask"][b].bool()
            )
            if bond.any():
                index = feats["rdkit_bounds_index"][b][:, bond]
                length = torch.linalg.norm(x[index[0]] - x[index[1]], dim=-1)
                lower = feats["rdkit_lower_bounds"][b][bond] * (1 - BOND_TOLERANCE)
                upper = feats["rdkit_upper_bounds"][b][bond] * (1 + BOND_TOLERANCE)
                out["bonds_checked"][sample] = bond.sum()
                out["bonds_out_of_range"][sample] = (
                    (length < lower) | (length > upper)
                ).sum()

    return out


def passes_checks(checks: dict[str, torch.Tensor]) -> torch.Tensor:
    """True for samples with no inverted or flattened centre and no bad bond."""
    return (
        (checks["chiral_centers_inverted"] == 0)
        & (checks["chiral_centers_flattened"] == 0)
        & (checks["bonds_out_of_range"] == 0)
    )


def select_affinity_sample(iptm: torch.Tensor, passed: torch.Tensor) -> int:
    """Index of the highest-ipTM sample among those that pass the checks.

    When every sample passes, or none does, this is the highest-ipTM sample
    overall, chosen exactly as before the checks existed.
    """
    if passed.any() and not passed.all():
        iptm = torch.where(passed, iptm, torch.full_like(iptm, float("-inf")))
    return int(torch.argsort(iptm, descending=True)[0].item())
