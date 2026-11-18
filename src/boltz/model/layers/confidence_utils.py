import torch
from torch import nn

from boltz.data import const


def compute_collinear_mask(v1, v2):
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    with torch.amp.autocast(pred_atom_coords.device.type, enabled=False):
        asym_id_token = feats["asym_id"]
        asym_id_atom = torch.bmm(
            feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
        ).squeeze(-1)

    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through the batch and modify the frames for nonpolymers
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            if (
                feats["mol_type"][i, token_idx] != const.chain_type_ids["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5
            if inference:
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            try:
                frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            except Exception as e:
                print(f"Failed to process {feats['pdb_id']} due to {e}")
            token_idx += num_tokens
            atom_idx += num_atoms

    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Compute masks for collinearity / overlap
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]


def compute_aggregated_metric(logits, end=1.0):
    # Compute aggregated metric from logits
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


def _weighted_mean_or_nan(values, weights, dims):
    """Return a weighted mean, using NaN when the weight sum is zero."""
    weight_sum = weights.sum(dim=dims)
    weighted_sum = (values * weights).sum(dim=dims)
    mean = weighted_sum / torch.clamp(weight_sum, min=1e-10)
    return torch.where(weight_sum > 0, mean, torch.full_like(mean, torch.nan))


def _compute_contact_weight_masks(feats, pred_distogram_logits, multiplicity):
    """Build the contact-weighted masks used for PDE- and PAE-style summaries."""
    pred_distogram_prob = nn.functional.softmax(
        pred_distogram_logits, dim=-1
    ).repeat_interleave(multiplicity, 0)
    contacts = torch.zeros(
        (1, 1, 1, pred_distogram_prob.shape[-1]),
        dtype=pred_distogram_prob.dtype,
        device=pred_distogram_prob.device,
    )
    contacts[:, :, :, : min(20, pred_distogram_prob.shape[-1])] = 1.0
    prob_contact = (pred_distogram_prob * contacts).sum(-1)

    token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
    token_pad_pair_mask = (
        token_pad_mask.unsqueeze(-1)
        * token_pad_mask.unsqueeze(-2)
        * (
            1
            - torch.eye(
                token_pad_mask.shape[1], device=token_pad_mask.device
            ).unsqueeze(0)
        )
    )
    token_pair_mask = token_pad_pair_mask * prob_contact
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    token_interface_pair_mask = token_pair_mask * (
        asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
    )
    return token_pair_mask, token_interface_pair_mask, asym_id


def compute_pae_summaries(pae, pred_distogram_logits, feats, multiplicity):
    """Compute contact-weighted aggregate PAE summaries."""
    token_pair_mask, token_interface_pair_mask, asym_id = _compute_contact_weight_masks(
        feats, pred_distogram_logits, multiplicity
    )

    complex_pae = _weighted_mean_or_nan(pae, token_pair_mask, dims=(1, 2))
    complex_ipae = _weighted_mean_or_nan(
        pae, token_interface_pair_mask, dims=(1, 2)
    )

    valid_asym_id = asym_id[
        feats["token_pad_mask"].repeat_interleave(multiplicity, 0).bool()
    ]
    asym_ids_list = torch.unique(valid_asym_id).tolist()

    pair_chains_pae = {}
    for idx1 in asym_ids_list:
        chain_pae = {}
        idx1_mask = (asym_id.unsqueeze(-1) == idx1).to(token_pair_mask.dtype)
        for idx2 in asym_ids_list:
            idx2_mask = (asym_id.unsqueeze(-2) == idx2).to(token_pair_mask.dtype)
            chain_pair_mask = token_pair_mask * idx1_mask * idx2_mask
            chain_pae[idx2] = _weighted_mean_or_nan(pae, chain_pair_mask, dims=(1, 2))
        pair_chains_pae[idx1] = chain_pae

    chains_pae = {idx: pair_chains_pae[idx][idx] for idx in asym_ids_list}
    return complex_pae, complex_ipae, chains_pae, pair_chains_pae


def compute_unweighted_pae_summaries(pae, feats, multiplicity, frame_mask=None):
    """Compute unweighted aggregate PAE summaries.

    Plain means over valid token pairs, where ``pae[b, i, j]`` is the expected
    error of token ``j`` after aligning on the frame of token ``i``. Padding and
    the diagonal are excluded, and tokens without a valid frame are excluded on
    the aligned axis, as for pTM / ipTM. This is the averaging used by the
    interface-PAE filters common in binder design (e.g. ``pae_interaction``),
    and matches the definition in upstream Boltz PR #707.

    ``compute_pae_summaries`` instead weights each pair by its predicted contact
    probability, as Boltz does for ``complex_pde`` / ``complex_ipde``.

    Parameters
    ----------
    pae : torch.Tensor
        The aggregated PAE values, shape (B * multiplicity, N, N), in Angstrom.
    feats : Dict[str, torch.Tensor]
        The input features.
    multiplicity : int
        The batch size of the diffusion roll-out.
    frame_mask : torch.Tensor, optional
        Mask of tokens with a valid (non-collinear) frame, shape
        (B * multiplicity, N). If None, only padding is masked.

    Returns
    -------
    Tensor
        complex_pae: mean PAE over all valid off-diagonal pairs, shape (B,).
    Tensor
        complex_ipae: mean PAE over inter-chain pairs, shape (B,); NaN for a
        single chain.
    Dict[int, Tensor]
        chains_pae: mean intra-chain PAE per chain.
    Dict[int, Dict[int, Tensor]]
        pair_chains_pae[c1][c2]: mean PAE of chain ``c2`` tokens aligned on the
        frames of chain ``c1``; NaN when no valid pair exists.

    """
    token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    if frame_mask is None:
        frame_mask = token_pad_mask
    frame_mask = frame_mask.reshape(-1, frame_mask.shape[-1]).float()

    not_diag = 1 - torch.eye(
        token_pad_mask.shape[1], device=pae.device, dtype=pae.dtype
    ).unsqueeze(0)
    pair_mask = (
        (frame_mask * token_pad_mask).unsqueeze(-1)
        * token_pad_mask.unsqueeze(-2)
        * not_diag
    )
    interface_pair_mask = pair_mask * (asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2))

    complex_pae = _weighted_mean_or_nan(pae, pair_mask, dims=(1, 2))
    complex_ipae = _weighted_mean_or_nan(pae, interface_pair_mask, dims=(1, 2))

    valid_asym_id = asym_id[token_pad_mask.bool()]
    asym_ids_list = torch.unique(valid_asym_id).tolist()

    pair_chains_pae = {}
    for idx1 in asym_ids_list:
        idx1_mask = (asym_id.unsqueeze(-1) == idx1).to(pair_mask.dtype)
        pair_chains_pae[idx1] = {
            idx2: _weighted_mean_or_nan(
                pae,
                pair_mask * idx1_mask * (asym_id.unsqueeze(-2) == idx2).to(pair_mask.dtype),
                dims=(1, 2),
            )
            for idx2 in asym_ids_list
        }

    chains_pae = {idx: pair_chains_pae[idx][idx] for idx in asym_ids_list}
    return complex_pae, complex_ipae, chains_pae, pair_chains_pae


def tm_function(d, Nres):
    d0 = 1.24 * (torch.clip(Nres, min=19) - 15) ** (1 / 3) - 1.8
    return 1 / (1 + (d / d0) ** 2)


def compute_ptms(logits, x_preds, feats, multiplicity, frame_mask=None):
    # It needs to take as input the mask of the frames as they are not used to compute the PTM
    if frame_mask is None:
        _, frame_mask = compute_frame_pred(
            x_preds, feats["frames_idx"], feats, multiplicity, inference=True
        )
    mask_collinear_pred = frame_mask
    # mask overlapping, collinear tokens and ions (invalid frames)
    mask_pad = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
    maski = mask_collinear_pred.reshape(-1, mask_collinear_pred.shape[-1])
    pair_mask_ptm = maski[:, :, None] * mask_pad[:, None, :] * mask_pad[:, :, None]
    asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    pair_mask_iptm = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
    )
    num_bins = logits.shape[-1]
    bin_width = 32.0 / num_bins
    end = 32.0
    pae_value = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    ).unsqueeze(0)
    N_res = mask_pad.sum(dim=-1, keepdim=True)
    tm_value = tm_function(pae_value, N_res).unsqueeze(1).unsqueeze(2)
    probs = nn.functional.softmax(logits, dim=-1)
    tm_expected_value = torch.sum(
        probs * tm_value,
        dim=-1,
    )  # shape (B, N, N)
    ptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_ptm, dim=-1)
        / (torch.sum(pair_mask_ptm, dim=-1) + 1e-5),
        dim=1,
    ).values
    iptm = torch.max(
        torch.sum(tm_expected_value * pair_mask_iptm, dim=-1)
        / (torch.sum(pair_mask_iptm, dim=-1) + 1e-5),
        dim=1,
    ).values

    # compute ligand and protein iPTM
    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    is_protein_token = (token_type == const.chain_type_ids["PROTEIN"]).float()

    ligand_iptm_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (
            (is_ligand_token[:, :, None] * is_protein_token[:, None, :])
            + (is_protein_token[:, :, None] * is_ligand_token[:, None, :])
        )
    )
    protein_ipmt_mask = (
        maski[:, :, None]
        * (asym_id[:, None, :] != asym_id[:, :, None])
        * mask_pad[:, None, :]
        * mask_pad[:, :, None]
        * (is_protein_token[:, :, None] * is_protein_token[:, None, :])
    )

    ligand_iptm = torch.max(
        torch.sum(tm_expected_value * ligand_iptm_mask, dim=-1)
        / (torch.sum(ligand_iptm_mask, dim=-1) + 1e-5),
        dim=1,
    ).values
    protein_iptm = torch.max(
        torch.sum(tm_expected_value * protein_ipmt_mask, dim=-1)
        / (torch.sum(protein_ipmt_mask, dim=-1) + 1e-5),
        dim=1,
    ).values

    # Compute pair chain ipTM
    chain_pair_iptm = {}
    asym_ids_list = torch.unique(asym_id).tolist()
    for idx1 in asym_ids_list:
        chain_iptm = {}
        for idx2 in asym_ids_list:
            mask_pair_chain = (
                maski[:, :, None]
                * (asym_id[:, None, :] == idx1)
                * (asym_id[:, :, None] == idx2)
                * mask_pad[:, None, :]
                * mask_pad[:, :, None]
            )

            chain_iptm[idx2] = torch.max(
                torch.sum(tm_expected_value * mask_pair_chain, dim=-1)
                / (torch.sum(mask_pair_chain, dim=-1) + 1e-5),
                dim=1,
            ).values
        chain_pair_iptm[idx1] = chain_iptm

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm
