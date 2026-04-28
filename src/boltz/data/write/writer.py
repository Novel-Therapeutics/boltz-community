import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import (
    Coords,
    Interface,
    Record,
    Structure,
    StructureV2,
    get_affinity_output_id,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


def _serialize_summary_value(value: object) -> object:
    """Convert tensor/scalar summary values into JSON-safe Python values."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _select_prediction_value(value: object, model_idx: int) -> object:
    """Extract and serialize the value for one model from nested prediction data."""
    if isinstance(value, dict):
        return {
            key: _select_prediction_value(item, model_idx)
            for key, item in value.items()
        }
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return _serialize_summary_value(value[model_idx])
    return _serialize_summary_value(value)


def _validate_prediction_leading_dim(
    name: str, value: object, expected: int
) -> None:
    """Validate that prediction tensors align with the flattened sample axis."""
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_prediction_leading_dim(f"{name}.{key}", item, expected)
        return
    if isinstance(value, torch.Tensor):
        if value.shape[0] != expected:
            msg = (
                f"Prediction field {name} has leading dim {value.shape[0]}, "
                f"expected {expected}."
            )
            raise ValueError(msg)


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Structure predictions are flattened as [batch_size * diffusion_samples, ...]
        # in record-major order.
        coords = prediction["coords"]
        pad_masks = prediction["masks"]
        batch_size = len(records)
        if pad_masks.shape[0] != batch_size:
            msg = (
                f"Prediction masks have batch dim {pad_masks.shape[0]}, "
                f"expected {batch_size}."
            )
            raise ValueError(msg)
        total_samples = coords.shape[0]
        if total_samples % batch_size != 0:
            msg = (
                f"Predicted sample count {total_samples} is not divisible by "
                f"batch size {batch_size}."
            )
            raise ValueError(msg)
        samples_per_record = total_samples // batch_size
        # Keep the allowlist explicit: only tensors indexed along the flattened
        # sample axis should be validated here. Add new per-sample outputs to
        # this list when extending writer support.
        for key in (
            "plddt",
            "pae",
            "pde",
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
            "complex_pae",
            "complex_ipae",
            "chains_pae",
            "pair_chains_pae",
            "pair_chains_iptm",
        ):
            if key in prediction:
                _validate_prediction_leading_dim(key, prediction[key], total_samples)
        for key in ("s", "z"):
            if key in prediction and prediction[key].shape[0] != batch_size:
                msg = (
                    f"Prediction field {key} has batch dim {prediction[key].shape[0]}, "
                    f"expected {batch_size}."
                )
                raise ValueError(msg)

        # Iterate over the records
        for record_idx, (record, pad_mask) in enumerate(zip(records, pad_masks)):
            start = record_idx * samples_per_record
            end = start + samples_per_record
            coord = coords[start:end]

            # Rank the samples for this record independently.
            if "confidence_score" in prediction:
                argsort = torch.argsort(
                    prediction["confidence_score"][start:end], descending=True
                )
                idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
            else:
                idx_to_rank = {
                    sample_idx: sample_idx for sample_idx in range(samples_per_record)
                }

            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][start + model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                # Save confidence summary, plddt, pae, pde
                # Pre-compute numpy arrays to avoid redundant .cpu().numpy()
                # calls and reduce the number of separate file operations.
                rank_suffix = f"{record.id}_model_{idx_to_rank[model_idx]}"

                if "plddt" in prediction:
                    confidence_summary_dict = {
                        key: prediction[key][start + model_idx].item()
                        for key in (
                            "confidence_score",
                            "ptm",
                            "iptm",
                            "ligand_iptm",
                            "protein_iptm",
                            "complex_plddt",
                            "complex_iplddt",
                            "complex_pde",
                            "complex_ipde",
                        )
                    }
                    pair_chains = prediction["pair_chains_iptm"]
                    confidence_summary_dict["chains_ptm"] = {
                        idx: pair_chains[idx][idx][start + model_idx].item()
                        for idx in pair_chains
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: pair_chains[idx1][idx2][start + model_idx].item()
                            for idx2 in pair_chains[idx1]
                        }
                        for idx1 in pair_chains
                    }
                    if "complex_pae" in prediction:
                        confidence_summary_dict["complex_pae"] = _select_prediction_value(
                            prediction["complex_pae"], start + model_idx
                        )
                        confidence_summary_dict["complex_ipae"] = _select_prediction_value(
                            prediction["complex_ipae"], start + model_idx
                        )
                        confidence_summary_dict["chains_pae"] = _select_prediction_value(
                            prediction["chains_pae"], start + model_idx
                        )
                        confidence_summary_dict["pair_chains_pae"] = _select_prediction_value(
                            prediction["pair_chains_pae"], start + model_idx
                        )
                    path = struct_dir / f"confidence_{rank_suffix}.json"
                    with path.open("w") as f:
                        json.dump(confidence_summary_dict, f, indent=4)

                    # Save plddt
                    np.savez_compressed(
                        struct_dir / f"plddt_{rank_suffix}.npz",
                        plddt=prediction["plddt"][start + model_idx].cpu().numpy(),
                    )

                if "pae" in prediction:
                    np.savez_compressed(
                        struct_dir / f"pae_{rank_suffix}.npz",
                        pae=prediction["pae"][start + model_idx].cpu().numpy(),
                    )

                if "pde" in prediction:
                    np.savez_compressed(
                        struct_dir / f"pde_{rank_suffix}.npz",
                        pde=prediction["pde"][start + model_idx].cpu().numpy(),
                    )

            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                if batch_size == 1:
                    s = prediction["s"].cpu().numpy()
                    z = prediction["z"].cpu().numpy()
                else:
                    s = prediction["s"][record_idx].cpu().numpy()
                    z = prediction["z"][record_idx].cpu().numpy()

                path = (
                    struct_dir
                    / f"embeddings_{record.id}.npz"
                )
                np.savez_compressed(path, s=s, z=z)

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201


class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        record = batch["record"][0]
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if record.affinity and record.affinity.chain_name:
            affinity_summary["binder_chain"] = record.affinity.chain_name
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / record.id
        struct_dir.mkdir(exist_ok=True)
        output_id = get_affinity_output_id(record)
        path = struct_dir / f"affinity_{output_id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
