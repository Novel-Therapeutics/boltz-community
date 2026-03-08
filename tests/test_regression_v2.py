"""Boltz-2 regression tests against golden tensors.

Similar to test_regression.py for Boltz-1, this captures intermediate outputs
from key Boltz-2 model components and compares them against saved golden tensors.

Golden tensors are generated from a real inference feature dict (captured via a
small end-to-end run), ensuring all feature keys are present and realistic.

Usage:
    # First run: generate golden tensors (requires GPU or CPU, downloads model)
    SAVE_GOLDEN=1 pytest tests/test_regression_v2.py::test_save_golden -s

    # Subsequent runs: compare against saved golden tensors
    pytest tests/test_regression_v2.py -m regression
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import unittest

from lightning_fabric import seed_everything

from boltz.main import BOLTZ2_URL_WITH_FALLBACK
from boltz.model.models.boltz2 import Boltz2

import test_utils

tests_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(tests_dir, "data")

GOLDEN_PATH = os.path.join(test_data_dir, "boltz2_regression_feats.pkl")

SEED = 42

# Small input for feature capture
CAPTURE_YAML = """\
version: 1
sequences:
  - protein:
      id: A
      sequence: ACDEFGHIKL
      msa: empty
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_model(device: torch.device) -> Boltz2:
    """Download (if needed) and load the Boltz-2 checkpoint."""
    cache = os.path.expanduser("~/.boltz")

    checkpoint_url = BOLTZ2_URL_WITH_FALLBACK[0]
    model_name = checkpoint_url.split("/")[-1]
    checkpoint = os.path.join(cache, model_name)
    if not os.path.exists(checkpoint):
        for url in BOLTZ2_URL_WITH_FALLBACK:
            try:
                test_utils.download_file(url, checkpoint)
                break
            except Exception:
                if url == BOLTZ2_URL_WITH_FALLBACK[-1]:
                    raise

    model: nn.Module = Boltz2.load_from_checkpoint(
        checkpoint, map_location=device, weights_only=False,
    )
    model.to(device)
    model.eval()
    return model


def _capture_feats_from_preprocessing(device: torch.device) -> dict:
    """Run preprocessing only (no GPU needed) to get real feature tensors.

    Uses the data pipeline so all keys are present and realistic.
    """
    from pathlib import Path

    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.main import process_inputs

    cache = Path(os.path.expanduser("~/.boltz"))
    mol_dir = cache / "mols"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write the input YAML
        input_path = tmpdir / "test.yaml"
        input_path.write_text(CAPTURE_YAML)

        out_dir = tmpdir / "output"

        # Run preprocessing only (no Trainer, no GPU)
        process_inputs(
            data=[input_path],
            out_dir=out_dir,
            ccd_path=cache / "ccd.pkl",
            mol_dir=mol_dir,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            max_msa_seqs=8192,
            use_msa_server=False,
            boltz2=True,
        )

        # Load the processed data to get features
        processed_dir = out_dir / "processed"
        manifest = Manifest.load(processed_dir / "manifest.json")

        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            mol_dir=mol_dir,
            num_workers=0,
        )
        data_module.setup(stage="predict")
        dataset = data_module.predict_dataloader().dataset

        # Get the first (only) example
        sample = dataset[0]

        # Move to device and add batch dim
        feats = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                feats[key] = val.unsqueeze(0).to(device)

    return feats


def _compute_trunk_init(model, feats):
    """Compute s_init and z_init (trunk initialization), mirroring Boltz2.forward()."""
    with torch.no_grad():
        s_inputs = model.input_embedder(feats)
        rel_pos_enc = model.rel_pos(feats)

        s_init = model.s_init(s_inputs)
        z_init = (
            model.z_init_1(s_inputs)[:, :, None]
            + model.z_init_2(s_inputs)[:, None, :]
        )
        z_init = z_init + rel_pos_enc
        z_init = z_init + model.token_bonds(feats["token_bonds"].float())
        if model.hparams.get("bond_type_feature", False):
            z_init = z_init + model.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + model.contact_conditioning(feats)

    return s_inputs, rel_pos_enc, s_init, z_init


# ── Golden tensor generation ────────────────────────────────────────────────

def test_save_golden():
    """Generate and save golden tensors from a real inference feature dict.

    Only runs when SAVE_GOLDEN=1 environment variable is set.

    Usage:
        SAVE_GOLDEN=1 pytest tests/test_regression_v2.py::test_save_golden -s
    """
    if not os.environ.get("SAVE_GOLDEN"):
        pytest.skip(
            "Set SAVE_GOLDEN=1 to generate golden tensors. "
            "Example: SAVE_GOLDEN=1 pytest tests/test_regression_v2.py::test_save_golden -s"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nCapturing features from real inference run...")
    feats = _capture_feats_from_preprocessing(device)

    print("Loading model...")
    model = _load_model(device)

    print("Computing golden outputs...")
    seed_everything(SEED)
    s_inputs, rel_pos_enc, s_init, z_init = _compute_trunk_init(model, feats)

    golden = {
        "seed": SEED,
        "feats": {k: v.cpu() for k, v in feats.items() if isinstance(v, torch.Tensor)},
        "s_inputs": s_inputs.cpu(),
        "relative_position_encoding": rel_pos_enc.cpu(),
        "s_init": s_init.cpu(),
        "z_init": z_init.cpu(),
    }

    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    torch.save(golden, GOLDEN_PATH)
    print(f"\nGolden tensors saved to {GOLDEN_PATH}")
    print(f"  s_inputs shape:  {s_inputs.shape}")
    print(f"  rel_pos shape:   {rel_pos_enc.shape}")
    print(f"  s_init shape:    {s_init.shape}")
    print(f"  z_init shape:    {z_init.shape}")


# ── Regression tests ─────────────────────────────────────────────────────────

def _golden_available():
    return os.path.exists(GOLDEN_PATH)


SKIP_MSG = (
    f"Golden tensors not found at {GOLDEN_PATH}. "
    "Run: SAVE_GOLDEN=1 pytest tests/test_regression_v2.py::test_save_golden -s"
)


@pytest.mark.regression
class RegressionTesterV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _golden_available():
            raise unittest.SkipTest(SKIP_MSG)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model(device)

        golden = torch.load(GOLDEN_PATH, map_location=device, weights_only=False)

        # Move golden feats to device
        feats = {}
        for key, val in golden["feats"].items():
            feats[key] = val.to(device) if isinstance(val, torch.Tensor) else val

        cls.model = model
        cls.golden = golden
        cls.feats = feats
        cls.device = device

    def test_input_embedder(self):
        """Input embedder output must match golden tensor."""
        exp = self.golden["s_inputs"].to(self.device)
        seed_everything(self.golden["seed"])
        act = self.model.input_embedder(self.feats)

        self.assertTrue(
            torch.allclose(exp, act, atol=1e-5),
            f"input_embedder mismatch: max diff = {(exp - act).abs().max().item():.2e}",
        )

    def test_rel_pos(self):
        """Relative position encoding must match golden tensor."""
        exp = self.golden["relative_position_encoding"].to(self.device)
        act = self.model.rel_pos(self.feats)

        self.assertTrue(
            torch.allclose(exp, act, atol=1e-5),
            f"rel_pos mismatch: max diff = {(exp - act).abs().max().item():.2e}",
        )

    def test_trunk_init(self):
        """Trunk initialization (s_init, z_init) must match golden tensors."""
        seed_everything(self.golden["seed"])

        _, _, s_init, z_init = _compute_trunk_init(self.model, self.feats)

        exp_s = self.golden["s_init"].to(self.device)
        exp_z = self.golden["z_init"].to(self.device)

        self.assertTrue(
            torch.allclose(exp_s, s_init, atol=1e-5),
            f"s_init mismatch: max diff = {(exp_s - s_init).abs().max().item():.2e}",
        )
        self.assertTrue(
            torch.allclose(exp_z, z_init, atol=1e-5),
            f"z_init mismatch: max diff = {(exp_z - z_init).abs().max().item():.2e}",
        )


if __name__ == "__main__":
    unittest.main()
