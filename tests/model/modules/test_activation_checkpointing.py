"""Tests for activation checkpointing in DiffusionTransformer and PairformerModule.

Verifies that the torch.utils.checkpoint migration produces identical outputs
to the non-checkpointed path, and that checkpointing is only active during training.
"""

import pytest
import torch


class TestDiffusionTransformerCheckpointing:
    """DiffusionTransformer must produce identical outputs with and without checkpointing."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        try:
            from boltz.model.modules.transformers import DiffusionTransformer  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import DiffusionTransformer: {e}")

    def test_checkpoint_matches_no_checkpoint(self):
        """Training forward pass is identical with activation_checkpointing on/off."""
        from boltz.model.modules.transformers import DiffusionTransformer

        dim, dim_pairwise, heads, depth = 32, 16, 4, 2
        B, N = 1, 8

        torch.manual_seed(0)
        model_ckpt = DiffusionTransformer(
            depth=depth, heads=heads, dim=dim,
            dim_pairwise=dim_pairwise, activation_checkpointing=True,
        )
        torch.manual_seed(0)
        model_no_ckpt = DiffusionTransformer(
            depth=depth, heads=heads, dim=dim,
            dim_pairwise=dim_pairwise, activation_checkpointing=False,
        )
        # Ensure identical weights
        model_no_ckpt.load_state_dict(model_ckpt.state_dict())

        a = torch.randn(B, N, dim)
        s = torch.randn(B, N, dim)
        z = torch.randn(B, N, N, dim_pairwise)
        mask = torch.ones(B, N)

        model_ckpt.train()
        model_no_ckpt.train()
        out_ckpt = model_ckpt(a, s, z, mask=mask)
        out_no_ckpt = model_no_ckpt(a, s, z, mask=mask)

        torch.testing.assert_close(out_ckpt, out_no_ckpt)

    def test_eval_ignores_checkpointing_flag(self):
        """In eval mode, activation_checkpointing=True has no effect."""
        from boltz.model.modules.transformers import DiffusionTransformer

        dim, dim_pairwise, heads, depth = 32, 16, 4, 1
        B, N = 1, 4

        model = DiffusionTransformer(
            depth=depth, heads=heads, dim=dim,
            dim_pairwise=dim_pairwise, activation_checkpointing=True,
        )
        model.eval()

        a = torch.randn(B, N, dim)
        s = torch.randn(B, N, dim)
        z = torch.randn(B, N, N, dim_pairwise)
        mask = torch.ones(B, N)

        # Should not raise — eval mode skips the checkpoint path
        out = model(a, s, z, mask=mask)
        assert out.shape == (B, N, dim)


class TestPairformerCheckpointing:
    """PairformerModule must produce identical outputs with and without checkpointing."""

    @pytest.fixture(autouse=True)
    def _check_deps(self):
        try:
            from boltz.model.modules.trunk import PairformerModule  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Cannot import PairformerModule: {e}")

    def test_checkpoint_matches_no_checkpoint(self):
        """Training forward pass is identical with activation_checkpointing on/off."""
        from boltz.model.modules.trunk import PairformerModule

        token_s, token_z, num_blocks = 32, 16, 2
        B, N = 1, 8

        torch.manual_seed(0)
        model_ckpt = PairformerModule(
            token_s=token_s, token_z=token_z, num_blocks=num_blocks,
            num_heads=4, dropout=0.0, pairwise_head_width=4,
            pairwise_num_heads=2, activation_checkpointing=True,
        )
        torch.manual_seed(0)
        model_no_ckpt = PairformerModule(
            token_s=token_s, token_z=token_z, num_blocks=num_blocks,
            num_heads=4, dropout=0.0, pairwise_head_width=4,
            pairwise_num_heads=2, activation_checkpointing=False,
        )
        model_no_ckpt.load_state_dict(model_ckpt.state_dict())

        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        model_ckpt.train()
        model_no_ckpt.train()
        s_ckpt, z_ckpt = model_ckpt(s, z, mask, pair_mask)
        s_no_ckpt, z_no_ckpt = model_no_ckpt(s, z, mask, pair_mask)

        torch.testing.assert_close(s_ckpt, s_no_ckpt)
        torch.testing.assert_close(z_ckpt, z_no_ckpt)

    def test_eval_ignores_checkpointing_flag(self):
        """In eval mode, activation_checkpointing=True has no effect."""
        from boltz.model.modules.trunk import PairformerModule

        token_s, token_z = 32, 16
        B, N = 1, 4

        model = PairformerModule(
            token_s=token_s, token_z=token_z, num_blocks=1,
            num_heads=4, dropout=0.0, pairwise_head_width=4,
            pairwise_num_heads=2, activation_checkpointing=True,
        )
        model.eval()

        s = torch.randn(B, N, token_s)
        z = torch.randn(B, N, N, token_z)
        mask = torch.ones(B, N)
        pair_mask = torch.ones(B, N, N)

        s_out, z_out = model(s, z, mask, pair_mask)
        assert s_out.shape == (B, N, token_s)
        assert z_out.shape == (B, N, N, token_z)
