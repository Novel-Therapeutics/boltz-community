"""Canonical mappings from the pinned Apache-2.0 kit; see NOTICE."""

import functools

import torch


def trimul_weights(m):
    """boltz TriangleMultiplication{Outgoing,Incoming} -> the core's ten canonical TriMul tensors (opt_core.trimul.WEIGHT_KEYS)."""
    d = m.p_in.weight.shape[0] // 2
    return dict(
        ln_in_w=m.norm_in.weight,
        ln_in_b=m.norm_in.bias,
        w_ap=m.p_in.weight[:d],
        w_ag=m.g_in.weight[:d],
        w_bp=m.p_in.weight[d:],
        w_bg=m.g_in.weight[d:],
        ln_out_w=m.norm_out.weight,
        ln_out_b=m.norm_out.bias,
        w_o=m.p_out.weight,
        w_og=m.g_out.weight,
    )


def transition_weights(m):
    """boltz Transition -> the core's canonical transition tensors (opt_core.attn.pair_fused.TRANSITION_KEYS)."""
    for lin in (
        m.fc1,
        m.fc2,
        m.fc3,
    ):  # every parameter of the module the pack is not given must not exist (never a silently dropped bias)
        assert lin.bias is None, (
            f"{type(m).__name__}: a projection bias the fused transition is not given ({lin})"
        )
    return dict(
        ln_w=m.norm.weight,
        ln_b=m.norm.bias,
        w_a=m.fc1.weight,
        w_b=m.fc2.weight,
        w_out=m.fc3.weight,
        eps=getattr(m.norm, "eps", 1e-5),
    )


def triatt_weights(m):
    """boltz TriangleAttention -> the core's canonical triangle-attention tensors (opt_core.attn.pair_fused.TRIATTN_KEYS)."""
    a = m.mha
    for lin in (
        m.linear,
        a.linear_q,
        a.linear_k,
        a.linear_v,
        a.linear_g,
    ):  # every parameter of the module the pack is not given must not exist (never a silently dropped bias)
        assert lin.bias is None, (
            f"{type(m).__name__}: a projection bias the fused block is not given ({lin})"
        )
    return dict(
        ln_w=m.layer_norm.weight,
        ln_b=m.layer_norm.bias,
        w_q=a.linear_q.weight,
        w_k=a.linear_k.weight,
        w_v=a.linear_v.weight,
        w_g=a.linear_g.weight,
        w_b=m.linear.weight,
        w_o=a.linear_o.weight,
        b_o=a.linear_o.bias,
        n_heads=a.no_heads,
        head_dim=a.c_hidden,
        eps=getattr(m.layer_norm, "eps", 1e-5),
    )


@functools.cache
def core_callable(word):
    from opt_core.kernels import triattn

    def call(q, k, v, bias, mask, scale):
        def stock(q, k, v, bias, mask=None, scale=None):
            from cuequivariance_torch.primitives.triangle import triangle_attention

            return triangle_attention(q, k, v, bias, mask=mask, scale=scale)

        return triattn.triangle_attention(
            q,
            k,
            v,
            bias,
            mask,
            scale,
            word=word,
            stock=stock,
            form="bias_only" if mask is None else "mask_bias",
        )

    return call
