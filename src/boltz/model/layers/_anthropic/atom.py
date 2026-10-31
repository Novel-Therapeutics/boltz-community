"""Pinned Anthropic fused atom encoder/decoder kernels.

Functional entry points: State, new_epoch, _encoder_fused and _decoder_fused.
The fork adapter owns admission, per-sample state and cleanup. Upstream glue,
keys-gather, installation, class-patching and sampler-wrapper APIs are removed.
BOLTZ_ATOM and BOLTZ_ATOM_GEMM are not supported switches in this integration.
See PROVENANCE.json for the source pin, original hashes and modifications.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import torch

from . import atom_kernels as K

STATS: Dict[str, Any] = {"epoch": 0, "refreshes": 0, "refresh_s": [], "shape_drops": 0,
                         "calls": {"enc_fused": 0, "dec_fused": 0}}
_EPOCH = {"n": 0}


# =============================================================== per-instance state ===============================================================
class State:
    """Static buffers + packed weights of one AtomDiffusion instance."""

    def __init__(self, diff, units, gemm):
        self.units = tuple(units); self.gemm = gemm
        self.filled = None                    # (epoch, key) of the last conditioning refresh
        self.key = None                       # (B, M, NTOK, device) of the resident buffers
        self.bufs: Dict[tuple, torch.Tensor] = {}
        self.w = None                         # packed weights (built lazily on the module's device)
        self.hoist_ok = False

    def buf(self, name, shape, dtype, device):
        k = (name, tuple(shape), dtype, str(device))
        t = self.bufs.get(k)
        if t is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"boltz_atom: static buffer {name}{tuple(shape)} requested during CUDA-graph capture — the eager first denoiser step of this sample() did not build it")
            t = torch.empty(tuple(shape), dtype=dtype, device=device)
            self.bufs[k] = t
        return t

    def drop_other_shapes(self, key):
        if self.key is not None and self.key != key:
            self.bufs = {}; self.filled = None
            STATS["shape_drops"] += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.key = key

    def release(self):
        self.bufs = {}; self.filled = None; self.key = None



# ================================================================= packed weights =================================================================
def _pack_layer(layer, device):
    apb = layer.pair_bias_attn
    f = lambda t: t.detach().to(device=device, dtype=torch.float32).contiguous()  # noqa: E731
    W1 = torch.cat([apb.proj_q.weight, apb.proj_k.weight, apb.proj_v.weight, apb.proj_g.weight], 0)     # [512,128]
    lw = {"W1T": f(W1.t()), "BQ": f(apb.proj_q.bias), "WOT": f(apb.proj_o.weight.t()),
          "WSGT": f(layer.transition.swish_gate[0].weight.t()), "WABT": f(layer.transition.a_to_b.weight.t()), "WBAT": f(layer.transition.b_to_a.weight.t()),
          "eps": float(layer.adaln.a_norm.eps), "eps2": float(layer.transition.adaln.a_norm.eps)}
    lw["bf16"] = {k: v.to(torch.bfloat16) for k, v in lw.items() if torch.is_tensor(v) and k != "BQ"}
    return lw


def _weights(st: State, net, device):
    if st.w is not None and st.w["device"] == str(device):
        return st.w
    enc, dec = net.atom_attention_encoder, net.atom_attention_decoder
    f = lambda t: t.detach().to(device=device, dtype=torch.float32).contiguous()  # noqa: E731
    w = {"device": str(device),
         "enc": [_pack_layer(l, device) for l in enc.atom_encoder.diffusion_transformer.layers],
         "dec": [_pack_layer(l, device) for l in dec.atom_decoder.diffusion_transformer.layers],
         "WR": f(enc.r_to_q_trans.weight),                                   # [128,3]
         "WATT": f(enc.atom_to_token_trans[0].weight.t()),                   # [128,768]
         "LNW": f(dec.atom_feat_to_atom_pos_update[0].weight), "LNB": f(dec.atom_feat_to_atom_pos_update[0].bias),
         "eps_out": float(dec.atom_feat_to_atom_pos_update[0].eps),
         "WPOS": f(dec.atom_feat_to_atom_pos_update[1].weight),              # [3,128]
         }
    w["WATT_bf16"] = w["WATT"].to(torch.bfloat16)
    st.w = w
    return w


def _lw_for(lw, hoist, prec):
    """Kernel weight dict of one layer: packed weights (bf16 copies under the bf16 precision lever) + the hoisted conditioning tensors."""
    d = dict(lw["bf16"]) if prec == "bf16" else {k: v for k, v in lw.items() if torch.is_tensor(v)}
    d["BQ"] = lw["BQ"]
    d.update(hoist)
    return d


# ============================================================ per-sample() hoists ============================================================
def _unrepeat_c(c, B, m):
    """The conditioning c on the un-repeated basis [B, M, 128] fp32: the encoder may hand it repeated (stock c_skip: [B*m, M, 128])."""
    if c.shape[0] == B:
        return c.float()
    if c.shape[0] == B * m:
        return c[::m].float()
    raise RuntimeError(f"boltz_atom: conditioning batch {c.shape[0]} is neither B={B} nor B*m={B * m}")


def _layer_hoists(layer, c):
    """The step-invariant per-atom conditioning of one DiffusionTransformerLayer, the stock expressions on c [B,M,128] fp32:
    AdaLN sigmoid(s_scale(s_norm(c))) and s_bias(s_norm(c)) (attention and transition), the two sigmoid output gates."""
    sn = layer.adaln.s_norm(c)
    S1 = torch.sigmoid(layer.adaln.s_scale(sn)); B1 = layer.adaln.s_bias(sn)
    G1 = layer.output_projection(c)
    sn2 = layer.transition.adaln.s_norm(c)
    S2 = torch.sigmoid(layer.transition.adaln.s_scale(sn2)); B2 = layer.transition.adaln.s_bias(sn2)
    G2 = layer.transition.output_projection(c)
    return {"S1": S1, "B1": B1, "G1": G1, "S2": S2, "B2": B2, "G2": G2}


def _refresh(st: State, net, feats, c, m):
    """Once per sample() epoch (first, eager denoiser call): fill the static buffers for this (B, M, NTOK) from feats and c."""
    B, M = feats["atom_pad_mask"].shape
    NT = feats["atom_to_token"].shape[-1]
    dev = feats["atom_pad_mask"].device
    key = (int(B), int(M), int(NT), str(dev))
    ep = STATS["epoch"]
    if st.filled == (ep, key):
        return
    if torch.cuda.is_available() and dev.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("boltz_atom: conditioning refresh requested during CUDA-graph capture — the first denoiser call of this sample() was not eager")
    t0 = time.time()
    st.drop_other_shapes(key)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        a2t = feats["atom_to_token"]
        padf = feats["atom_pad_mask"].float()
        st.buf("pad", (B, M), torch.float32, dev).copy_(padf)
        # token index per atom (one-hot rows -> index; empty rows -> -1) and the one-hot check by construction
        a2tf = a2t.float()
        rs = a2tf.sum(-1)
        onehot = bool(((a2t == 0) | (a2t == 1)).all().item()) and bool(((rs == 0) | (rs == 1)).all().item())
        st.onehot = onehot
        tok = torch.where(rs > 0.5, a2tf.argmax(-1), torch.full_like(rs, -1, dtype=torch.long)).to(torch.int32)
        st.buf("tok", (B, M), torch.int32, dev).copy_(tok)
        cnt = a2tf.sum(1)                                                                     # [B, NT] atoms per token (fp32, as stock's sum(dim=1))
        st.buf("wtok", (B, NT), torch.float32, dev).copy_(1.0 / (cnt + 1e-6))
        order = torch.argsort(torch.where(tok >= 0, tok, torch.full_like(tok, NT)).long(), dim=1, stable=True).to(torch.int32)   # atoms grouped by token, stable
        st.buf("order", (B, M), torch.int32, dev).copy_(order)
        off = torch.zeros(B, NT + 1, dtype=torch.int32, device=dev)
        off[:, 1:] = torch.cumsum(cnt.round().to(torch.int32), dim=1).to(torch.int32)
        st.buf("off", (B, NT + 1), torch.int32, dev).copy_(off)
        cu = _unrepeat_c(c, B, m).contiguous()
        for part, layers in (("enc", net.atom_attention_encoder.atom_encoder.diffusion_transformer.layers),
                             ("dec", net.atom_attention_decoder.atom_decoder.diffusion_transformer.layers)):
            for i, layer in enumerate(layers):
                for name, val in _layer_hoists(layer, cu).items():
                    st.buf((part, i, name), (B, M, 128), torch.float32, dev).copy_(val)
    st.filled = (ep, key)
    STATS["refreshes"] += 1; STATS["refresh_s"].append(round(time.time() - t0, 4)); STATS["refresh_s"] = STATS["refresh_s"][-50:]


def _hoist(st, part, i):
    B, M = st.key[0], st.key[1]
    g = lambda n: st.bufs[((part, i, n), (B, M, 128), torch.float32, st.key[3])]  # noqa: E731
    return {n: g(n) for n in ("S1", "B1", "G1", "S2", "B2", "G2")}


def _b(st, name, shape, dtype):
    return st.bufs[(name, tuple(shape), dtype, st.key[3])]


# ================================================================ fused (fast tier) ================================================================
def _fused_transformer(st, part, w, x, qkv, g, ob, bias, m, R, M, Kw, prec, tail):
    """Layers 1..3 of one atom transformer after the first kernel wrote x/qkv/g: 3x (windowed attention, out+transition[+next AdaLN/proj | tail])."""
    B = st.key[0]
    pad = _b(st, "pad", (B, M), torch.float32)
    if bias.dim() != 5 or bias.shape[1] != Kw or bias.shape[2] != 32 or bias.shape[3] != 128:
        raise RuntimeError(f"boltz_atom: pair bias of shape {tuple(bias.shape)} is not [B,K,32,128,L*H]")
    bias = bias.contiguous()
    LD = bias.shape[-1]
    layers = w[part]
    for i in range(3):
        lw = _lw_for(layers[i], _hoist(st, part, i), prec)
        K.launch_win_attn(qkv=qkv, g=g, bias=bias, ld=LD, loff=4 * i, bmult=m, pad=pad, ob=ob, R=R, M=M, K=Kw, cmult=m, prec=prec)
        if i < 2:
            nlw = _lw_for(layers[i + 1], _hoist(st, part, i + 1), prec)
            K.launch_out_transition(ob=ob, x=x, lw=lw, epi=1, nlw=nlw, qkv=qkv, g=g, R=R, M=M, cmult=m, prec=prec, eps=layers[i]["eps2"])
        else:
            tail(lw)


def _encoder_fused(self, feats, q, c, atom_enc_bias, to_keys, r=None, multiplicity=1):
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m)
    B, M = st.key[0], st.key[1]; NT = st.key[2]
    dev = q.device; prec = st.gemm
    w = _weights(st, net, dev)
    R = B * m * M; Kw = M // 32
    act = torch.bfloat16 if prec == "bf16" else torch.float32
    x = torch.empty((R, 128), dtype=torch.float32, device=dev)
    qkv = torch.empty((R, 384), dtype=act, device=dev); g = torch.empty((R, 128), dtype=act, device=dev); ob = torch.empty((R, 128), dtype=act, device=dev)
    qa = torch.empty((R, 768), dtype=torch.float32, device=dev)
    a = torch.empty((B * m, NT, 768), dtype=torch.float32, device=dev)
    q0 = _unrepeat_c(q, B, m).contiguous()                       # q [B,M,128] (stock repeats it; the kernel indexes the un-repeated rows)
    pos = r.reshape(R, 3).contiguous().float()
    lw0 = _lw_for(w["enc"][0], _hoist(st, "enc", 0), prec)
    K.launch_adaln_qkvg(x=x, pro=1, q0=q0, pos=pos, wr=w["WR"], lw=lw0, qkv=qkv, g=g, R=R, M=M, cmult=m, ntok=NT, prec=prec, eps=w["enc"][0]["eps"])
    watt = w["WATT_bf16"] if prec == "bf16" else w["WATT"]

    def tail(lw):
        K.launch_out_transition(ob=ob, x=x, lw=lw, epi=2, watt=watt, qa=qa, R=R, M=M, cmult=m, prec=prec, eps=w["enc"][2]["eps2"])

    _fused_transformer(st, "enc", w, x, qkv, g, ob, atom_enc_bias, m, R, M, Kw, prec, tail)
    K.launch_segmean(qa=qa, order=_b(st, "order", (B, M), torch.int32), off=_b(st, "off", (B, NT + 1), torch.int32), wtok=_b(st, "wtok", (B, NT), torch.float32),
                     a=a, Bm=B * m, M=M, ntok=NT, mult=m)
    STATS["calls"]["enc_fused"] += 1
    q_out = x.view(B * m, M, 128)
    c_skip = c if m == 1 else c.repeat_interleave(m, 0)          # stock hands the decoder c repeated; shape-identical for m == 1
    return a, q_out, c_skip, to_keys


def _decoder_fused(self, a, q, c, atom_dec_bias, feats, to_keys, multiplicity=1):
    st = self._boltz_atom_state; net = self._boltz_atom_net
    m = int(multiplicity)
    _refresh(st, net, feats, c, m)
    B, M = st.key[0], st.key[1]; NT = st.key[2]
    dev = q.device; prec = st.gemm
    w = _weights(st, net, dev)
    R = B * m * M
    with torch.autocast("cuda", enabled=False):
        a_to_q = self.a_to_q_trans(a.float()).contiguous()       # [B*m, NT, 128] — the token-level GEMM stays cuBLAS
    act = torch.bfloat16 if prec == "bf16" else torch.float32
    x = torch.empty((R, 128), dtype=torch.float32, device=dev)
    qkv = torch.empty((R, 384), dtype=act, device=dev); g = torch.empty((R, 128), dtype=act, device=dev); ob = torch.empty((R, 128), dtype=act, device=dev)
    rup = torch.empty((R, 3), dtype=torch.float32, device=dev)
    qskip = q.reshape(R, 128).contiguous()
    lw0 = _lw_for(w["dec"][0], _hoist(st, "dec", 0), prec)
    K.launch_adaln_qkvg(x=x, pro=2, qskip=qskip, a2q=a_to_q, tok=_b(st, "tok", (B, M), torch.int32), lw=lw0, qkv=qkv, g=g, R=R, M=M, cmult=m, ntok=NT, prec=prec,
                        eps=w["dec"][0]["eps"])

    def tail(lw):
        K.launch_out_transition(ob=ob, x=x, lw=lw, epi=3, lnw=w["LNW"], lnb=w["LNB"], wpos=w["WPOS"], rup=rup, R=R, M=M, cmult=m, prec=prec,
                                eps=w["dec"][2]["eps2"], eps_out=w["eps_out"])

    _fused_transformer(st, "dec", w, x, qkv, g, ob, atom_dec_bias, m, R, M, M // 32, prec, tail)
    STATS["calls"]["dec_fused"] += 1
    return rup.view(B * m, M, 3)


def new_epoch() -> int:
    """Declare a new sample() epoch by hand (tests / direct score-model calls outside sample()): the next encoder call re-hoists."""
    _EPOCH["n"] += 1; STATS["epoch"] = _EPOCH["n"]
    return _EPOCH["n"]
