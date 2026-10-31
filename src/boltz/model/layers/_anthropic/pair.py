"""Pinned Anthropic resident-pair kernels, adapted for functional fork integration.

Entry points used by fused_inference: parse_word, _stack_word, _enter,
_make_ctx, _pair_sublayers, _exit and FusedProjZ. The adapter owns admission,
sequence attention, invocation state and reporting. Upstream class-patching,
registry, graph and installation APIs have been removed. Only
BOLTZ_FUSED_INFERENCE controls the fork route; upstream lever switches do not.
See PROVENANCE.json for the source pin, original hashes and modifications.
"""
from __future__ import annotations

import math
import os
import sys
import weakref
from typing import Any, Dict, List, Optional

import torch

TAG = "boltz2-opt"
NAME = "LOCAL.boltz2.pairfuse"      # kit-local strategy id (opt_core report grammar: LOCAL.<kit>.<name>; registry LEVER_IDS)
RESIDENCIES = ("bf16", "fp32")
# The row word: "<residency>[,<site>=<impl>...]" — residency bf16|fp32, then optional provider picks per site.  A provider is a
# callable behind the interface contract v1; the ones named here are what this tree carries (an unknown pick is
# refused by name at apply).
PROVIDERS = {
    "trimul":     {"v4": "opt_core fpf_trimul_v4 (core name) trimul_v4_forward(residual=True, out=z)",
                   "site": "RETIRED (boltz2 0.3.12): the TRIMUL track's in-tree bz2_trimul kit copy is gone — refused by name at apply (trimul:site:kit_copy_retired); the core provider's rows serve its cells: `trimul=core.fast` / `core.v4` / `core.native` (opt_core.kernels.trimul)",
                   "site_tanh": "RETIRED with `site` (the tanh-form sigmoid sub-word of the same kit copy) — refused by name at apply (trimul:site_tanh:kit_copy_retired)"},
    "triatt":     {"k2b": "opt_core.attn.pair_fused tri_attn_block(impl=fpf, ln=fused, core=k2b) -- an available word (the K2B cell by name)",
                   "flash": "the same block with the core's flash_triattn cell as the attention core -- an available word",
                   "core.<word>": "the same block with the attention core served by the core triangle-attention provider (opt_core.kernels.triattn) BY TIER WORD: core.fast (the fast row) / core.big (the memory row) / core.exact -- the provider's measured cell per (cc, dtype, head_dim, heads, N) names the row on each card; a row name (core.k2b, core.triattn_native ...) is an available opt-in word",
                   "default": "the same block with the block's OWN keyed attention core (opt_core.attn.pair_fused core=\"default\", DEFAULT_CORE_TABLE: the triattn provider's `fast` tier word at and above the table's keys threshold per (cc, dtype, head_dim, heads) — flash_triattn by name on its refusal —, the flash_triattn cell below it and for unlisted classes; 0.3.23)"},
    "transition": {"fpf": "opt_core.attn.pair_fused transition(impl=fpf, ln=fused, residual in place)",
                   "core.<word>": "opt_core.kernels.transition transition(word=<word>, residual in place): a provider ROW or TIER word (core.fast = the cell's measured fast winner: v2:fast on 9.0 / v2@<launch> on 8.0 at (128,512); core.v2:fast; core.v1:lnfused ...), opt_core >= 0.5.34.0"},
    "pairbias":   {"pf": "bz_pairfuse.pair_bias (Triton, LN+128->16 fp32 FMA, one pass)"},
}
# `<site>=core.<word>` (sites trimul / triatt): the core's ONE provider for the op (opt_core.kernels.trimul / opt_core.kernels.triattn: every carried
# row + the measured cell table per compute capability) serves the site BY WORD — a row name (triatt: k2b k2 flash cuda_sm90a
# exact_headsplit cueq …; trimul: v4 tmk3_fast tmk3_exact native cueq …) or a tier word (fast | exact | big: the cell's measured winner; the fast / big rows name `trimul=core.fast` in their BOLTZ_PAIRFUSE word).
# A row that cannot serve this box / shape refuses BY NAME before the stack starts (the provider's `Refusal`: the stack's fallback word
# `<site>:<row>:<kind>`, e.g. `triatt:cuda_sm90a:no_prebuilt:…`, `trimul:tx_sm90a:cc:8.0!=9.0(…)`) — nothing is substituted. The picks named in
# PROVIDERS above are this tree's own bindings; the core words are how a measured winner enters a row (modes.py).
CORE_PREFIX = "core."
RETIRED_PICKS = {"trimul": ("site", "site_tanh")}   # picks whose kit copy left this tree: still words of the grammar (parse_word names them), refused BY NAME at apply — nothing is substituted
HANDABLE_ROWS = ("k2b", "flash", "k2", "cuda_sm90a", "triattn_native", "exact_headsplit")   # rows this driver can bind when the provider hands a pick to them by name
PICK_ENV = {"triatt": "BOLTZ_PAIRFUSE_TRIATT", "trimul": "BOLTZ_PAIRFUSE_TRIMUL"}
TIER_PICK_WORDS = ("exact", "fast", "big")             # the provider faces' tier vocabulary: a `core.<tier>` pick is a word of this grammar on every core version (a face that predates a tier word refuses it BY NAME at the probe and the pick steps aside to `fast`, named on the line: handed=)
CORE_SITES = ("trimul", "triatt", "transition")
DEFAULT_PICKS = {"trimul": "v4", "triatt": "core.fast", "transition": "fpf", "pairbias": "pf"}      # the driver's own bindings (the bare driver, the fp32 residency word): the tri-attention site by the core provider's `fast` tier word
RESIDENCY_PICKS = {"bf16": {"triatt": "core.fast"}}   # a residency word's own defaults over DEFAULT_PICKS (the bf16 word: the tri-attention site by the provider's `fast` tier word; the rows name their tier words explicitly: fast `triatt=core.fast`, big `triatt=core.big`)
PF_DEFAULT_CORE = "default"                            # == opt_core.attn.pair_fused.DEFAULT_CORE: the pick `default` hands the block this core word


def default_picks(residency=None) -> dict:
    """The picks a residency word starts from: DEFAULT_PICKS (the driver's own bindings) with the residency's tier words (RESIDENCY_PICKS) over them."""
    d = dict(DEFAULT_PICKS)
    d.update(RESIDENCY_PICKS.get(str(residency or "").strip().lower(), {}))
    return d


def core_word(pick) -> Optional[str]:
    """'core.k2' -> 'k2'; anything else -> None."""
    return pick[len(CORE_PREFIX):] if isinstance(pick, str) and pick.startswith(CORE_PREFIX) and len(pick) > len(CORE_PREFIX) else None


def core_words(site: str) -> tuple:
    """The words the core's provider face for `site` accepts (rows + tiers; the faces are standard-library-only at import)."""
    if site == "triatt":
        from opt_core.kernels import triattn as KA
        return tuple(KA.ROW_NAMES) + tuple(KA.TIER_WORDS)
    if site == "trimul":
        from opt_core.kernels import trimul as KT
        return tuple(KT.ROW_NAMES) + tuple(KT.TIER_WORDS)
    if site == "transition":                                                    # opt_core.kernels.transition: rows v2 / v1 / pf / lnl / ... + tiers
        from opt_core.kernels import transition as KX
        return tuple(KX.ROW_NAMES) + tuple(KX.TIER_WORDS)
    return ()
FP32_ROWS_ADMITTED = False     # the fp32 residency reads LN from fp32 z through pair_fused rows (v3_fp32z / v1_fp32x) that are not admitted in this release: refused by
                                # name at apply (the kit consults no core gate variable; only this source constant admits them)


def parse_word(word: str, environ=None):
    """'bf16' | 'fp32' | 'bf16,trimul=v4,triatt=k2b' -> (residency, picks) or ValueError naming the offending token.  A site's pick may also
    arrive as its own word (PICK_ENV: BOLTZ_PAIRFUSE_TRIATT=core.<row> — the mode rows' form, one lever = one word); the same grammar."""
    toks = [t.strip().lower() for t in str(word).split(",") if t.strip()]
    if not toks or toks[0] not in RESIDENCIES:
        raise ValueError(f"{NAME}: residency word {toks[0] if toks else word!r} not in {RESIDENCIES}")
    env = os.environ if environ is None else environ
    for site, key in PICK_ENV.items():                                # a pick word of its own joins the comma form (the comma form wins when both name the site)
        v = str(env.get(key, "")).strip().lower()
        if v and not any(t.startswith(site + "=") for t in toks[1:]):
            toks.append(f"{site}={v}")
    picks = default_picks(toks[0])                                   # the residency word's own defaults (bf16: triatt=core.fast); the tokens below and the pick words above override them
    for t in toks[1:]:
        if "=" not in t:
            raise ValueError(f"{NAME}: word token {t!r} is not <site>=<impl>")
        site, impl = t.split("=", 1)
        if site not in PROVIDERS:
            raise ValueError(f"{NAME}: unknown site {site!r} (sites: {sorted(PROVIDERS)})")
        if impl not in PROVIDERS[site]:
            w = core_word(impl) if site in CORE_SITES else None
            if w is None or (w.split("@")[0] not in TIER_PICK_WORDS and w.split("@")[0] not in core_words(site)):
                raise ValueError(f"{NAME}: site {site} has no provider {impl!r} in this tree (has: {sorted(PROVIDERS[site])}"
                                 + (f", core.<{'|'.join(core_words(site))}>" if site in CORE_SITES else "") + ")")
        picks[site] = impl
    return toks[0], picks
NMIN = 128                      # below this token count the stack takes the original forward
C_SERVED = (128,)

_STATE: Dict[str, Any] = {"applied": False, "residency": None, "picks": {}, "orig": {}, "t_apply": None, "proj_z_patched": [], "word": None, "handed_triatt": {}}
TRIATT_SITE_WORDS = {"cueq": "triatt_cueq", "k2b": "triatt_k2b", "flash": "triatt_flash", "default": "triatt_default", "core": "triatt_core"}   # every word _core_for returns,
SITE_WORDS = ("trimul", "trimul_core") + tuple(TRIATT_SITE_WORDS.values()) + ("transition", "pair_bias", "pair_bias_stock")                            # by the pick it serves: the one
                                                                                                                                                         # table the sites census is seeded from
STATS: Dict[str, Any] = {"stack_calls": 0, "stack_served": 0, "served_by": {}, "fallback": {}, "layers": 0, "noseq_layers": 0,
                         "sites": {w: 0 for w in SITE_WORDS},        # per site word; counted with _count (a word outside the table is tallied, never a KeyError)
                         "core_cells": {}, "handed": {}, "mask_trivial": 0, "mask_real": 0, "mask_pred_synced": 0, "mask_pred_cached": 0, "mask_pred_given": 0, "host_syncs": 0,
                         "errors": {}, "shapes": {}, "alloc": {}, "first_served": None}


def _count(d: dict, k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


# =====================================================================================================================================
# the sequence-attention pair bias: LN(z) @ W^T -> [B, H, N, N] fp32 in ONE pass over the resident z (Triton)
# =====================================================================================================================================
_PB = {"kernel": None}


def _pair_bias_kernel():
    if _PB["kernel"] is not None:
        return _PB["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _pb_kernel(z_ptr, lnw_ptr, lnb_ptr, w_ptr, out_ptr, M, stride_zr, eps,
                   C: tl.constexpr, H: tl.constexpr, BM: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < M
        cols = tl.arange(0, C)
        r64 = rows.to(tl.int64)
        x = tl.load(z_ptr + r64[:, None] * stride_zr + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)   # [BM, C] (bf16 or fp32 z, cast in registers)
        mean = tl.sum(x, axis=1) / C
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        lnw = tl.load(lnw_ptr + cols).to(tl.float32)
        lnb = tl.load(lnb_ptr + cols).to(tl.float32)
        y = xc * rstd[:, None] * lnw[None, :] + lnb[None, :]                                   # fp32 LayerNorm output (torch: biased variance, rsqrt(var+eps))
        hh = tl.arange(0, H)
        acc = tl.zeros([BM, H], dtype=tl.float32)
        for h in tl.static_range(H):                                                           # the 128->H projection as H broadcast-multiply + row reductions in fp32
            w_h = tl.load(w_ptr + h * C + cols).to(tl.float32)                                 # FMA units, no tensor core: really fp32 (no TF32 anywhere), and 2.4x faster
            col = tl.sum(y * w_h[None, :], axis=1)                                             # than tl.dot(input_precision="ieee") at this [BM,128]x[128,16] shape (pb_bench.py)
            acc = tl.where(hh[None, :] == h, col[:, None], acc)
        optr = out_ptr + hh[None, :].to(tl.int64) * M + r64[:, None]                          # out[h, row]: the 'b i j h -> b h i j' rearrange folded into the store
        tl.store(optr, acc, mask=rmask[:, None])

    _PB["kernel"] = _pb_kernel
    return _pb_kernel


def pair_bias(z: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, w: torch.Tensor, eps: float = 1e-5, BM: int = 64) -> torch.Tensor:
    """AttentionPairBias.proj_z on the resident z: Rearrange('b i j h -> b h i j')(Linear(LayerNorm(z))) -> [B, H, N, N] fp32, one pass over z.
    z [B, N, N, C] bf16|fp32 (last dim contiguous, rows uniformly strided); ln_w/ln_b [C] fp32; w [H, C] fp32 (no bias — boltz's proj_z Linear has none)."""
    import triton
    assert z.dim() == 4 and z.stride(-1) == 1
    B, N, N2, C = z.shape
    H = int(w.shape[0])
    out = torch.empty((B, H, N, N2), dtype=torch.float32, device=z.device)
    k = _pair_bias_kernel()
    M = N * N2
    w32 = w if (w.dtype == torch.float32 and w.is_contiguous()) else w.float().contiguous()
    lw = ln_w if ln_w.dtype == torch.float32 else ln_w.float()
    lb = ln_b if ln_b.dtype == torch.float32 else ln_b.float()
    for b in range(B):
        zb = z[b]
        if not (zb.stride(1) == C and zb.stride(0) == N2 * C):
            zb = zb.contiguous()
        grid = (triton.cdiv(M, BM),)
        k[grid](zb, lw, lb, w32, out[b], M, C, float(eps), C=C, H=H, BM=BM, num_warps=4)
    return out


class FusedProjZ(torch.nn.Module):
    """Stands in for AttentionPairBias.proj_z = Sequential(LayerNorm(c_z), Linear(c_z, H, bias=False), Rearrange('b ... h -> b h ...')) on ONE
    instance: the same function of z in one Triton pass (fp32 arithmetic) for a CUDA z of C in {64,128,256} and H a power of two <= 64;
    anything else (or a z that is not 4-D) takes the original Sequential on z.float() — counted."""

    def __init__(self, orig: torch.nn.Sequential):
        super().__init__()
        self.orig = orig
        ln, lin = orig[0], orig[1]
        self.eps = float(ln.eps)
        self.ok = (isinstance(ln, torch.nn.LayerNorm) and isinstance(lin, torch.nn.Linear) and lin.bias is None and ln.weight is not None
                   and int(lin.weight.shape[1]) in (64, 128, 256) and int(lin.weight.shape[0]) in (4, 8, 16, 32, 64))

    def forward(self, z):
        if self.ok and torch.is_tensor(z) and z.is_cuda and z.dim() == 4 and z.dtype in (torch.bfloat16, torch.float32) and z.shape[-1] == self.orig[1].weight.shape[1]:
            _count(STATS["sites"], "pair_bias")
            return pair_bias(z, self.orig[0].weight, self.orig[0].bias, self.orig[1].weight, self.eps)
        _count(STATS["sites"], "pair_bias_stock")
        return self.orig(z.float() if torch.is_tensor(z) and z.dtype != torch.float32 else z)


# =====================================================================================================================================
# the sites (provider calls; weights packed once per module instance and cached on it)
# =====================================================================================================================================
class SiteRefused(Exception):
    def __init__(self, site: str, reason: str):
        super().__init__(f"{site}:{reason}")
        self.site, self.reason = site, reason


def _trimul_weights(m):
    w = getattr(m, "_pairfuse_w", None)
    if w is None:
        from opt_core.kernels.fpf_trimul_v4 import generic as G     # the core's unit by its core name (this tree routes / exports nothing for it: the provider's own cell table serves)
        from .weights import trimul_weights as weights_of                     # the kit's own canonical mapping of the module's parameters (a|b row blocks of p_in/g_in)
        w = G.pack_weights(**weights_of(m), cache_owner=m, cache_key="pairfuse")
        m._pairfuse_w = w
    return w


def _trimul_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("trimul", "v4")


def trimul_site(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor) -> torch.Tensor:
    """TriangleMultiplication{Outgoing,Incoming} site, by the row's provider pick: v4 (below) | core.<word> (the core's provider, by word)."""
    pick = _trimul_pick()
    w = core_word(pick)
    if w is not None:
        return _trimul_site_core(m, z4, outgoing=outgoing, mask=mask, res_out=res_out, word=w)
    return _trimul_site_v4(m, z4, outgoing=outgoing, mask=mask, res_out=res_out)


_CORE_TRIMUL_CACHE: Dict[Any, Any] = {}      # ONE cache for the core's triangle-multiplication face in this process: packed weights keyed by the weight
                                              # tensors' identity (one entry per module), ONE workspace / cast buffer per process (the face keeps one per cache —
                                              # a cache per module would hold a workspace per module)


def _core_trimul_weights(m):
    """The ten canonical TriMul tensors of the module (the kit's own map; the core face packs and caches them per weight set in the process cache)."""
    w = getattr(m, "_pairfuse_core_w", None)
    if w is None:
        from .weights import trimul_weights as weights_of
        w = m._pairfuse_core_w = dict(weights_of(m))
    return w, _CORE_TRIMUL_CACHE


def _note_core_cell(site: str, N: int, text: str) -> None:
    """Census: which provider row / cell served the site at this token count (once per (site, N); the LEVER line carries it)."""
    key = f"{site}@{int(N)}"
    if key not in STATS["core_cells"]:
        STATS["core_cells"][key] = str(text)
        print(f"[{TAG}] {NAME} CORE-CELL site={site} n={int(N)} {text}", file=sys.stderr, flush=True)


def _trimul_site_core(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor, word: str) -> torch.Tensor:
    """Provider trimul=core.<word>: the core's triangle-multiplication face (opt_core.kernels.trimul) BY WORD on the resident z — LN_in, projections,
    gates, contraction, LN_out and the gated output projection inside the selected row, z + update returned in z's dtype and stored into res_out
    (the rows that write in place return res_out's own storage; the others' result is copied in: one pass).  A refusal with the tensors in hand is
    by name (SiteRefused -> the driver fails closed; the probe asked the face's table before the stack started)."""
    from opt_core.kernels import trimul as KT
    w, cache = _core_trimul_weights(m)
    try:
        out = KT.triangle_multiplication(z4, mask, direction="outgoing" if outgoing else "incoming", weights=w, word=word, residual=True, cache=cache)
    except KT.Refusal as e:
        raise SiteRefused("trimul", (f"{e.row}:{e.kind}" if getattr(e, "row", None) else str(e.kind)).replace(" ", "_")[:120])
    if out.data_ptr() != res_out.data_ptr() or out.dtype != res_out.dtype or tuple(out.shape) != tuple(res_out.shape):
        res_out.copy_(out.reshape(res_out.shape))
    _count(STATS["sites"], "trimul_core")
    return res_out


def _trimul_site_v4(m, z4: torch.Tensor, *, outgoing: bool, mask, res_out: torch.Tensor) -> torch.Tensor:
    """Provider trimul=v4: out = z + TriMul(z) written INTO res_out (== z4: in place; K1 reads all of z before K3 writes tile by tile, K3 reads
    each tile's gate/residual before storing it).  z4/res_out [B, N, N, C] contiguous, same dtype (bf16 | fp32).  The launch statement is
    fpf_trimul_v4.generic.trimul_packed's own (cell resolution, warm probe, safety net) with out= supplied."""
    from opt_core.kernels.fpf_trimul_v4 import generic as G, kernels as GK
    w = _trimul_weights(m)
    try:
        cfg = G._check(z4, mask, w, None, None)
        pk = (str(z4.device), int(w["C"]), int(w["D"]), bool(w.get("has_bias")), z4.dtype == torch.float32)
        if pk not in G._PROBES:                                       # the warm numerics probe, once per (device, shape class) — exactly as trimul_packed
            STATS["host_syncs"] += 1                                  # (it synchronises: first call of the class only — see the capture contract)
            if G.probe(z4.device, pk[1], pk[2], pk[3], pk[4]) is not None:
                cfg = G._check(z4, mask, w, None, None)
    except G.TrimulUnsupported as e:
        raise SiteRefused("trimul", e.reason)
    G.COUNTS["generic_calls"] += 1
    out = G.CELLS.run(lambda cc_: GK.trimul_v4_forward(z4, bool(outgoing), mask, w, cc_, eps=1e-5, residual=True, stock_round=G._STOCK_ROUND, pad=16, out=res_out),
                      cfg, z4.device)
    G.COUNTS["generic_served"] += 1
    _count(STATS["sites"], "trimul")
    return out


def _triatt_weights(m, device):
    W = getattr(m, "_opt_pairblock_W", None)                         # canonical pairblock pack, owned by the invocation
    if W is None:
        from opt_core.attn import pair_fused as PF
        from .weights import triatt_weights as weights_of
        w = weights_of(m)
        W = PF.pack_triattn_weights(ln_w=w["ln_w"], ln_b=w["ln_b"], w_q=w["w_q"], w_k=w["w_k"], w_v=w["w_v"], w_g=w["w_g"], w_b=w["w_b"],
                                   w_o=w["w_o"], b_o=w["b_o"], n_heads=w["n_heads"], head_dim=w["head_dim"], eps=w["eps"], device=device)
        m._opt_pairblock_W = W
    return W


def _triatt_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("triatt", DEFAULT_PICKS["triatt"])   # no word applied (a bare driver): the provider's `fast` tier word


def _core_for(n_tokens: int):
    """The attention core of a served block at this token count: the row's pick -- the core provider's face by TIER word (core.<word>: its measured
    cell names the row per card and N; the rows' binding), or an available word by name (k2b, flash, default = the block's own keyed core)."""
    pick = _triatt_pick()
    handed = _STATE["handed_triatt"].get(int(n_tokens))              # a core pick the provider handed to another row / word at this token count (by name, at the probe)
    if handed is not None:
        pick = handed if handed in ("k2b", "flash", "default") else CORE_PREFIX + handed
    if pick == "flash":
        return "flash_triattn", TRIATT_SITE_WORDS["flash"]
    if pick == "default":                                             # the block's own keyed core (opt_core.attn.pair_fused DEFAULT_CORE_TABLE; counted as pf_cores= on the line)
        return PF_DEFAULT_CORE, TRIATT_SITE_WORDS["default"]
    w = core_word(pick)
    if w is not None:
        from .weights import core_callable              # the kit's one binding of the core's triangle-attention face by word (shared with the block adapter)
        return core_callable(w), TRIATT_SITE_WORDS["core"]
    return "k2b", TRIATT_SITE_WORDS["k2b"]


def triatt_site(m, z4: torch.Tensor, *, ending: bool, mask, inplace: bool):
    """TriangleAttention{Starting,Ending}Node site.  inplace=True (bf16 residency): z4 <- z4 + u in the epilogue (ending: u scattered transposed),
    returns z4.  inplace=False (fp32 residency): returns the bf16 update u in z4's frame (a transposed VIEW for the ending node)."""
    from opt_core.attn import pair_fused as PF
    W = _triatt_weights(m, z4.device)
    core, word = _core_for(int(z4.shape[-2]))
    scale = 1.0 / math.sqrt(m.mha.c_hidden)
    try:
        r = PF.tri_attn_block(z4, W, mask=mask, ending=ending, residual=inplace, impl="fpf", core=core, ln="fused", scale=scale)
    except PF.Unsupported as e:
        raise SiteRefused("triatt", e.reason)
    except Exception as e:                                           # the core's provider refused with the tensors in hand: by name
        if type(e).__name__ == "CoreRefused":
            raise SiteRefused("triatt", f"core:{getattr(e, 'row', '?')}:{getattr(e, 'kind', '?')}") from None
        raise
    _count(STATS["sites"], word)
    return r


def _transition_weights(m, device):
    T = getattr(m, "_opt_transition_T", None)                        # canonical transition pack
    if T is None:
        from opt_core.attn import pair_fused as PF
        from .weights import transition_weights as weights_of
        w = weights_of(m)
        T = PF.pack_transition_weights(ln_w=w["ln_w"], ln_b=w["ln_b"], w_a=w["w_a"], w_b=w["w_b"], w_out=w["w_out"], eps=w["eps"], device=device)
        m._opt_transition_T = T
    return T


def _core_transition_weights(m, device):
    W = getattr(m, "_opt_transition_W", None)                        # canonical transition pack for core.<word> variants
    if W is None:
        from opt_core.kernels import transition as KT
        from .weights import transition_weights as weights_of
        w = weights_of(m)
        W = KT.pack(w_a=w["w_a"], w_b=w["w_b"], w_o=w["w_out"], ln_w=w["ln_w"], ln_b=w["ln_b"], eps=w["eps"], device=device)
        m._opt_transition_W = W
    return W


def transition_pick() -> str:
    return (_STATE.get("picks") or DEFAULT_PICKS).get("transition", "fpf")


def transition_site(m, z4: torch.Tensor, *, inplace: bool):
    """Transition site (pair): inplace=True -> z4 <- z4 + Transition(z4) in the kernel epilogue (row-local: in place is safe), returns z4;
    inplace=False -> the bf16 update.  Provider = the row's pick: fpf (pair_fused, LN fused) | core.<word> (opt_core.kernels.transition by word)."""
    from opt_core.attn import pair_fused as PF
    pick = transition_pick()
    if pick.startswith("core."):
        from opt_core.kernels import transition as KT
        W = _core_transition_weights(m, z4.device)
        try:
            r, sel = KT.transition(z4, W, word=pick[5:], residual=bool(inplace), out=(z4 if inplace else None), n_tokens=int(z4.shape[-2]), family="pair")
        except KT.Refusal as e:
            raise SiteRefused("transition", "core:%s:%s" % (getattr(e, "row", None) or pick[5:], str(getattr(e, "kind", e))[:60]))
        rw = sel.row + ((":" + sel.variant) if getattr(sel, "variant", None) else "") + (("@" + "".join(f"{k.lower()}{v}" for k, v in sel.cfg.items())) if isinstance(getattr(sel, "cfg", None), dict) else (("@" + str(sel.cfg)) if getattr(sel, "cfg", None) else ""))
        STATS.setdefault("transition_rows", {}); STATS["transition_rows"][rw] = STATS["transition_rows"].get(rw, 0) + 1
        _count(STATS["sites"], "transition")
        return r
    T = _transition_weights(m, z4.device)
    try:
        if inplace:
            r = PF.transition(z4, T, residual=True, ln="fused", out=z4)
        else:
            r = PF.transition(z4, T, residual=False, ln="fused")
    except PF.Unsupported as e:
        raise SiteRefused("transition", e.reason)
    _count(STATS["sites"], "transition")
    return r


# =====================================================================================================================================
# eligibility (once per (module, N, dtype, device): every site's provider is asked BEFORE the first launch; a refusal is the stack's fallback word)
# =====================================================================================================================================
def _autocast_bf16(z) -> bool:
    return bool(z.is_cuda and torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.bfloat16)


def _stack_word(first_layer, z, use_kernels: bool, training: bool) -> Optional[str]:
    """None = served; else the fallback reason (the module's original forward runs, counted)."""
    if training:
        return "training"
    if not use_kernels:
        return "kernels_off"
    if not (torch.is_tensor(z) and z.dim() == 4 and z.dtype in (torch.float32, torch.bfloat16) and _autocast_bf16(z)):
        return "autocast_off"
    C, N = int(z.shape[-1]), int(z.shape[-2])
    if C not in C_SERVED:
        return f"c:{C}"
    if N < NMIN:
        return "below_min_tokens"
    key = (N, str(z.device), _STATE["residency"])
    cache = getattr(first_layer, "_pairfuse_elig", None)
    if cache is None:
        cache = first_layer._pairfuse_elig = {}
    if key in cache:
        return cache[key] or None
    word = _probe_sites(first_layer, z)
    cache[key] = word or ""
    return word


def _probe_core_triatt(mod, zp, word: str):
    """Ask the core's triangle-attention face whether `word` serves this (card, dtype, head_dim, heads, N) — its table, no launch.  Records the
    selected row + cell for the census.  -> (ok, reason)."""
    try:
        from opt_core.kernels import triattn as KA
        cc = torch.cuda.get_device_capability(zp.device)
        N = int(zp.shape[-2]); D = int(mod.mha.c_hidden); H = int(mod.mha.no_heads)
        stack = None
        if tuple(cc) == (9, 0) and (word.split("@")[0] in ("cuda_sm90a", "triattn_native") or word in TIER_PICK_WORDS):
            from opt_core.kernels.triattn import cuda_sm90a as _C
            stack = _C.stack_key()                                   # the CUDA rows' per-ABI prebuilt is refused HERE by name when absent (no_prebuilt:…), not at load
        sel = KA.select(cc, "bf16", D, H, N, word=word, stack=stack)
    except Exception as e:  # noqa: BLE001 — the face's Refusal (by name) or an import problem
        row, kind, fb = getattr(e, "row", None), getattr(e, "kind", None), getattr(e, "fallback", None)
        reason = (f"{row}:{kind}" if kind else f"{type(e).__name__}:{e}").replace(" ", "_")[:160]
        if kind and str(kind).startswith("unknown_word") and word in TIER_PICK_WORDS and word != "fast":   # a tier word this core's face predates (core_word_absent): the pick steps aside BY NAME to the `fast` tier word
            _STATE["handed_triatt"][N] = "fast"
            _note_core_cell("triatt", N, f"handed {word}->fast (core_word_absent:{word})")
            _count(STATS["handed"], f"triatt@{N}:{word}->fast:core_word_absent")
            return _probe_core_triatt(mod, zp, "fast")
        if kind and isinstance(fb, str) and fb in HANDABLE_ROWS:      # the provider names the row to bind instead: the pick steps aside BY NAME to it for this token count
            _STATE["handed_triatt"][N] = fb                          # (never silent: `handed=` on the LEVER line, CORE-CELL line here; the gate holds — a named step-aside, not a fallback)
            _note_core_cell("triatt", N, f"handed {word}->{fb} ({reason})")
            _count(STATS["handed"], f"triatt@{N}:{word}->{fb}:{kind}".replace(" ", "_")[:200])
            if fb in ("k2b", "flash"):
                return True, ""
            return _probe_core_triatt(mod, zp, fb)                    # another face row: ask it the same question
        if kind and str(kind).startswith("no_cell"):                 # the provider's tables carry NO cell for this word at this (card, dtype, head_dim, heads) — not even an inherited
            _STATE["handed_triatt"][N] = "default"                   # one: the site steps aside BY NAME to the block's own keyed core (opt_core.attn.pair_fused core='default': its
            _note_core_cell("triatt", N, f"handed {word}->default ({reason})")   # table's row where listed, the flash_triattn cell elsewhere) for this token count; `handed=` on the LEVER
            _count(STATS["handed"], f"triatt@{N}:{word}->default:{kind}".replace(" ", "_")[:200])   # line names it; the driver keeps serving the stack; the gate holds
            return True, ""
        return False, reason
    _note_core_cell("triatt", N, KA.describe(sel))
    return True, ""


def _probe_core_trimul(mod, zp, word: str):
    """Ask the core's triangle-multiplication face whether `word` serves this (card, precision, c_z, c_hidden, N) — its table, no launch. -> (ok, reason)."""
    try:
        from opt_core.kernels import trimul as KT
        w, _cache = _core_trimul_weights(mod)
        cc = KT._device_cc(zp); prec, _cdt = KT.call_precision(zp)
        dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
        res = "fp32" if prec.startswith("f32z_") else None
        N, C, D = int(zp.shape[-2]), int(zp.shape[-1]), int(w["w_ap"].shape[0])
        sels = []
        for d in ("outgoing", "incoming"):
            sel = KT.select(cc, dt, C, D, N, d, word=word, residency=res, stack=KT.stack_word(zp), tf32=(prec == "tf32"))
            KT.admits(sel.row, cc, dt, C, D, N, residency=res, backward=False, residual=True, batch=int(zp.shape[0]),
                      abi=(KT.tx_abi_tag() if str(sel.row).startswith("tx_sm90a") else None))
            sels.append(sel)
    except Exception as e:  # noqa: BLE001
        row, kind = getattr(e, "row", None), getattr(e, "kind", None)
        return False, (f"{row}:{kind}" if kind else f"{type(e).__name__}:{e}").replace(" ", "_")[:160]
    _note_core_cell("trimul", N, " | ".join(KT.describe(s) for s in sels))
    return True, ""


def _probe_sites(layer, z) -> Optional[str]:
    """Ask every provider whether it serves this (N, C, dtype) — no launch."""
    res_dt = torch.bfloat16 if _STATE["residency"] == "bf16" else torch.float32
    N = int(z.shape[-2]); dev = z.device
    zp = torch.empty((1, N, N, int(z.shape[-1])), dtype=res_dt, device=dev)      # a probe tensor of the resident dtype (values never read)
    try:
        try:
            from opt_core.kernels.fpf_trimul_v4 import generic as G
            from opt_core.attn import pair_fused as PF
        except ImportError as e:
            return "import:" + (getattr(e, "name", None) or "import")
        if core_word(_trimul_pick()) is not None:
            ok, reason = _probe_core_trimul(layer.tri_mul_out, zp, core_word(_trimul_pick()))
        else:
            ok, reason = G.supported(zp, None, weights=_trimul_weights(layer.tri_mul_out))
        if not ok:
            return _no_cell_word("trimul", reason) or ("trimul:" + str(reason))
        if core_word(_triatt_pick()) is not None:                       # the provider's table first (by word, no launch): the row for this (card, N) or its refusal by name
            ok, reason = _probe_core_triatt(layer.tri_att_start, zp, core_word(_triatt_pick()))
            if not ok:
                return "triatt:" + str(reason)
        core, _ = _core_for(N)
        inplace = _STATE["residency"] == "bf16"
        for mod, ending in ((layer.tri_att_start, False), (layer.tri_att_end, True)):
            ok, reason = PF.supported_triattn(zp, _triatt_weights(mod, dev), None, impl="fpf", core=core, ending=ending, ln="fused", residual=inplace)
            if not ok:
                return "triatt:" + _short(reason)
        if transition_pick().startswith("core."):                      # the provider resolves the word for this (C, hidden, N, card) or names why not; a stock row = not served here, by name
            from opt_core.kernels import transition as KT
            Wc = _core_transition_weights(layer.transition_z, dev)
            try:
                sel = KT.select(transition_pick()[5:], c=int(zp.shape[-1]), hidden=Wc.hidden, n_tokens=N, dtype="bf16", family="pair", residual=inplace, mask=False,
                                device=dev, rows_count=int(zp.numel() // zp.shape[-1]))
            except KT.Refusal as e:
                nc = _no_cell_word("transition", str(getattr(e, "kind", "") or ""))
                if nc:
                    return nc
                return "transition:core:%s:%s" % (getattr(e, "row", None) or transition_pick()[5:], _short(str(getattr(e, "kind", e))))
            if sel.row in getattr(KT, "STOCK_ROWS", ("torch_swiglu", "engine_module", "compile")):   # the face names the stock op for this word on this card (no served cell): the stack takes the
                return f"{NO_CELL}:transition:stock:{sel.row}"                                            # per-layer path BY NAME (declared), never a refusal
        else:
            ok, reason = PF.supported_transition(zp, _transition_weights(layer.transition_z, dev), impl="fpf", ln="fused", residual=inplace)
            if not ok:
                return "transition:" + _short(reason)
    finally:
        del zp
    return None


NO_CELL = "no_cell"                                                   # the stack word's prefix when a site's provider carries no cell for its word on this card: `no_cell:<site>:<detail>`


def _no_cell_word(site: str, reason) -> Optional[str]:
    """A site whose provider's tables carry no cell for its tier word at this (card, shape) — `no_cell…` in the face's refusal kind: the stack
    word `no_cell:<site>:<detail>` (the WHOLE stack steps aside BY NAME to the per-layer path — the module's original forward, where the row's
    per-layer levers serve; a DECLARED reason: counted on the LEVER line as fallback_by=, the gate holds).  None for any other refusal."""
    r = str(reason or "")
    if "no_cell" not in r:
        return None
    detail = r[r.index("no_cell") + len("no_cell"):].lstrip(":").split(" ")[0][:60]
    return f"{NO_CELL}:{site}:{detail}" if detail else f"{NO_CELL}:{site}"


def _short(r: str) -> str:
    if r.startswith("no-cell:"):
        parts = r.split(":")
        return ":".join(parts[:4])
    return r.split(" ")[0][:60]


# =====================================================================================================================================
# the layer drivers
# =====================================================================================================================================
def _draw(device, columnwise: bool, B: int, N: int) -> None:
    """get_dropout_mask's torch.rand_like(z[:, :, 0:1, 0:1]) (or the column-wise z[:, 0:1, :, 0:1]) — the SAME draw (shape, fp32, order) so the
    CUDA Philox offset advances exactly as stock's; the value (mask == 1.0 at eval) is not needed."""
    shape = (B, 1, N, 1) if columnwise else (B, N, 1, 1)
    torch.rand(shape, dtype=torch.float32, device=device)


class _Ctx:
    __slots__ = ("mask_trimul", "mask_att", "inplace", "B", "N")


# ---- the all-ones predicate: once per mask tensor (or given by the caller), never per layer / per stack call
_MASK_PRED: List[Any] = []                                            # [(weakref(tensor), version, value)] — most recently used first, live entries only, at most 8
_MASK_GIVEN: Dict[str, Any] = {"value": None}


def _mask_ver(t) -> int:
    """The tensor's version counter, or -1 for an inference tensor (torch.inference_mode(): no version counter — and no in-place
    mutation by non-inference code either; boltz predicts under inference_mode and builds the mask once per forward)."""
    try:
        return -1 if t.is_inference() else int(t._version)
    except RuntimeError:                                             # 'Inference tensors do not track version counter'
        return -1


def _mask_pred_lookup(pair_mask) -> Optional[bool]:
    ver = _mask_ver(pair_mask)
    for i, (ref, v, val) in enumerate(_MASK_PRED):
        if ref() is pair_mask and v == ver:                          # the SAME live tensor object, unmodified since it was reduced (a live object's identity is unique)
            if i:
                _MASK_PRED.insert(0, _MASK_PRED.pop(i))              # most recently used first
            return val
    return None


def _mask_pred_store(pair_mask, val: bool) -> None:
    _MASK_PRED[:] = [e for e in _MASK_PRED if e[0]() is not None]    # dead tensors leave the cache (host-only bookkeeping)
    _MASK_PRED.insert(0, (weakref.ref(pair_mask), _mask_ver(pair_mask), val))
    del _MASK_PRED[8:]


def mask_is_trivial(pair_mask) -> bool:
    """True when pair_mask is None or all ones.  Given by the caller (`mask_predicate`) -> no device work; else cached per tensor object ->
    no sync; else ONE host sync (the reduction), cached for every later stack call / layer / recycling pass that passes the same tensor."""
    if pair_mask is None:
        return True
    given = _MASK_GIVEN["value"]
    if given is not None:
        STATS["mask_pred_given"] += 1
        return bool(given)
    val = _mask_pred_lookup(pair_mask)
    if val is not None:
        STATS["mask_pred_cached"] += 1
        return val
    val = bool((pair_mask == 1).all())                                # the ONE host sync
    STATS["mask_pred_synced"] += 1; STATS["host_syncs"] += 1
    _mask_pred_store(pair_mask, val)
    return val


def _make_ctx(zr: torch.Tensor, pair_mask) -> _Ctx:
    ctx = _Ctx()
    ctx.B, ctx.N = int(zr.shape[0]), int(zr.shape[1])
    ctx.inplace = _STATE["residency"] == "bf16"
    trivial = mask_is_trivial(pair_mask)                              # given | cached per tensor | ONE host sync per new mask tensor (never per layer)
    if trivial:
        STATS["mask_trivial"] += 1
        ctx.mask_trimul = None; ctx.mask_att = None
    else:
        STATS["mask_real"] += 1
        pm = pair_mask if pair_mask.dim() == 3 else pair_mask.unsqueeze(0)
        pm = pm.to(dtype=torch.float32).contiguous()
        ctx.mask_trimul = pm if ctx.B > 1 else pm[0]                # fpf_trimul_v4: [N,N] | [B,N,N] float multiplies a and b (stock: x * mask.unsqueeze(-1))
        ctx.mask_att = pm                                            # pair_fused: [B,N,N] pair mask -> its key-mask view per node
    return ctx


def _pair_sublayers(layer, zr: torch.Tensor, ctx: _Ctx) -> torch.Tensor:
    """The five pair statements of PairformerLayer / PairformerNoSeqLayer on the resident z (stock order, stock RNG draws)."""
    dev = zr.device
    B, N = ctx.B, ctx.N
    # z = z + dropout * self.tri_mul_out(z, mask=pair_mask)
    _draw(dev, False, B, N)
    trimul_site(layer.tri_mul_out, zr, outgoing=True, mask=ctx.mask_trimul, res_out=zr)
    # z = z + dropout * self.tri_mul_in(z, mask=pair_mask)
    _draw(dev, False, B, N)
    trimul_site(layer.tri_mul_in, zr, outgoing=False, mask=ctx.mask_trimul, res_out=zr)
    # z = z + dropout * self.tri_att_start(z, mask=pair_mask, ...)
    _draw(dev, False, B, N)
    if ctx.inplace:
        triatt_site(layer.tri_att_start, zr, ending=False, mask=ctx.mask_att, inplace=True)
    else:
        zr.add_(triatt_site(layer.tri_att_start, zr, ending=False, mask=ctx.mask_att, inplace=False))
    # z = z + dropout * self.tri_att_end(z, mask=pair_mask, ...)   (column-wise draw)
    _draw(dev, True, B, N)
    if ctx.inplace:
        triatt_site(layer.tri_att_end, zr, ending=True, mask=ctx.mask_att, inplace=True)
    else:
        zr.add_(triatt_site(layer.tri_att_end, zr, ending=True, mask=ctx.mask_att, inplace=False))
    # z = z + self.transition_z(z)
    if ctx.inplace:
        transition_site(layer.transition_z, zr, inplace=True)
    else:
        zr.add_(transition_site(layer.transition_z, zr, inplace=False))
    return zr


def _enter(z: torch.Tensor) -> torch.Tensor:
    """The driver-owned resident buffer for this stack call (never the caller's tensor)."""
    res_dt = torch.bfloat16 if _STATE["residency"] == "bf16" else torch.float32
    return z.to(dtype=res_dt, memory_format=torch.contiguous_format, copy=True)


def _exit(zr: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return zr if zr.dtype == like.dtype else zr.to(like.dtype)
