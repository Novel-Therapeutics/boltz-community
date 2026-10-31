# Experimental fused inference

`BOLTZ_FUSED_INFERENCE` defaults to `off`. The prototype accepts `dit` (fused
bf16 token transformer), `atom` (fused bf16 atom encoder/decoder), `denoiser`
(both), `pair` (resident bf16 pair track with fused projections and residuals),
and `all` (all ported fusions). `on/true/1` mean `all`; `false/0` mean `off`.
These are reduced-precision, changed-rounding paths. They do not promise identical
coordinates or scores, or improved scientific accuracy.

The implementation carries the actual pinned Anthropic Triton kernels, rather
than replacing their dataflow with several eager PyTorch operations. Original
source hashes, modifications and Apache-2.0 notices are in
`src/boltz/model/layers/_anthropic/`. The pinned shared `opt_core` from commit
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e` is required for the pair-track route.
The validated-runtime target is CUDA bf16 inference on Blackwell with PyTorch
2.12 or later. The base package still supports its existing minimum PyTorch;
unsupported contexts retain the original implementation.

Admission excludes CPU, training, enabled gradients, compiled/custom forward
implementations, hooks, unsupported geometry and unversioned model weights.
The token kernel serves width 768 / 16 heads of width 48; the atom kernels serve
three encoder plus three decoder layers of width 128, four heads of width 32,
and 32-query/128-key windows. Atom assignment must be one-hot-or-empty. The
independent conditioning cache must be off for fused denoising. Packing is
admitted against free GPU memory; errors after admission propagate, without a
silent scientific retry. Model weights and conditioning must stay fixed during
one inference invocation; packs are rebuilt for the next invocation.

The fork's sampler, corrected chunking, guidance, particle resampling, random
draws and alignment remain in place. Pairformer sequence attention preserves
the fork's SDPA choice; the opt-in fused denoiser uses the upstream Triton
attention kernels. Functional pair bias routing avoids replacing registered model modules. Context and packed
values are released in `finally`; provider-native payload caches are cleared
when the pair scope closes. No global class-forward patches are installed.

The graph prototype from PR #36 was also tested with the fusions. It preserved
predictions but did not improve the selected workloads, so this branch carries
the eager fusions only. PR #36 remains separate; no graph code or graph switch
is added here. The upstream whole-sampler rollout is not ported: its fast row
steps aside under potentials, whereas the fork's existing guidance loop is
preserved here.

## Evaluation protocol

Measured on 30 October 2026 (Eastern), RTX PRO 6000 Blackwell Workstation 96 GB
(cc 12.0), driver 580.65.06, PyTorch 2.12.0+cu130, Triton 3.7.0 and
cuEquivariance 0.10.0. The upstream kit and shared core are pinned to
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`; controls use stock Boltz 2.2.1 and
the fork at `1431d57` (2.13.0 plus a documentation change). The fork control uses
FlashPairformer `fast`, `--flash_attn` and conditioning caching off.

Existing checkpoints, exact assembly/sequence inputs, MSA, template and scorer
hashes were checked before reuse. Controls and variants share frozen featurized
batches and initial CPU/CUDA/NumPy/Python RNG. A source, runtime or precision
change has a separate raw scientific identity, with a separate scoring identity.
Timing repetitions have an explicit engineering purpose. No results enter a
search population and failed/unknown attempts are never blindly retried.

Each setting has one warmup and three synchronized timed calls in a resident
model process. Times cover the complete model prediction, including per-call
packing and cleanup, and exclude model load, preprocessing, serialization and
search-controller work. These are computation timings, not search throughput.
Peak memory below is PyTorch allocated GPU memory, not total device reservation.
All settings use three recycling steps and 200 sampling steps; seeds are 0 for
the short/ligand inputs and 101 for the guided input. The ligand uses at most
1,024 MSA rows; the guided input uses at most 64. Input/receipt identities and
full numeric comparisons are in [the machine-readable report](benchmarks/fused-inference-blackwell.json).

## Selected port

The final source contains eager resident-pair fusion and fused token/atom
encoder/decoder kernels, and defaults to **off**. Select `pair`, `dit` or
`denoiser` to isolate components, or `all` for the combined path:

```bash
BOLTZ_FUSED_INFERENCE=all BOLTZ_FLASHPAIRFORMER=fast BOLTZ_DENOISER_CACHE=off \
  boltz predict input.yaml --flash_attn
```

Install the pinned `opt_core` using the [README setup](../README.md#flashpairformer-kernels)
and use the measured PyTorch 2.12 runtime. This is an opt-in reduced-precision
experiment, not a production migration or a new default.

| Input | Fork control | Final fused port | Time saved | Peak GiB, fork → port |
| --- | ---: | ---: | ---: | ---: |
| 117 tokens, 1 sample | 4.10 s | 1.96 s | 52.1% | 2.07 → 2.89 |
| 846 tokens, 1 sample | 15.69 s | 5.72 s | 63.6% | 7.80 → 6.48 |
| 846 tokens, 5 samples | 36.90 s | 8.57 s | 76.8% | 9.19 → 8.05 |
| 1,380 tokens, potentials, 3 particles / cap 1 | 91.01 s | 41.14 s | 54.8% | 17.52 → 16.34 |

An independent 846-token, one-sample ablation measured pair-only, token-only,
atom-only, combined atom/token, and full fusion. The standalone atom gain is
small; the principal gains come from pair and token fusion.

| Component | Seconds | Time saved vs fork |
| --- | ---: | ---: |
| Pair | 11.18 | 28.7% |
| Token transformer | 10.76 | 31.4% |
| Atom encoder/decoder | 15.28 | 2.6% |
| Atom + token | 10.25 | 34.7% |
| Pair + atom + token | 5.74 | 63.4% |

## Predictions and scores

Coverage is **three distinct input assemblies, four workload settings and eight
paired output structures per backend**: one short-protein prediction, one
single-sample ligand-complex prediction, five predictions of that same ligand
assembly in the five-sample setting, and one selected guided binder-complex
prediction. Each setting used one fixed seed (0 for short/ligand, 101 for guided).
The guided sampler used three particles and returned one selected prediction.

Each setting was run four times (one warmup and three timing repeats). Those
repeats were bit-identical within a mode: 16 final model calls per backend,
producing 32 recorded output structures, but eight paired output slots rather
than 32 independent results. The 156 total calls include stock/upstream
references, component ablations and final-source checks; they do not represent
156 independent targets or seeds. Structural comparison metrics use one
representative repeat because the other repeats have identical output hashes.

The port does **not** preserve predictions or scores bit for bit. The changed
rounding can lead to a different guided trajectory and selected pose even with
the same final RNG state. A higher model confidence or binding score does not
establish better experimental accuracy.

The deliberate changes include bf16 operands in the originally fp32 denoiser,
bf16 storage of the pair residual stream, and fused operations with different
rounding/reduction order. These explain loss of bit identity. Amplification
through 200 diffusion steps and guided particle selection is a plausible
explanation for the large pose change; the first divergent operation and the
contribution of each change have not been traced. No multi-seed accuracy band
was measured.

| Input | Aligned all-atom RMSD vs fork | Max absolute confidence change | Mean absolute PAE change |
| --- | ---: | ---: | ---: |
| 117 tokens, 1 sample | 0.316 Å | 0.00134 | 0.165 Å |
| 846 tokens, 1 sample | 1.723 Å | 0.00658 | 0.175 Å |
| 846 tokens, 5 samples | 0.127–1.722 Å | 0.00638 | 0.212 Å |
| 1,380 tokens, potentials, 3 particles / cap 1 | 7.571 Å | 0.00228 | 0.170 Å |

For the guided case, fitting the target gives **27.64 Å binder RMSD**.
The existing full-dimer continuous binding product changes from
**0.17594 to 0.16624 (-5.5%)**;
binder pLDDT changes from 0.82066 to 0.78748.
The denoiser-only ablation instead scores
0.20178
and has 8.33 Å target-aligned binder RMSD. These are distinct predictions,
not interchangeable evaluations of a candidate. One guided target/seed does
not establish an accuracy or seed-variation band. **Do not switch ongoing
Novel/FoldCraft populations on the strength of these timings.** A future
adoption needs an explicit predictor/scoring evidence version and broader
scientific validation.

All final repetitions are internally bit-identical, preserve the control's final
CPU/CUDA RNG state, and match the previous ablation's raw output hashes. The
final hardening and removal of graph plumbing did not change these outputs.

## Complete upstream fast configuration

The full requested upstream `fast` mode was evaluated, rather than assuming
that the fork's partial integration represented it. The default fused-OPM `f1`
schedule exceeds this Blackwell card's shared memory. Its `v2` schedule also
requires 102,400 bytes versus 101,376 available. Existing `v1` passed synthetic
qualification and served all real OPM calls without fallback or errors.
The exact LayerNorm provider still does not admit cc 12.0, retains PyTorch
LayerNorm, and makes the upstream census exit 3. The following are **requested
fast-mode results with that unsupported LayerNorm gate**, not fully admitted
upstream fast execution.

| Matched input/settings | Fork | Upstream fast, OPM v1 |
| --- | ---: | ---: |
| 117 tokens, 1 sample | 4.10 s | 0.55 s |
| 846 tokens, 1 sample | 15.69 s | 4.77 s |
| 846 tokens, 5 samples | 36.90 s | 7.72 s |
| 1,380 tokens, guided, 3-particle cap 3 | 86.37 s | 71.45 s |

Upstream 2.2.1's chunk-indexing bug ignores cap 1 for three particles. The
upstream guided comparison therefore uses cap 3 on both sides; the port's
production-shaped comparison above uses the fork's corrected cap 1. Batching
changes are not credited as optimization gains. Under potentials upstream
intentionally turns off rollout, token fusion and several related hoists;
the functional port's fused denoiser under guidance is a separately tested
extension using the fork's unchanged sampler.

The complete configuration also enables its writer/prefetch paths. Those were
observed executing, but their benefit is not isolated by model-computation
timing. The remaining MSA, template, normalization and worker levers were not
individually ablated or ported in this change.

## Graph comparison and validation

The separate graph prototype replayed real captured calls and reproduced the
fused eager outputs and scores bit for bit. It did not improve these settings:

| Input | Eager fusion | Fusion + graph |
| --- | ---: | ---: |
| 117 tokens, 1 sample | 1.92 s | 2.13 s |
| 846 tokens, 1 sample | 5.74 s | 6.36 s |
| 846 tokens, 5 samples | 8.62 s | 9.35 s |
| 1,380 tokens, potentials, 3 particles / cap 1 | 41.32 s | 41.83 s |

This branch therefore leaves PR #36 separate. It does not carry its graph code.
The three selected fusions have actual served-kernel counters in every admitted
benchmark. Pair fusion deliberately steps aside below its minimum token count
and on the C=64 template track; the short case's gain comes from denoising.

Final validation: 496 CPU tests passed on the minimum PyTorch 2.4.1 runtime
(62 skipped, 11 deselected). Five CUDA tests passed on the measured Blackwell
runtime, with nonzero model parameters, uneven chunks, padding, output
comparisons, RNG checks, scope/error cleanup, input immutability and stable
registered module identities. Pair tests cover N=128/846/1380 and require the
actual large Blackwell `tmk3_fast`/`v4` rows. Earlier graph-composition tests also
passed; the graph measurements above use the immutable evaluation snapshot.

```bash
python -m pytest tests/model/modules/test_fused_inference.py -v
python -m pytest tests/model/modules/test_fused_inference.py -m cuda -v
```

A total of 156 pretrained model calls completed across the reference,
component-ablation and final-source phases, with explicit warmup/timing
replication purposes. Existing assets and controls were reused; scientific
receipts, numerical outputs and unsuccessful engineering qualifications remain
preserved. The original production predictor was resumed without restarting its
controller or search, and a subsequent completed export was verified. No
production settings, populations or historical scores were replaced.
