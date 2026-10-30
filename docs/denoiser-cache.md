# Denoiser invariant caching

With `BOLTZ_DENOISER_CACHE=on`, eager Boltz-2 sampling reuses
timestep-independent conditioning work within one `AtomDiffusion.sample`
invocation. When enabled, caching requires CUDA, eval mode, gradients disabled,
and either fp32 execution or CUDA bf16 autocast. Compiled or custom score models
retain their original call path. CPU, training, Boltz-1 and direct denoiser
calls do not automatically create a cache.

The regular Boltz-2 CLI disables autocast around sampling, so its denoiser runs
in fp32 even when the trunk uses bf16. The bf16 cache path supports direct
library callers that enable autocast; it is not the regular CLI sampling path.

Set `BOLTZ_DENOISER_CACHE=on` to opt in; the default is `off` pending performance
validation across representative workloads. `true/false` and `1/0` are also
accepted. Invalid values are rejected before prediction setup on CPU and GPU,
and before any direct sampling invocation. The setting is read once per
sampling invocation. A direct `sample` caller should use `eval()` and
`torch.no_grad()` or `torch.inference_mode()`, as the regular prediction path does.

The cache reuses:

- The SingleConditioning prefix: concatenation of the expanded trunk and input
  embeddings, LayerNorm and the single embedding projection.
- The fp32 atom-conditioning tensor and its sample-multiplicity expansion.
- Atom-layer AdaLN conditioning normalization, sigmoid scale and bias, and the
  attention/transition output gates derived from that conditioning.

The original sample expansion and matrix shapes are preserved. Uneven sample
chunks have separate entries. The Fourier/time contribution, subsequent single
conditioning transitions, token-transformer modulation, coordinate-dependent
atom activations, attention and output projections still run each denoiser call.
Main diffusion conditioning was already computed once and is unchanged.

Each invocation owns versioned snapshots of its trunk/input embeddings and atom
conditioning. This isolates them from caller edits and lets internal in-place
edits invalidate entries even under Lightning's inference mode. Entry signatures
include tensor geometry, version counters, weight identity/version, normalization
settings and execution precision. Unversioned inputs or parameters are not used
for reusable entries. Changes to cached result tensors also invalidate entries.
The cache is passed explicitly through atom layers and is not stored on a model
or in global tensor state. Checkpoint keys are unchanged.

Retained tensor storage is bounded by the smaller of 128 MiB and one eighth of
GPU free memory at admission. Accounting counts shared storage once, including
retained source/weight references. A value that does not fit is computed normally
without being retained; that role is then bypassed for the remainder of the
sampling invocation, without repeated cache validation or size accounting.
If input snapshots do not fit, the invocation uses the
original path. Large quadratic pair-bias expansions are not cached. All entries
are cleared on return or exception. Exceptions, including OOM, propagate normally.
Debug logging for `boltz.model.layers.denoiser_inference` reports hits, builds,
budget declines and peak retained storage once per sampling invocation.

The budget can fill with ordinary workloads: cached atom modulation grows with
batch size, samples per chunk, atom count and atom width. Entries are kept at the
original expanded batch shape to preserve numerical agreement. Metadata plans
retain parameter/module traversal results, while lightweight binding and
normalization-setting checks still detect replacement or configuration edits.
Per-call tensor/version and precision checks are retained.

This ports the invariant-work approach from Anthropic's pinned
[`dit_hoist` level 1](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/opt/forward/dit_hoist/src/boltz_dit_hoist.py)
into the existing sampler. It does not replace sampling, guidance or indexing
with the bundle's stock-version-specific implementation and requires no
`opt_core` installation.

Run the synthetic checks with:

```bash
python -m pytest tests/model/modules/test_denoiser_inference.py -v
python -m pytest tests/model/modules/test_denoiser_inference.py -m cuda -v -rs
```

The tests use artificial tensors and randomized nonzero model parameters, with
no sequence input or pretrained checkpoint. They compare complete sampler
outputs and final RNG states, exercise batched records and uneven chunks, and
cover resampling, guidance, alignment, precision changes, mutation invalidation,
memory-budget fallback and exception cleanup. Compiled-score tests use Dynamo's
`eager` backend and the existing `fullgraph=False` configuration; component
full-graph tests cover SingleConditioning and AdaLN. CPU admission probes are
explicitly mocked; CUDA cases exercise the production admission predicate.

On a three-step, two-record synthetic case with 11 samples and a cap of five,
the fixed prefix and each observed atom normalization run twice, for the two
chunk shapes, instead of nine times. Fourier and token normalization still run
nine times. This establishes eliminated work, not an end-to-end speed-up.

On 27 October 2026, a real pretrained prediction comparison used the 846-token
`examples/ligand.yaml` assembly on an isolated NVIDIA A10G, PyTorch
2.12.0+cu130, FlashPairformer `fast` at the pinned `opt_core` commit above, and
`--flash_attn`. Both modes used seed 0, three recycling steps, 200 sampling
steps, fp32 denoising and default contact guidance, without `--use_potentials`.
Existing hash-verified processed input, MSA and checkpoints were reused.

| Samples / chunk cap | Prediction, cache off | Prediction, cache on | Prediction time saved | Sampler time saved |
| --- | --- | --- | --- | --- |
| 1 / 1 | 56.87 s | 56.50 s | 0.65% | 1.36% |
| 5 / 5 | 156.44 s | 156.14 s | 0.19% | 0.25% |

These are median synchronized model-prediction times, including trunk, sampling
and confidence, with the same resident model and exact featurized batch. Model
loading, preprocessing and result serialization are excluded. Each case had
one warmup per mode followed by two timed repetitions per mode in balanced
off/on/on/off order. All repetitions had identical output tensor hashes and
CPU/CUDA RNG states within each sample-count case; coordinates and confidence
scores had zero difference. This checks numerical agreement, not predictive
accuracy against experimental structures.

One sample retained all 26 entries (121.1 MiB). Five samples retained six entries
(123.3 MiB), refused 20 entries once, and bypassed their bookkeeping on 3,980
later calls. Whole-prediction peak allocated GPU memory was unchanged in these
cases. The small gains on this one assembly and GPU do not justify enabling the
cache globally. It remains opt-in; broader workloads and CUDA-graph integration
need separate evaluation. Real potential-guided and affinity predictions were
not timed.

Validation of the review fixes: 470 tests passed on the minimum PyTorch 2.4.1
CPU runtime (57 skipped, 11 deselected). Eight denoiser CUDA cases and all
15 MSA CUDA cases passed on the isolated A10G. Sampling tests include fp32 and
bf16 autocast, partial budgets, mixed-precision conditioning, guidance and
uneven chunks. Formatting after validation was checked for identical Python
ASTs against the tested source snapshot.
