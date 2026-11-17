# Experimental CUDA-graph denoiser

This prototype is **off by default**. Opt in for a prediction with:

```bash
BOLTZ_DENOISER_GRAPH=on boltz predict input.yaml --out_dir output
```

Set `BOLTZ_DENOISER_GRAPH=off` to use ordinary eager denoising. The switch also
accepts `true/false` and `1/0`; invalid values fail at prediction startup on CPU
and GPU. The independent `BOLTZ_DENOISER_CACHE` switch retains its own default
of `off`. Graphs and invariant caching can be enabled together.

The existing Boltz-2 sampler owns its noise generation, random augmentation,
sample chunks, particle resampling, potentials, rigid alignment and Euler
updates. Only `preconditioned_network_forward` is captured. Each chunk shape
gets a separate graph, including smaller final chunks and batched records.
There is no replacement sampler, approximate alignment, new precision policy,
fusion, model parameter or checkpoint key.

Automatic admission requires the stock eager Boltz-2 score-model class, CUDA,
eval mode, gradients disabled, and either fp32 or bf16 autocast. CPU, training,
compiled or custom score models and fp16 autocast retain eager execution.
Forward hooks and unversioned parameters/buffers also decline graph admission.
Unversioned CPU input tensors decline because CPU work is not replayed by a
CUDA graph. Inference-mode CUDA inputs are read live at their captured memory
addresses; their shape, layout, dtype and bindings are checked before replay.
Keep the stock model architecture/configuration fixed within an invocation.
Concurrent model execution during capture is outside this prototype's scope.

The first call for a shape executes eagerly. Three extra warmups and capture
run with CPU/CUDA RNG states preserved. A denoiser that consumes randomness
during warmup is refused. The first replay must match the eager result byte for
byte before that graph is admitted. Subsequent calls update only the coordinate
and sigma buffers, validate bindings/versions/precision, and replay. Regular
weight updates, normalization changes, input replacement and changes to
invariant-cache values invalidate affected entries. A returned graph output is
borrowed only until the next replay; the sampler immediately copies it into its
own output buffer.

Graphs belong to one `sample()` call. Finally cleanup synchronizes their devices
and releases graph objects and tensor references on return or exception. No
graph survives into another prediction. Graph private-pool accounting is capped
at 8 GiB and one quarter of free GPU memory at capture, with a conservative
working-set/headroom check before allocation. This is additional memory beyond
the invariant cache's separate 128 MiB budget. Stacks without private-pool
accounting fall back to eager execution. Capture refusals are remembered
per chunk shape; eager fallback retains the same sampler. OOM propagates, and
replay errors propagate rather than silently repeating a scientific attempt.

Debug logging for `boltz.model.layers.denoiser_graph` reports captures, replays,
eager calls, invalidations, capture time and peak private-pool size. Capture and
headroom refusals emit a warning once per shape. Check replay counts when
benchmarking: setting the switch alone does not establish that a graph served.

Run the synthetic checks with:

```bash
python -m pytest tests/model/modules/test_denoiser_graph.py -v
python -m pytest tests/model/modules/test_denoiser_graph.py -m cuda -v
```

They exercise whole-sampler output/RNG equality in fp32 and bf16, cache-on/off,
batched records, uneven chunks, alignment, guidance and particle resampling.
They require actual captures and replays, and cover live inference-mode inputs,
weight updates, budget fallback, exceptions, startup validation and cleanup.

## Measured behavior

On 27 October 2026, matched graph-off/on predictions ran on RTX PRO 6000
Blackwell (96 GB, cc 12.0), PyTorch 2.14.0+cu130 and Triton 3.8, with SDPA
enabled and the pinned FlashPairformer `fast` provider. Invariant caching was
off in both modes. Production generation paused at completed work boundaries,
with the same live controller/search processes and RNG retained, then resumed.

| Input | Samples / cap | Snapshot | Graph off | Graph on | Prediction time change |
| --- | --- | --- | --- | --- | --- |
| 117-token protein | 1 / 1 | Final | 4.09 s | 3.32 s | 18.9% less |
| 846-token protein/ligands | 1 / 1 | Initial | 15.70 s | 16.24 s | 3.4% more |
| 846-token protein/ligands | 5 / 5 | Initial | 36.95 s | 37.74 s | 2.2% more |
| 1,380-token full-dimer binder, potentials enabled | 1 / 1 | Initial | 90.81 s | 90.80 s | No measurable gain |

Each row uses three timed repetitions per mode after one warmup per mode, in
balanced off/on/on/off/off/on order. Timings include trunk, sampling and
confidence with the same resident model and exact featurized batch. Graph
construction, its extra warmups and final cleanup are included; model loading,
preprocessing and serialization are excluded. Existing processed inputs, MSA,
checkpoints and scoring policies were hash-verified and reused. Historical
predictions remain separate. Two additional bounded profiler repetitions per
row are excluded from the timing results.

The larger rows used the initial validated snapshot. Final hardening added
refusal of unversioned CPU inputs and device-explicit cleanup; the denoiser,
sampler, capture/replay body, chunking and RNG handling are unchanged. The
117-token row measures that final snapshot. Larger-row timings were not rerun
after those defensive changes; source snapshots and their comparison are
retained with the evidence.

All 40 prediction calls matched their case's graph-off reference for recorded
tensor hashes, raw coordinates, PAE, pLDDT, confidence/pTM/ipTM, score dictionaries
and final CPU/CUDA RNG state. The full-dimer case also used the established
continuous interface/binding scorer and policy: its binding product was
0.18200970651126183 in both modes. Actual replay counts were 199 per ordinary
prediction and 599 for the potential-guided, three-particle case. There were no
graph refusals or invalidations in the measured cases. This establishes tested
numerical agreement, not improved scientific accuracy or universal equivalence
on other inputs/stacks. Real affinity-head scores were not benchmarked.

The final snapshot passed 478 CPU tests on minimum PyTorch 2.4.1 (67 skipped,
11 deselected), and all 10 graph CUDA tests on Blackwell. The CUDA suite includes
invariant-cache-on/off, fp32/bf16, uneven chunks and guidance/resampling.

The three-call profiling window on the short input reduced CPU launch API calls
from 6,705 to 762. Launch count alone did not predict wall time on larger inputs:
capture and validation costs offset the benefit. Graph private pools were
72 MiB for the short input, approximately 1.41/2.96 GiB for the two ligand cases,
and 3.77 GiB for the guided full-dimer case. Graph pools are additional reserved
memory; peak allocated-memory counters alone do not describe that overhead.

Keep the prototype opt-in. The short case merits further evaluation, while
these results give no reason to enable it globally for full-dimer Novel or
FoldCraft evaluations. One example at each setting does not establish a token
cutoff or a general performance rule. The graph/invariant-cache combination has
synthetic parity coverage but has not received a real combined timing benchmark.

CUDA graphs mainly remove CPU dispatch overhead; a large sampling-time fraction
does not establish that dispatch is the bottleneck. See [PyTorch CUDA graph
semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
Anthropic's pinned [sampler rollout](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/opt/forward/sampler/src/bz_sampler.py)
is the related full-rollout design; this prototype keeps our sampler's existing
guidance and chunking instead of installing its stock-version-specific loop.
