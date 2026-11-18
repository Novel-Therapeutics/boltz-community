# Anthropic engine compatibility (experimental)

This opt-in entrypoint runs the **original** `boltz2_opt` engine at
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. It delegates its parser, profiles,
worker, kernel selection, hardware admission, census and exit rules to that
engine, with the scoped Blackwell adaptations documented below. Launch it through
`boltz2-opt` or `python -m boltz2_opt`.
The fork's regular entrypoint is `boltz predict`.

The engine, worker, optimization levers and required reference numerical source
are included in the boltz-community distribution. Install the separately distributed
`opt_core` provider using the pinned checkout's `common/opt_core` instructions in
[the README](../README.md#flashpairformer-kernels). The integrated engine requires
opt_core 0.5.228.0 or newer. The base distribution requires Python 3.11+ and
PyTorch 2.10+. The optional GPU paths have their own runtime requirements.
The historical GPU validation below used PyTorch 2.12.0+cu130,
Triton 3.7.0 and the recorded opt_core version; its
[pinned stack recipe](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltz2/STOCK.md)
describes that earlier evidence, not the current installation requirements.

The CPU suite covers Python 3.11 with PyTorch 2.10 and Python 3.14 with
PyTorch 2.10 and 2.15. The optimized GPU
profiles have not been revalidated on PyTorch 2.15. Reference numerical hashes
remain intact. Original runtime admission pins apply except for the recorded
PyTorch 2.12.0/CUDA 13.0 Blackwell qualification below: a pinned optimization can step aside
or refuse an unsupported stack. Earlier bitwise and performance results do not
establish equivalence on the new runtime. Legacy autocast API fallbacks have
been removed and the reviewed source contract records that adjustment.

```sh
boltz2-opt check --mode fast
boltz2-opt pred --mode fast --input complex.yaml --out_dir output
# The module spelling uses the same parser and behavior.
python -m boltz2_opt pred --mode exact --input complex.yaml --out_dir output
```

The original `off`, `exact`, `fast`, `big`, `BOLTZ2_OPT`, `--no-compile`,
`--allow-partial`, ablations and exit rules are preserved. Blackwell admission
additions retain the existing runtime equality checks and failure gates.
The engine is opt-in. Its explicit CLI defaults to `fast`, as upstream does.
With the optimization variables unset or `BOLTZ2_OPT=off`, Python startup
leaves the engine inactive. The installed hook reports unknown `BOLTZ2_OPT*`
and `BOLTZ2_BIG_*` variables at interpreter startup with exit 3. Use the
engine's `pred` command to select an optimized profile.
`python -m boltz.upstream` remains an alias for earlier experimental users.
No external engine checkout, stock Boltz wheel, extraction cache or
`BOLTZ_ANTHROPIC_ROOT` setting is needed for execution. If a standalone
`boltz2_opt` distribution is installed, uninstall it **before** installing this
community distribution; both distributions own the same Python package.

Use a fresh output directory for each engine `pred` or `warm` invocation.
Existing native or engine results are rejected before launch. The directory
keeps the original engine layout and records the mode and source contract in
`.boltz2-opt-run.json`. Keep the marker with the results, including after a failed
or interrupted attempt. Native `boltz predict` uses its usual resume behavior
in a separate directory. Pass `--out_dir` before the `--` stock-argument separator.
The engine checks each seed in a worker call for a distinct output path.

## Compatibility boundary and tradeoffs

The integration verifies its packaged engine and numerical source against an
explicit source contract. Reference numerical modules are carried as source in
this distribution, with their original hashes and third-party notices. It retains the fork's model
class, checkpoint loader, data pipeline, output writers and reporting, while using
the original model forward and the original trunk, pair, MSA and denoiser modules.
The WASTE proof checks the actual loaded numerical-module paths against their
original hashes. It does
**not** pretend the fork is an unmodified stock installation. Prediction transcripts
record `source_contract_sha256`, which covers the packaged fork, reference core
and engine sources and assets. `check --json` reports the fork and reference
identities separately.

Within the explicitly selected engine process:

- Upstream owns attention routing and precision, including BF16 pair residency.
  The fork's FlashAttention sequence branch and experimental FP32 pair adaptation
  are bypassed.
- Upstream owns sampler chunking. The fork's corrected cap for FK particle batches
  is bypassed, so a cap of one can still batch three particles upstream.
- Upstream owns affinity pose selection and steering. The fork's stereo-aware
  affinity selection is bypassed. Stereochemistry/bond checks are retained as
  output reporting after prediction; they do not change coordinates or scores.
- Upstream owns guidance potential math, including union-contact weighting and
  gradients. The fork's stable union weights, corrected union gradient, energy
  dtype fix and v2.14.0 symmetric-chain center-of-mass fixes apply in native
  inference. The engine uses upstream's matching atom and residue constraint
  features, so v2.14.0's extra intra-chain covalent guidance also applies only
  in native inference. Cropped constraint remapping and rejection of explicit
  self-bonds still apply in the engine.
- The fork's independent FlashPairformer route, MSA hoists and denoiser cache are
  bypassed. Their upstream counterparts are selected by the original profile.
- Original CSV/A3M parsing and MSA pairing are restored. This bypasses the
  fork's paired-duplicate retention, pairing bookkeeping correction, UniRef
  pairing-key behavior and query-trimming fallback. These change model inputs;
  preserving them would change the tensors the engine receives. The engine uses the
  fork's multi-model template parsing and PDB connectivity and termination fixes.
  Its confidence JSON includes original input chain labels and structure checks.
  The single-GPU head also supplies the fork's extra PAE summaries, including
  with writer overlap. The original multi-GPU head omits those extra summaries.
  The engine
  prepares inputs through the original CLI and applies its per-input and per-seed
  failure and provider rules. In `off`, writer-recorded skips count as failed
  units while completed structures remain eligible for affinity. A missing
  structure for any remaining affinity request is reported as an error. Affinity retains the engine's original
  single-binder orchestration.

No fork source is globally rolled back. Native fork execution retains these fixes
and optimizations. The compatibility entrypoint verifies the recorded source
snapshot. Maintainers can refresh the source contract after reviewing changes
with `python3 scripts/update_upstream_contract.py` on Python 3.11 or newer.
The runtime only reads the contract.

## Validation status

### Blackwell kernel qualification, 18 November 2026

[Recorded receipts and comparisons](benchmarks/blackwell-profile-qualification.json)
cover the RTX PRO 6000 Blackwell on bizon, compute capability 12.0, PyTorch
2.12.0+cu130, Triton 3.7.0 and opt_core 0.5.228.0. The tested implementation is
commit `2db3842`, based on `0607812`. GPU access waited for production to exit;
no production process was paused. Disposable validation containers completed and
the GPU had no compute processes afterward.

Two separate causes produced the earlier refusals:

- LayerNorm, fused DiT softmax/glue and exact MSA PWA/transition had architecture
  allowlists covering previously tested GPUs. Blackwell was unqualified. Direct
  Blackwell probes passed bitwise: 66 LayerNorm classes, 14 softmax classes,
  nine glue classes, ten transition configurations, and a uniquely matching PWA
  summation candidate at each of ten shapes. Source- and runtime-scoped adapter
  admission now retains their runtime bit comparisons. Provider files and
  numerical kernel bodies are unchanged.
- Fused OPM's default tile exceeds this GPU's per-block shared-memory limit.
  Reducing pipeline stages alone still required 131,072 bytes against a 101,376
  byte limit. The community `community_sm120` configuration reduces `BJ` from
  eight to four and stages from three to two. Its compiled kernel uses 99,328
  bytes, preserving the fused prologue, TMA and fused projection. Explicit user
  and provider card configurations take precedence. This remains a tolerance
  optimization in `fast`/`big`, with nonzero-weight numerical probes recorded.

The installed wheel was tested after removing the original engine tree and
standalone distributions from disposable containers. The 117-token unguided
example and 636-token guided 8QLV case completed with all three optimized
profiles returning **0**. An additional unguided 636-token `off`/`exact` pair
returned **0** and exercised fused softmax/glue, with both MSA exact cells
proven on 41,721,600 compared elements and then serving 15 calls each. No
`--allow-partial` flag or new card-specific disabled-lever list was introduced.
The focused CPU/source-contract suite passed 40 checks.

Execution success does not establish whole-prediction equality or production
accuracy. The public example's `exact` coordinates and confidence match `off`.
For 8QLV, the aligned all-atom `exact`/`off` PDB RMSD is 0.010594 Å with guidance
and 0.014243 Å without guidance. The cause of this full-profile difference is
unresolved; individual kernel checks do not certify end-to-end bitwise equality.
Guided 8QLV seed 0 DockQ is 0.755949 (`off`), 0.756501 (`exact`), 0.434991 (`fast`)
and 0.599490 (`big`). These are selected regression diagnostics. They do not
justify automatically adopting `fast` or `big` for Novel Design production.
Cold process timings in the receipts are not speedup measurements. Other GPUs,
PyTorch versions, affinity and wider workloads need separate qualification.

### Historical compatibility snapshots

The GPU reports below retain the identities of the earlier measured sources.
The integration on v2.14.0 changes the source identity. Those receipts provide
historical parity and timing evidence for their recorded snapshots.

The earlier compatibility harness was validated on Blackwell on 2 November 2026 with PyTorch 2.12.0+cu130,
Triton 3.7.0 and opt_core 0.5.228.0. [Recorded results](benchmarks/anthropic-engine-parity.json):

- **23 matched configurations / 46 predictions:** every shared input/output tensor,
  confidence score and initial/final RNG state matched the original engine bit for
  bit. PDB records matched after ignoring entirely blank lines. Extra fork report
  fields remain extra fields.
- Guided `off`/`fast`/`big`: 8QLV, 8JEO and 8PIH, two matched seeds each; an
  additional guided `exact` control used 8PIH seed 0.
- Unguided `off`/`exact`/`fast`/`big`: the existing 117-token public example, seed 0.
  The census confirmed one sampler capture and **199 replays** for `exact` and
  `fast`, level-2 conditioning hoists, and trunk graph capture/replay. `big` used
  its original eager path and served **200 fused DiT steps**.
- Focused CPU regression suite: **264 passed, 37 skipped**, including 12 new
  compatibility-boundary tests. The original parser help, parsed flag examples
  and profile table were unchanged. A built community-only wheel also passed
  with stock `boltz` distribution metadata absent.

The final audit added bootstrap routing for both one-shot and served affinity
children. This transport-only change has 12 boundary tests and a CPU affinity
startup check; one final-source unguided `fast` prediction matched both saved
controls, including RNG. The report distinguishes the 46-prediction GPU snapshot
from this final source delta. No numerical implementation changed.

Mean DockQ over the two seeds; upstream and the compatible fork give the same values:

| Complex | `off` | `fast` | `big` |
| --- | ---: | ---: | ---: |
| 8QLV | 0.756 | 0.376 | 0.598 |
| 8JEO | 0.253 | 0.329 | 0.410 |
| 8PIH | 0.030 | 0.032 | 0.032 |

The 8QLV loss also occurs in the original complete package. These were selected
regression cases, so this is an integration test, not an unbiased estimate of
accuracy or an accuracy guarantee for a fast profile. The recorded cold timings
share JIT caches and are diagnostic only; they do not measure a speedup.

**The historical snapshots retained upstream Blackwell limitations.** The original exact LayerNorm
provider declines compute capability 12.0 (`cc:12.0_not_proven_(9.0,_8.0)`), and the
original default fused OPM configuration produces `OutOfResources` on the deep-MSA
regression inputs. Both implementations write predictions but return exit **3**
for the affected profiles. `--allow-partial` does not override these failed
execution/evidence gates. No kernel configuration, rounding policy or acceptance
gate was silently changed to make the exit green. The `check` command plans a run;
it cannot establish that every kernel will execute successfully.

Keep this path opt-in. Normal fork execution retains its existing fixes; globally
reverting them would change users' input processing and sampling. The compatible
path deliberately restores original MSA semantics, precision, chunking and affinity
selection. Template, ligand/cropped-constraint and invalid-input behavior, H100
native paths, multiple GPUs and multi-input persistent-worker reuse need their
own qualification before claiming full coverage. Affinity child startup is CPU-tested;
affinity predictions, alternative checkpoints and Boltz-1 are not qualified here.
Additional fork reports remain different from unmodified stock.

Guided profiles differ upstream: `fast --use_potentials` disables fused DiT and its
graph/hoist, while `big --use_potentials` supports eager fused DiT. Use each resolved
profile's census as the execution evidence. Remaining upstream kernel refusals
retain their failure behavior. The current Blackwell adaptations and their
qualification are explicit above.

## Historical integrated-source validation

The measured source port retained the original numerical bodies, parser, profile
table and worker implementation. Control-plane changes locate packaged assets, launch the
integrated distribution and identify the actual fork/reference sources. The
Apache-2.0 engine notices and third-party licences are carried with those assets.
The build backend preserves upstream's installed autoload hook for both wheel and
editable installs.

The installed distribution was checked in an isolated CPU container after
uninstalling both stock Boltz and the standalone engine and removing the original
kit tree. All three profiles stage their original workers and package-relative
assets resolve. Wheel and editable installs passed; a wheel rebuilt from the
source distribution has the same 253 non-metadata payload files byte for byte.
Strict checkpoint loading reproduced all 5,102 state keys and the saved Torch,
NumPy and Python constructor RNG states. The final-source CPU cohort passed
**280 checks**, including 18 boundary checks; 37 CUDA tests were deselected.

[Installed-source results](benchmarks/anthropic-source-port-validation.json)
separate the following snapshots from the earlier 46-prediction harness:

- The initial packaged-source snapshot ran the same **23 configurations** against
  reused original controls: three selected regression complexes, two seeds each,
  guided `off`/`fast`/`big`, one additional guided `exact` control and the 117-token
  unguided control under all four modes. Every shared input tensor and initial/final
  RNG state matched. Twenty-two outputs matched the historical controls bit for
  bit; the remaining `big` 8QLV output was subsequently reproduced exactly by the
  unmodified original engine's repeatability test.
- The final-source delta restores the original potential module inside engine
  workers, including union-contact guidance. Its unforced 8QLV `big` prediction
  matched the original control bit for bit. A forced-pocket fixture exercised the
  restored contact path with identical shared inputs and RNG states. Its original
  versus port valid-atom RMSD was 0.140 Å and DockQ was 0.76765 versus 0.77352.
  Original-only repeats of that fixture varied by up to 0.389 Å. This fixture is a
  compatibility diagnostic, not an independent accuracy example or an equivalence
  guarantee.
- The packaged unguided `fast` route executed one sampler capture and **199
  replays**, level-2 conditioning hoists, and one trunk capture with two replays.
  These are execution counts, not merely requested settings. The overall profile
  still returned 3 for the inherited Blackwell refusals described above.

### Repeatability

Four same-input, RNG-restored `big` calls in each implementation produced the
**same two exact output variants across all 18 shared tensors** on unforced 8QLV.
Their valid-atom RMSD was 0.00574 Å and their confidence-score difference was
0.00000560. Thus the earlier packaged-source outlier is an observed upstream
outcome, rather than evidence of a new integration error. The `off` repeats were
bit-identical. Timing and diagnostic repetitions are not independent accuracy
examples.

Matched RNG does not guarantee deterministic CUDA reductions. The original
guidance uses `scatter_reduce`; [PyTorch documents CUDA nondeterminism for this
operation](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.scatter_reduce_.html).
The repeated original runs establish non-repeatability here; the particular
operation responsible was not isolated. No determinism or precision setting was
changed to hide it.

### Warmed prediction timing

One warmup and three timed calls per mode, on 8QLV seed 0 (636 padded tokens,
5,014 valid atoms), three recycles, 200 sampling steps, one diffusion sample,
three FK particles, a parallel cap of three and guidance enabled:

| Implementation / mode | Median `predict_step` | Final-repeat DockQ |
| --- | ---: | ---: |
| Port, original-compatible `off` | 19.950 s | 0.75595 |
| Port, `big` | 7.267 s | 0.41976 |
| Original engine, `big` | 7.369 s | 0.41976 |

The port's `big` prediction was **2.75× faster than original-compatible `off`** on
this case. Timings synchronize CUDA and exclude process startup, checkpoint loading,
input preparation and output writing. They do not compare against native
`boltz predict --flash_attn`, and they do not establish an end-to-end production
speedup. The last-repeat PyTorch allocation peaks were 5.28 GiB (`off`) and
6.15 GiB (`big`), so this case does not demonstrate a memory saving.

The accuracy loss is also upstream's, as the table shows. Optimized profiles
returned **3** on this Blackwell stack, matching the original engine; partial
execution latency is not successful complete-profile qualification. Keep the
feature off by default. H100/H200/GH200 complete-profile tests, affinity predictions,
other checkpoints, multi-GPU/persistent multi-input qualification and broader
accuracy evaluation remain outstanding before production adoption.
