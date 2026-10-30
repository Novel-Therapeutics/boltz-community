# boltz-community

Community-maintained fork of [Boltz](https://github.com/jwohlwend/boltz) with bug fixes, broader compatibility, and CI.

## What's different from upstream?

**Compatibility:**
- Apple Silicon (MPS) support: `boltz predict --accelerator mps` ([#1](https://github.com/Novel-Therapeutics/boltz-community/pull/1))
- Intel GPU (XPU) support: `boltz predict --accelerator xpu`. Thanks to [@UniversalBasicInc](https://github.com/Novel-Therapeutics/boltz-community/pull/20).
- Added `boltz-fix-macos-libomp` to fix `OMP: Error #15` on macOS: it points the duplicate `libomp.dylib` copies bundled by torch and scikit-learn at one runtime, instead of the unsafe `KMP_DUPLICATE_LIB_OK=TRUE` ([#15](https://github.com/Novel-Therapeutics/boltz-community/pull/15), thanks to [@fnachon](https://github.com/fnachon))
- Dependency pins relaxed from `==` to `>=`
- `fairscale` dependency removed — replaced with PyTorch built-in `torch.utils.checkpoint`
- `numpy<2.0` cap removed
- `requires-python` widened to `>=3.10` (removed `<3.13` cap)
- Requires PyTorch 2.4 or newer; CI tests the 2.4 minimum alongside the latest release. Compatible with Lightning 2.6+

**Bug fixes:**
- Fixed ROCm DDP crashes by allocating tensors directly on the target device instead of CPU-then-move ([#654](https://github.com/jwohlwend/boltz/pull/654))
- Fixed `--write_full_pae` and `--write_full_pde` being ignored, and added matching `--no_write_full_pae` and `--no_write_full_pde` flags ([#602](https://github.com/jwohlwend/boltz/pull/602))
- Fixed MSA CSV parsing removing paired duplicate sequences across different taxa ([#584](https://github.com/jwohlwend/boltz/pull/584))
- Fixed MSA feature construction incorrectly including paired sequences as unpaired ([#582](https://github.com/jwohlwend/boltz/pull/582))
- Fixed consecutive CA filter rejecting valid protein chains containing metal ions ([#576](https://github.com/jwohlwend/boltz/pull/576))
- Fixed template alignment forcing gapless matches, breaking templates with indels ([#538](https://github.com/jwohlwend/boltz/pull/538))
- Fixed relative MSA paths resolved from CWD instead of input file directory ([#500](https://github.com/jwohlwend/boltz/pull/500))
- Fixed MSA authentication headers forcing `Content-Type`, which breaks some MSA servers ([#488](https://github.com/jwohlwend/boltz/pull/488))
- Fixed CLI help/default mismatches and added option validation for MSA pairing strategy ([#463](https://github.com/jwohlwend/boltz/pull/463))
- Fixed sign error in binding energy calculation documentation ([#363](https://github.com/jwohlwend/boltz/pull/363))
- Fixed broken v1 attention code path in `PairformerLayer` ([#602](https://github.com/jwohlwend/boltz/pull/602))
- Fixed SIGSEGV crash on ligands with invalid implicit valence ([#649](https://github.com/jwohlwend/boltz/issues/649))
- Fixed `--subsample_msa` defaulting to False instead of True ([#628](https://github.com/jwohlwend/boltz/issues/628))
- Fixed 2-char elements (Ca, Fe, Br, Cl) misidentified in PDB/mmCIF output ([#458](https://github.com/jwohlwend/boltz/issues/458))
- Fixed atom name overflow (>4 chars) crashing large molecule processing ([#494](https://github.com/jwohlwend/boltz/issues/494))
- Fixed null bytes in A3M files crashing MSA parsing ([#509](https://github.com/jwohlwend/boltz/issues/509))
- Fixed bfloat16 dtype mismatch in potentials ([#625](https://github.com/jwohlwend/boltz/issues/625))
- Fixed CCD tar re-download on every run when `mols/` already exists ([#633](https://github.com/jwohlwend/boltz/issues/633))
- Fixed empty CIF files causing cryptic errors ([#641](https://github.com/jwohlwend/boltz/issues/641))
- Fixed hardcoded "LIG" residue name in PDB HETATM records ([#630](https://github.com/jwohlwend/boltz/issues/630))
- Fixed mmCIF entity deduplication for chemically distinct ligands ([#630](https://github.com/jwohlwend/boltz/issues/630))
- Fixed chirality constraint computation missing stereo assignment ([#589](https://github.com/jwohlwend/boltz/issues/589))
- Fixed multi-CCD ligands not dropping leaving atoms ([#631](https://github.com/jwohlwend/boltz/issues/631))
- Fixed `--preprocessing-threads` overcommitting CPUs ([#564](https://github.com/jwohlwend/boltz/issues/564))
- Fixed a silent wrong answer: inference no longer substitutes a different record when one fails; the error now propagates ([098d63f](https://github.com/Novel-Therapeutics/boltz-community/commit/098d63f))
- Fixed a possible stack overflow in training/validation data loading by bounding retries at 10 ([098d63f](https://github.com/Novel-Therapeutics/boltz-community/commit/098d63f))
- Fixed `boltz predict` exiting with code 0 when every input fails validation ([0df0383](https://github.com/Novel-Therapeutics/boltz-community/commit/0df0383))
- Fixed MSA pairing keys lost when loading cached A3M files ([#627](https://github.com/jwohlwend/boltz/issues/627))
- Fixed affinity prediction crashing when structure prediction fails (covalent ligands, OOM); affected records are skipped with a warning ([#620](https://github.com/jwohlwend/boltz/issues/620), [#624](https://github.com/jwohlwend/boltz/issues/624))
- Fixed affinity for repeated ligand binders: each binder copy now gets its own `affinity_<record>_<chain>.json` ([#647](https://github.com/jwohlwend/boltz/issues/647))
- Fixed Boltz-2 checkpoint loading crash due to extra `mse_rotational_alignment` kwarg ([#644](https://github.com/jwohlwend/boltz/issues/644))
- Fixed direct Boltz-2 model use from checkpoint hyperparameters crashing during sampling when `steering_args` is missing ([#680](https://github.com/jwohlwend/boltz/issues/680))
- Fixed empty checkpoint files causing cryptic load aborts: empty cached weights are re-downloaded, with a clear error otherwise ([#664](https://github.com/jwohlwend/boltz/issues/664))
- Fixed distorted CPU predictions with wrong bond lengths: Boltz-2 ran in `bf16-mixed` on CPU and now uses float32 ([#653](https://github.com/jwohlwend/boltz/issues/653))
- Fixed missing PAE summaries in `confidence_*.json`: `complex_pae`, `complex_ipae`, `chains_pae` and `pair_chains_pae`, contact-weighted like PDE, also with `--no_write_full_pae` ([#607](https://github.com/jwohlwend/boltz/issues/607))
- Fixed cuequivariance triangle multiplication crashes by falling back to PyTorch when kernels are unavailable ([#485](https://github.com/jwohlwend/boltz/issues/485), thanks to [#682](https://github.com/jwohlwend/boltz/pull/682))
- Fixed the same cuequivariance crash in triangle attention, which the v2.10.3 fix missed ([#485](https://github.com/jwohlwend/boltz/issues/485))
- Fixed forced contact restraints losing all signal when union weighting underflowed for large violations ([#621](https://github.com/jwohlwend/boltz/issues/621), thanks to [#682](https://github.com/jwohlwend/boltz/pull/682))
- Fixed the contact-union gradient sign, so soft-OR contact restraints pull with the correct magnitude ([7032160](https://github.com/Novel-Therapeutics/boltz-community/commit/7032160))
- Fixed diffusion ignoring `--max_parallel_samples` when the sample count divides evenly, which could batch everything at once and run out of memory ([#7](https://github.com/Novel-Therapeutics/boltz-community/issues/7))
- Fixed MSAs rejected as "does not match input sequence" when aligned to a longer UniProt sequence than the input construct: the MSA is now trimmed to the construct, tolerating up to 5% mismatches ([#6](https://github.com/Novel-Therapeutics/boltz-community/pull/6))
- Fixed PDB templates crashing with `IndexError` when Gemmi drops entity sequences; relative template paths now resolve from the YAML's directory ([#669](https://github.com/jwohlwend/boltz/issues/669))
- Fixed YAML `bond` constraints between residues (for example ACE-CY3 cyclization) getting no bond-length bounds for guidance and no mmCIF `_struct_conn` ([#675](https://github.com/jwohlwend/boltz/issues/675))
- Fixed Boltz-2 fine-tuning/validation crashing on `self.validate_structure`, which the constructor never stored ([#11](https://github.com/Novel-Therapeutics/boltz-community/issues/11))
- Fixed training preprocessing leaving `cluster_id` and `msa_id` empty, so training ran with uniform cluster weights and no MSAs. Inference unaffected; re-preprocess and retrain ([#686](https://github.com/jwohlwend/boltz/issues/686), [#14](https://github.com/Novel-Therapeutics/boltz-community/issues/14))
- Fixed affinity running about 5× slower than needed: it hard-coded one sample per pass; it now uses `--max_parallel_samples`, capped at `--diffusion_samples_affinity` ([5103619](https://github.com/Novel-Therapeutics/boltz-community/commit/5103619))
- `--use_potentials` now also steers the affinity pass, which samples its own pose; unsteered, it often had the wrong stereochemistry. The affinity step is about 2.4× slower with the flag ([#30](https://github.com/Novel-Therapeutics/boltz-community/pull/30))
- Fixed salt-form ligand SMILES aborting the whole affinity batch with an uncatchable RDKit error ([7407b86](https://github.com/Novel-Therapeutics/boltz-community/commit/7407b86))
- Raised MSA submit and download timeouts from 6.02 s to 300 s, removing spurious `api.colabfold.com` timeouts ([#688](https://github.com/jwohlwend/boltz/pull/688))
- Fixed transient MSA download errors being cached as `out.tar.gz` and later failing with `tarfile.ReadError`; downloads are validated before caching ([#18](https://github.com/Novel-Therapeutics/boltz-community/pull/18))
- Fixed `ZeroDivisionError` in the learning-rate scheduler with warmup disabled (training only) ([#709](https://github.com/jwohlwend/boltz/pull/709), thanks to [@edusbarbosa](https://github.com/edusbarbosa))
- Fixed PDB output losing ligand bond orders: `CONECT` now repeats partners for double and triple bonds ([#196](https://github.com/jwohlwend/boltz/pull/196), thanks to [@chr-szogun](https://github.com/chr-szogun))
- Fixed PDB output corrupting columns for multi-character chain names: each chain gets a unique one-character ID, with a warning ([#210](https://github.com/jwohlwend/boltz/pull/210), thanks to [@ameya98](https://github.com/ameya98))
- Fixed chain ids over 5 characters being truncated and failing later with `KeyError`; they are now rejected with a clear error ([#181](https://github.com/jwohlwend/boltz/issues/181), [#188](https://github.com/jwohlwend/boltz/pull/188), thanks to [@alexcritschristoph](https://github.com/alexcritschristoph) and [@jsilter](https://github.com/jsilter))
- Fixed mmCIF output of `cyclic: true` polymers missing the ring-closing bond ([#710](https://github.com/jwohlwend/boltz/issues/710), thanks to [@TSC2412](https://github.com/TSC2412))

**Improvements:**
- Published to PyPI as [`boltz-community`](https://pypi.org/project/boltz-community/): `pip install boltz-community` or `uv add boltz-community` ([#12](https://github.com/Novel-Therapeutics/boltz-community/issues/12))
- `boltz predict` now aborts when any input fails processing; `--skip_bad_inputs` skips them instead ([c4ab80c](https://github.com/Novel-Therapeutics/boltz-community/commit/c4ab80c))
- Deferred heavy imports (torch, rdkit, Lightning) so CLI help and input validation start instantly ([396275a](https://github.com/Novel-Therapeutics/boltz-community/commit/396275a))
- `--devices` accepts specific GPU IDs as well as a count, e.g. `--devices 0,1` ([#5](https://github.com/Novel-Therapeutics/boltz-community/pull/5))
- Added unweighted PAE summaries (`*_unweighted`) alongside the contact-weighted ones, for binder metrics such as `pae_interaction`. See [docs](docs/prediction.md) ([#707](https://github.com/jwohlwend/boltz/pull/707), thanks to [@AtharvaTilewale](https://github.com/AtharvaTilewale))
- Added `--batch_size` for Boltz-2 structure inference. Affinity stays single-record, and `--use_potentials` needs `--batch_size 1` ([#645](https://github.com/jwohlwend/boltz/issues/645))
- Documented that converting `affinity_pred_value` to ΔG treats an IC50 as a dissociation constant, which it is not in general ([#698](https://github.com/jwohlwend/boltz/pull/698), thanks to [@Michaelihc](https://github.com/Michaelihc))
- `boltz predict` warns when an input has stereocentres and `--use_potentials` is off; without it, stereochemistry is often wrong. See [Stereochemistry](docs/prediction.md#stereochemistry) ([#710](https://github.com/jwohlwend/boltz/issues/710))
- Added structure checks to Boltz-2's `confidence_*.json`: inverted or flattened stereocentres and out-of-range bonds per model. Affinity uses the best sample that passes ([#29](https://github.com/Novel-Therapeutics/boltz-community/pull/29))

**Performance improvements:**
- An experimental CUDA-graph denoiser prototype is available with `BOLTZ_DENOISER_GRAPH=on` (default `off`). See [scope, numerical checks and benchmark results](docs/denoiser-graph.md).
- Optional denoiser conditioning reuse is available with `BOLTZ_DENOISER_CACHE=on` (default `off`). It caches fixed conditioning within one sampling call; see [scope, memory limits and validation](docs/denoiser-cache.md). Use `BOLTZ_DENOISER_CACHE=off` to disable it.
- Triangle attention and triangle multiplication run on Anthropic's FlashPairformer kernels by default when [`opt_core`](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/common/opt_core) is installed, with the cuEquivariance and PyTorch paths as fallbacks (see [FlashPairformer kernels](#flashpairformer-kernels))
- Added optional FlashAttention-2 / SDPA attention via `--flash_attn`: less memory and faster on Ampere+ GPUs, numerically equivalent ([#13](https://github.com/Novel-Therapeutics/boltz-community/pull/13))
- Checkpoints are staged on CPU before moving to the accelerator: ~1.75 s faster warm load and ~4.4 GiB less peak VRAM, for ~1 GiB more host RAM ([#16](https://github.com/Novel-Therapeutics/boltz-community/pull/16), thanks to [@corinwagen](https://github.com/corinwagen))
- Cached molecule file reads and symmetry deserialization across samples ([b2d9e06](https://github.com/Novel-Therapeutics/boltz-community/commit/b2d9e06))
- Removed a dead O(n_tokens × n_chains) loop in pocket distance computation ([6dae342](https://github.com/Novel-Therapeutics/boltz-community/commit/6dae342))
- Tensors across model modules now allocated directly on device instead of CPU-then-transfer ([#654](https://github.com/jwohlwend/boltz/pull/654))
- Vectorized the featurizer's MSA pairing fill, removing a per-row Python loop ([ba3f6cf](https://github.com/Novel-Therapeutics/boltz-community/commit/ba3f6cf))
- `process_atom_features` pre-allocates its arrays and fills `atom_to_token` per token, not per atom ([ba3f6cf](https://github.com/Novel-Therapeutics/boltz-community/commit/ba3f6cf))

**Tests & CI:**
- 362 tests vs. 5 in upstream `jwohlwend/boltz`: unit, end-to-end smoke, golden-output regression (Boltz-1 and Boltz-2) and determinism tests
- GitHub Actions CI with CPU runners (every push/PR) and GPU T4 runners (push to main)

## Contributors

boltz-community builds on [Boltz](https://github.com/jwohlwend/boltz) by Jeremy Wohlwend, Gabriele Corso, Saro Passaro and the upstream contributors. GitHub doesn't list contributors on forks, so see the [contributors page](https://github.com/Novel-Therapeutics/boltz-community/graphs/contributors) for everyone with commits here.

Thanks to everyone whose code, upstream PRs or reports are in this fork:

- [@UniversalBasicInc](https://github.com/UniversalBasicInc): Intel GPU support
- [@corinwagen](https://github.com/corinwagen): CPU checkpoint staging
- [@fnachon](https://github.com/fnachon): macOS libomp fix
- [@AtharvaTilewale](https://github.com/AtharvaTilewale): unweighted PAE summaries
- [@edusbarbosa](https://github.com/edusbarbosa): scheduler fix with warmup disabled
- [@chr-szogun](https://github.com/chr-szogun): ligand bond orders in PDB output
- [@ameya98](https://github.com/ameya98): PDB chain IDs
- [@alexcritschristoph](https://github.com/alexcritschristoph) and [@jsilter](https://github.com/jsilter): chain id length
- [@Michaelihc](https://github.com/Michaelihc): affinity-to-ΔG documentation
- [@TSC2412](https://github.com/TSC2412): cyclic D-peptide stereochemistry report

## Contributing

Pull requests are welcome! If you have a bug fix, test improvement, or compatibility enhancement, please open a PR.

## Installation

Install from PyPI:

```
pip install boltz-community
```

With CUDA kernels:

```
pip install "boltz-community[cuda]"
```

`uv` works the same way:

```
uv add boltz-community            # or: uv add "boltz-community[cuda]"
```

If you are installing on CPU-only or non-CUDA GPU hardware, omit `[cuda]`. Note that the CPU version is significantly slower than the GPU version.

### FlashPairformer kernels

Anthropic's [uplifting-biomolecular-modeling](https://github.com/anthropics/uplifting-biomolecular-modeling) release includes `opt_core`, a package of FlashPairformer kernels (Triton and prebuilt CUDA) for the Pairformer's triangle attention and triangle multiplication. It is not on PyPI. Install it the way that repository documents, as an editable install from a clone, so that its prebuilt CUDA kernels are found (a regular `pip install` of `common/opt_core` leaves some of them out, and those calls then use the slower Triton kernels):

```
pip install "boltz-community[cuda]"
git clone https://github.com/anthropics/uplifting-biomolecular-modeling.git
git -C uplifting-biomolecular-modeling checkout f4f62fa6592ae4938d49b1757bea0cfeff9f468e
pip install -e uplifting-biomolecular-modeling/common/opt_core
```

The adapter's API checks and A10G validation used `opt_core` 0.5.228.0 at commit [`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`](https://github.com/anthropics/uplifting-biomolecular-modeling/commit/f4f62fa6592ae4938d49b1757bea0cfeff9f468e), pinned above. Native shared-cache invalidation still requires GPU validation on compute capability 9.0 (H100, H200 or GH200); the A10G results do not cover it. If installing another revision, run `TestOptCoreInterface` in `tests/model/layers/test_flashpairformer.py` to check the required APIs, including `trimul.native.shared_geometries()` and `payload_cache()`.

When `opt_core` is importable, `boltz predict` uses it by default on NVIDIA GPUs with compute capability 8.0 or newer (A100, H100, and later). For each call, `opt_core` picks the kernel it measured fastest for the GPU and input size. Any call it cannot serve (an unsupported GPU or shape, a missing prebuilt binary, a kernel error) runs on cuEquivariance, and without cuEquivariance on PyTorch, as before. A kernel error disables FlashPairformer for the rest of the process with a warning.

The route is limited to Boltz-2 evaluation with gradients disabled and CUDA bf16 autocast enabled. Boltz-1, training (including no-grad recycling passes), and the fp32 affinity module keep their existing kernel paths.

Only calls inside `Boltz2.forward` enter this scope automatically; calling the triangle layers directly keeps the existing kernels unless you explicitly enter `boltz.model.layers.flashpairformer.inference_scope()` under the same inference conditions.

Set `BOLTZ_FLASHPAIRFORMER` to choose the tier:

- `fast` (default): the fastest kernels; outputs differ from cuEquivariance by bf16 rounding.
- `exact`: only kernels that are bit-identical to cuEquivariance on a stack Anthropic measured; everything else runs cuEquivariance.
- `off`: never use `opt_core`.

Changes to `BOLTZ_FLASHPAIRFORMER` take effect on subsequent calls in the same process. After a kernel error has disabled the route, restart the process to enable it again; changing the tier does not clear that failure.

Normal PyTorch parameter updates invalidate packed triangle-multiplication weights automatically, including opt_core's shared native caches. For custom weight changes that bypass version tracking (such as writes through `.data`), call `boltz.model.layers.flashpairformer.clear_weight_caches()` before predicting again. Triangle multiplication with weights created inside `torch.inference_mode()` uses the existing fallback because those weights have no mutation counter.

`--no_kernels` turns off FlashPairformer along with cuEquivariance.

### Installing the bleeding-edge `main` branch

```
pip install "boltz-community @ git+https://github.com/Novel-Therapeutics/boltz-community.git"
```

### Apple Silicon (MPS)

On Macs with Apple Silicon, you can run inference on the GPU via MPS:

```
boltz predict input.yaml --accelerator mps --use_msa_server
```

MPS mode automatically uses float32 precision and single-device execution. Performance is slower than CUDA but significantly faster than CPU.

Several wheels (torch, scikit-learn) bundle their own copy of `libomp.dylib`, and
loading more than one OpenMP runtime into the same process crashes with an `OMP:
Error` about libomp being initialized twice. The common workaround
`KMP_DUPLICATE_LIB_OK=TRUE` silences the safety check but is officially unsafe
per libomp's own warning text. Instead, repoint the duplicate copies at a single
canonical libomp so only one is ever loaded:

```
boltz-fix-macos-libomp
```

Safe to re-run and to run again after upgrading torch or scikit-learn.

### Intel GPUs (XPU)

On Intel GPUs, install PyTorch's XPU build first, then Boltz, and run with `--accelerator xpu`:

```
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install boltz-community
boltz predict input.yaml --accelerator xpu --use_msa_server
```

XPU mode uses bf16 mixed precision, as on CUDA, and runs on a single device. To pick a card, set
`ZE_AFFINITY_MASK=<index>`, the Intel counterpart of `CUDA_VISIBLE_DEVICES`. With `--seed`,
deterministic algorithms are switched on (about 3% slower); reruns were byte-identical in testing, but
this is empirical for the operations Boltz uses today rather than guaranteed.
Tested on an Intel Arc Pro B70 with PyTorch 2.14.

## Releasing

Releases are tag-driven. Pushing a `v*.*.*` tag triggers [.github/workflows/release.yml](.github/workflows/release.yml), which builds the sdist + wheel, validates them with `twine check`, and publishes to PyPI via [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/) (OIDC) — no long-lived API token is stored in this repo.

### One-time setup (PyPI side)

1. Register the `boltz-community` package on PyPI (create as a new project; the name is currently available).
2. Open the project's *Publishing* settings and add a Trusted Publisher with:
   - **Owner:** `Novel-Therapeutics`
   - **Repository:** `boltz-community`
   - **Workflow filename:** `release.yml`
   - **Environment:** `pypi`
3. In the GitHub repo Settings → Environments, create an environment named `pypi`. Optionally add a required-reviewer rule so a publish has to be approved by a maintainer before the OIDC exchange runs.

### Per-release flow

1. Land all changes on `main`.
2. Bump `version` in [pyproject.toml](pyproject.toml).
3. Add a bullet under *Bug fixes* / *Improvements* / *Performance improvements* in this README. Update the test count if new tests landed.
4. Commit as `Release X.Y.Z` and push `main`.
5. Tag the release commit: `git tag -a vX.Y.Z -m "Release X.Y.Z" && git push origin vX.Y.Z`.
6. The release workflow builds, checks, and publishes the artifacts to PyPI. Watch the run under the *Actions* tab.
7. Create the GitHub release with the release notes (see prior releases for the *Highlights / Details / Commits since* structure).

### Recovering from a failed publish

PyPI rejects re-uploading the same version. If the build succeeded but the publish step failed (e.g. environment approval timed out), you can re-run the *Publish to PyPI* job from the Actions tab without re-tagging. If a version is published but broken, **bump the patch and ship a new release** — never delete or yank without a follow-up that supersedes it.

---

*Everything below is from the upstream Boltz README.*

---

<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://boltz.bio/join-slack) <br> <br>
</div>



![](docs/boltz1_pred_figure.png)


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://boltz.bio/join-slack).

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. Its value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log10(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md#authentication-to-msa-server).
 
## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://boltz.bio/join-slack) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels, and by Anthropic's FlashPairformer kernels when [`opt_core`](https://github.com/anthropics/uplifting-biomolecular-modeling) is installed. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```
