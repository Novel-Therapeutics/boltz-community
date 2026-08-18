# boltz-community

Community-maintained fork of [Boltz](https://github.com/jwohlwend/boltz) with bug fixes, broader compatibility, and CI.

## What's different from upstream?

**Compatibility:**
- Apple Silicon (MPS) support: `boltz predict --accelerator mps`
- Added `boltz-fix-macos-libomp` CLI to fix `OMP: Error #15` on macOS caused by multiple wheels (torch, scikit-learn) each bundling their own `libomp.dylib` — repoints the duplicates at a single canonical libomp via `install_name_tool` so only one OpenMP runtime is ever loaded, instead of silencing the safety check with `KMP_DUPLICATE_LIB_OK=TRUE` (which libomp's own docs warn is unsafe). Thanks to [@fnachon](https://github.com/Novel-Therapeutics/boltz-community/pull/15).
- Dependency pins relaxed from `==` to `>=`
- `fairscale` dependency removed — replaced with PyTorch built-in `torch.utils.checkpoint`
- `numpy<2.0` cap removed
- `requires-python` widened to `>=3.10` (removed `<3.13` cap)
- Compatible with PyTorch 2.6+ and Lightning 2.6+

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
- Fixed silent wrong-answer bug: inference `__getitem__` no longer substitutes a different record on failure — errors now propagate
- Fixed potential stack overflow in training/validation data loading via bounded retry (max 10 attempts)
- Fixed `boltz predict` exiting silently with code 0 when all inputs fail validation (e.g. requesting affinity for a protein chain)
- Fixed MSA pairing keys lost when loading cached A3M files ([#627](https://github.com/jwohlwend/boltz/issues/627))
- Fixed affinity prediction crashing when structure prediction fails (e.g. covalent ligands, OOM) — now skips affected records with a warning ([#620](https://github.com/jwohlwend/boltz/issues/620), [#624](https://github.com/jwohlwend/boltz/issues/624))
- Fixed affinity prediction for repeated ligand binders: inputs that request affinity for one copy of a repeated ligand entity no longer fail, and affinity is now reported per binder copy as `affinity_<record>_<chain>.json` ([#647](https://github.com/jwohlwend/boltz/issues/647))
- Fixed Boltz-2 checkpoint loading crash due to extra `mse_rotational_alignment` kwarg ([#644](https://github.com/jwohlwend/boltz/issues/644))
- Fixed direct Boltz-2 model use from checkpoint hyperparameters crashing during sampling when `steering_args` is missing ([#680](https://github.com/jwohlwend/boltz/issues/680))
- Fixed empty checkpoint files causing cryptic `load_from_checkpoint` aborts — now re-downloads empty cached weights and raises a clear error before model load ([#664](https://github.com/jwohlwend/boltz/issues/664))
- Fixed CPU inference producing distorted structures with wrong bond lengths — Boltz-2 was incorrectly using `bf16-mixed` precision on CPU; now forces float32 ([#653](https://github.com/jwohlwend/boltz/issues/653))
- Fixed missing PAE summaries in `confidence_*.json` — aggregated `complex_pae`, `complex_ipae`, `chains_pae`, and `pair_chains_pae` are now included, remain available with `--no_write_full_pae`, use contact-weighted aggregation like PDE, and write undefined values as `null` ([#607](https://github.com/jwohlwend/boltz/issues/607))
- Fixed cuequivariance triangle multiplication kernel crashes by falling back to PyTorch when kernels are unavailable or unsupported ([#485](https://github.com/jwohlwend/boltz/issues/485), with thanks to [#682](https://github.com/jwohlwend/boltz/pull/682))
- Fixed cuequivariance triangle attention kernel crashes by falling back to PyTorch when kernels are unavailable or unsupported — extends the v2.10.3 fix to the triangle attention path that was missed ([#485](https://github.com/jwohlwend/boltz/issues/485))
- Fixed forced contact restraints losing all signal when union weighting underflowed for large distance violations ([#621](https://github.com/jwohlwend/boltz/issues/621), with thanks to [#682](https://github.com/jwohlwend/boltz/pull/682))
- Fixed incorrect contact-union gradient sign so soft-OR contact restraints apply gradient pressure with the correct magnitude across union members
- Fixed diffusion sampling ignoring `--max_parallel_samples` in divisible cases (for example `10` samples with a parallel limit of `5`), which could batch everything into one large chunk and trigger avoidable OOMs
- Fixed MSA discarded as "does not match input sequence" when pre-computed MSAs are aligned to a full UniProt sequence but the input uses a shorter PDB construct — Boltz now finds the construct as a contiguous subsequence within the MSA query and trims all MSA rows accordingly, instead of falling back to a dummy single-sequence MSA. Tolerates up to 5% mismatches (selenomethionine substitutions, expression tags, minor construct mutations). Applies to both Boltz-1 and Boltz-2.
- Fixed PDB templates crashing with `IndexError: list index out of range` when Gemmi drops entity sequence metadata during PDB→mmCIF conversion — template parsing now falls back to the observed polymer residues, and relative template `pdb`/`cif` paths are resolved from the YAML file directory instead of the current working directory ([#669](https://github.com/jwohlwend/boltz/issues/669))
- Fixed YAML `bond` constraints for custom cross-residue covalent bonds (for example ACE-CY3 cyclization) missing atom-level bond-length bounds for physical guidance and `_struct_conn` records in mmCIF output ([#675](https://github.com/jwohlwend/boltz/issues/675))
- Fixed Boltz-2 fine-tuning/validation crashes when downstream methods read `self.validate_structure`; the constructor now stores the `validate_structure` argument on the model ([novel-therapeutics/boltz-community#11](https://github.com/Novel-Therapeutics/boltz-community/issues/11))
- Fixed training-data preprocessing: `scripts/process/rcsb.py` was looking up clusters by `pdb_id_entity_id` while `scripts/process/cluster.py` keys its output by `hash_sequence(seq)` (proteins/RNA/short polymers) or by CCD code (ligands), so every chain silently got `cluster_id=-1` and `ClusterSampler` weighted everything uniformly. Records also had `msa_id=""` for protein chains, so training silently ran with no MSA features. Both fields are now populated correctly, and `entity_id` is propagated through to the record ([#686](https://github.com/jwohlwend/boltz/issues/686)). Does not affect inference; users who trained or fine-tuned with the documented pipeline need to re-preprocess and re-train.
- Fixed affinity prediction running 5× slower than necessary: upstream hardcoded `max_parallel_samples=1` for the affinity diffusion path, which was silently masked by upstream's buggy `chunk(multiplicity % max + 1)` math (the divisible case collapsed to `chunk(1)`, batching all samples in one pass). When our earlier diffusion fix replaced that with the correct `split(max_parallel_samples)`, the hardcoded `1` started to actually take effect, forcing N sequential single-sample forward passes per affinity record. The affinity path now honors the user's `--max_parallel_samples`, capped at `--diffusion_samples_affinity` so it doesn't claim more parallelism than diffusion will run.
- Fixed uncatchable RDKit abort on salt-form ligand SMILES — `standardize()` now primes the implicit-valence cache with `UpdatePropertyCache(strict=False)` before `LargestFragmentChooser`, so a single bad SMILES (e.g. salts containing a terminal alkyne such as `C#CCN(C)CC(=C)c1ccc(Cl)c(Cl)c1.O=C(O)C(=O)O`) no longer takes down the entire affinity batch via a C++ pre-condition violation that Python `try/except` can't catch
- Bumped MSA submit POST and result-download timeouts from 6.02s to 300s (status polling stays at 6.02s) — eliminates spurious timeouts on `api.colabfold.com` requests for larger sequences or slower networks ([#688](https://github.com/jwohlwend/boltz/pull/688))
- Fixed transient MSA download responses being cached as `out.tar.gz` and later failing with a cryptic `tarfile.ReadError` — result downloads now require a successful HTTP response and a readable archive containing the expected A3M files before they are saved

**Improvements:**
- Published to PyPI as [`boltz-community`](https://pypi.org/project/boltz-community/) — `pip install boltz-community` and `uv add boltz-community` now work without the git URL ([#12](https://github.com/Novel-Therapeutics/boltz-community/issues/12)). Releases are tag-driven via GitHub Actions + PyPI Trusted Publisher (OIDC, no long-lived tokens).
- Added `--skip_bad_inputs` flag: by default `boltz predict` now aborts when any input fails processing; pass `--skip_bad_inputs` to skip bad inputs and continue with the rest
- Deferred heavy imports (torch, rdkit, pytorch-lightning) so `boltz.main` loads instantly for CLI help and input validation
- `--devices` now accepts a comma-separated list of specific GPU device IDs in addition to a device count (e.g. `--devices 0,1` targets GPUs 0 and 1; use `CUDA_VISIBLE_DEVICES=1 boltz predict ...` to target a single GPU by index)
- Added `--batch_size` for Boltz-2 structure inference so multiple inputs can be processed per prediction batch. Current limits: affinity prediction remains single-record (`batch_size=1`), and guided inference (`--use_potentials` / contact guidance) is only supported with `--batch_size 1`

**Performance improvements:**
- Added optional FlashAttention-2 / PyTorch SDPA acceleration for triangle attention and pair-biased attention via `--flash_attn` (off by default). Reduces attention memory footprint and speeds up inference on Ampere+ GPUs while remaining numerically equivalent to the manual einsum path within float-precision tolerance (verified by parity tests)
- Inference checkpoints are deserialized on CPU and then moved to the accelerator once by Lightning, instead of being deserialized straight onto the accelerator. Lightning builds the module on CPU, so a device-resident checkpoint gets copied back to CPU one tensor at a time before the finished module is moved to the device again; staging on CPU skips that round trip. Measured on a Tesla T4: **~1.75s faster** warm-cache load, and **~4.4 GiB less peak VRAM** during loading (4,484 MiB reserved → 0), at a cost of **~1.0 GiB more peak host RAM** (4.1 → 5.1 GiB). Cold loads are disk-bound and change little (~0.8s). The memory trade favours the accelerator, which is usually the binding constraint. Predictions are byte-identical (verified by fixed-seed mmCIF diff). (Distinct from the module-level tensor allocation below, which concerns tensors created during the forward pass, not checkpoint loading.) Thanks to [@corinwagen](https://github.com/Novel-Therapeutics/boltz-community/pull/16).
- Cached molecule file reads and symmetry deserialization across samples
- Removed dead O(n_tokens × n_chains) loop in pocket distance computation
- Tensors across model modules now allocated directly on device instead of CPU-then-transfer ([#654](https://github.com/jwohlwend/boltz/pull/654))
- Featurizer MSA pairing fill rewritten with vectorized numpy indexing (eliminates per-row Python loop)
- `process_atom_features` pre-allocates output arrays and fills `atom_to_token` in one slice per token (eliminates per-atom appends)

**Tests & CI:**
- 284 tests in this fork vs. 5 tests in current upstream `jwohlwend/boltz`: unit tests (CPU), smoke tests (end-to-end inference), regression tests (golden output verification for Boltz-1 and Boltz-2), determinism tests, MSA trim subsequence matching (8 cases), diffusion chunking regression tests, Boltz-2 validation constructor coverage, and featurizer pre-allocation correctness
- GitHub Actions CI with CPU runners (every push/PR) and GPU T4 runners (push to main)

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

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

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
