# boltz-community

Community-maintained fork of [Boltz](https://github.com/jwohlwend/boltz) with bug fixes, broader compatibility, and CI.

## What's different from upstream?

**Compatibility:**
- Apple Silicon (MPS) support: `boltz predict --accelerator mps`
- Dependency pins relaxed from `==` to `>=`
- `fairscale` dependency removed — replaced with PyTorch built-in `torch.utils.checkpoint`
- `numpy<2.0` cap removed
- `requires-python` widened to `>=3.10` (removed `<3.13` cap)
- Compatible with PyTorch 2.6+ and Lightning 2.6+

**Bug fixes:**
- Cherry-picked community PRs: [#654](https://github.com/jwohlwend/boltz/pull/654), [#602](https://github.com/jwohlwend/boltz/pull/602), [#584](https://github.com/jwohlwend/boltz/pull/584), [#582](https://github.com/jwohlwend/boltz/pull/582), [#576](https://github.com/jwohlwend/boltz/pull/576), [#538](https://github.com/jwohlwend/boltz/pull/538), [#500](https://github.com/jwohlwend/boltz/pull/500), [#488](https://github.com/jwohlwend/boltz/pull/488), [#463](https://github.com/jwohlwend/boltz/pull/463), [#363](https://github.com/jwohlwend/boltz/pull/363)
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
- Fixed consecutive CA filter rejecting valid protein chains containing metal ions ([#576](https://github.com/jwohlwend/boltz/pull/576))
- Fixed template alignment forcing gapless matches, breaking templates with indels ([#538](https://github.com/jwohlwend/boltz/pull/538))
- Fixed relative MSA paths resolved from CWD instead of input file directory ([#500](https://github.com/jwohlwend/boltz/pull/500))
- Fixed affinity prediction crashing when structure prediction fails (e.g. covalent ligands, OOM) — now skips affected records with a warning ([#620](https://github.com/jwohlwend/boltz/issues/620), [#624](https://github.com/jwohlwend/boltz/issues/624))
- Fixed Boltz-2 checkpoint loading crash due to extra `mse_rotational_alignment` kwarg ([#644](https://github.com/jwohlwend/boltz/issues/644))
- Fixed CPU inference producing distorted structures with wrong bond lengths — Boltz-2 was incorrectly using `bf16-mixed` precision on CPU; now forces float32 ([#653](https://github.com/jwohlwend/boltz/issues/653))
- Fixed MSA discarded as "does not match input sequence" when pre-computed MSAs are aligned to a full UniProt sequence but the input uses a shorter PDB construct — Boltz now finds the construct as a contiguous subsequence within the MSA query and trims all MSA rows accordingly, instead of falling back to a dummy single-sequence MSA. Tolerates up to 5% mismatches (selenomethionine substitutions, expression tags, minor construct mutations). Applies to both Boltz-1 and Boltz-2.

**Improvements:**
- Added `--skip_bad_inputs` flag: by default `boltz predict` now aborts when any input fails processing; pass `--skip_bad_inputs` to skip bad inputs and continue with the rest
- Deferred heavy imports (torch, rdkit, pytorch-lightning) so `boltz.main` loads instantly for CLI help and input validation
- `--devices` now accepts a comma-separated list of specific GPU device IDs in addition to a device count (e.g. `--devices 0,1` targets GPUs 0 and 1; use `CUDA_VISIBLE_DEVICES=1 boltz predict ...` to target a single GPU by index)

**Performance (~5% faster inference on GPU):**
- Model weights now load directly to GPU instead of CPU-then-transfer
- Cached molecule file reads and symmetry deserialization across samples
- Removed dead O(n_tokens × n_chains) loop in pocket distance computation
- Tensors across model modules now allocated directly on device instead of CPU-then-transfer ([#654](https://github.com/jwohlwend/boltz/pull/654))
- Featurizer MSA pairing fill rewritten with vectorized numpy indexing (eliminates per-row Python loop)
- `process_atom_features` pre-allocates output arrays and fills `atom_to_token` in one slice per token (eliminates per-atom appends)

**Tests & CI:**
- 190+ tests: unit tests (CPU), smoke tests (end-to-end inference), regression tests (golden output verification for Boltz-1 and Boltz-2), determinism tests, MSA trim subsequence matching (8 cases), and featurizer pre-allocation correctness
- GitHub Actions CI with CPU runners (every push/PR) and GPU T4 runners (push to main)

## Contributing

Pull requests are welcome! If you have a bug fix, test improvement, or compatibility enhancement, please open a PR.

## Installation

Install from GitHub:

```
pip install "boltz-community @ git+https://github.com/Novel-Therapeutics/boltz-community.git"
```

With CUDA kernels:

```
pip install "boltz-community[cuda] @ git+https://github.com/Novel-Therapeutics/boltz-community.git"
```

If you are installing on CPU-only or non-CUDA GPU hardware, use the first command without `[cuda]`. Note that the CPU version is significantly slower than the GPU version.

### Apple Silicon (MPS)

On Macs with Apple Silicon, you can run inference on the GPU via MPS:

```
boltz predict input.yaml --accelerator mps --use_msa_server
```

MPS mode automatically uses float32 precision and single-device execution. Performance is slower than CUDA but significantly faster than CPU.

Early benchmark on an M3 MacBook Air (single data point — community testing welcome):

| Case | T4 GPU (bf16) | M3 Air MPS (f32) | Ratio |
|------|---------------|-------------------|-------|
| 10-res peptide + ligand (affinity) | 107s | 114s | 1.1x |
| 163-res protein + ligand (affinity) | 144s | 251s | 1.7x |
| 260-res protein + ligand (affinity) | 193s | 383s | 2.0x |
| 163-res protein (structure only) | 57s | 74s | 1.3x |

The gap grows with protein size, likely due to attention layers scaling quadratically without bf16. If you run benchmarks on other Apple Silicon chips, please share your results in an issue!

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
 
## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


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
