"""Process-local import overlay; no torch import before the worker's zygote."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from .contract import contract


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _install_waste_proof(module, stock):
    from .contract import digest

    def check_pin():
        # The original gate walks boltz.__file__, which is the fork.
        # Check the actual loaded reference files, with the exact same
        # hashes, instead of pretending that the fork is stock.
        for relative, wanted in module.PINNED_SHA256.items():
            name = relative[:-3].replace("/", ".")
            loaded = __import__(name, fromlist=["__file__"])
            actual = Path(loaded.__file__)
            if (
                actual.resolve() != (stock / relative).resolve()
                or digest(actual) != wanted
            ):
                raise RuntimeError(f"reference numerical source mismatch: {relative}")

    module._check_pin = check_pin


class _Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, stock):
        super().__init__(name, path)
        self.stock = stock

    def exec_module(self, module):
        super().exec_module(module)
        if module.__name__ == "boltz.main":
            # Reuse the fork's affinity completeness check inside the
            # reference CLI and the standalone affinity leg.
            fork_cli = _load_source(
                "boltz.upstream._fork_cli", Path(__file__).parents[1] / "main.py"
            )

            def filter_inputs_affinity(manifest, outdir, override=False):
                requests = module.Manifest([r for r in manifest.records if r.affinity])
                return fork_cli.filter_inputs_affinity(requests, outdir, override)

            module.filter_inputs_affinity = filter_inputs_affinity
        if module.__name__ == "boltz2_opt.affinity_leg":
            # Both the one-shot leg and the served affinity process must enter
            # the same reference core, without installing structure levers.
            for field in ("command", "server_command"):
                original = getattr(module, field)

                def command(*args, _original=original, **kwargs):
                    argv = _original(*args, **kwargs)
                    index = argv.index("boltz2_opt.affinity_leg")
                    argv[index] = "boltz.upstream.worker_launch"
                    argv.insert(index + 1, "--reference-affinity")
                    return argv

                setattr(module, field, command)
        if module.__name__ == "boltz2_opt.waste":
            # This adapter uses spec_from_file_location, bypassing meta finders.
            # Preserve its loader; adapt its proof after the verified file loads.
            original_load = module._load

            def load():
                provider = original_load()
                _install_waste_proof(provider, self.stock)
                return provider

            module._load = load
        if module.__name__ == "boltz_waste_levers":
            _install_waste_proof(module, self.stock)
        if module.__name__ == "boltz.data.feature.featurizerv2":
            reference = _load_source(
                "boltz.upstream._reference_features",
                self.stock / "boltz/data/feature/featurizerv2.py",
            )
            module.construct_paired_msa = reference.construct_paired_msa
            # The reference potentials do not understand the fork's new
            # guidance-only priors or intra-chain connection ranges.
            module.process_residue_constraint_features = (
                reference.process_residue_constraint_features
            )

            def process_chain_feature_constraints(data, ref_element=None):
                return reference.process_chain_feature_constraints(data)

            module.process_chain_feature_constraints = process_chain_feature_constraints
        if module.__name__ == "boltz.model.models.boltz2":
            # Retain the fork class/constructor/IO, but give upstream's patchers
            # the exact original forward body and its original global namespace.
            reference = _load_source(
                "boltz.upstream._reference_model",
                self.stock / "boltz/model/models/boltz2.py",
            )
            module.Boltz2.forward = reference.Boltz2.forward
            module.Boltz2.forward.__module__ = module.__name__
            original_predict = module.Boltz2.predict_step

            def predict_step(self, batch, batch_idx, dataloader_idx=0):
                result = original_predict(self, batch, batch_idx, dataloader_idx)
                if result is not None and not result.get("exception", False):
                    # Reporting only: does not select poses or change scores.
                    result.update(
                        module.check_structures(
                            result["coords"].detach(),
                            batch,
                            self.predict_args["diffusion_samples"],
                        )
                    )
                return result

            module.Boltz2.predict_step = predict_step


class _CoreFinder(importlib.abc.MetaPathFinder):
    def __init__(self, stock: Path):
        self.stock = stock
        self.modules = set(contract()["numeric_modules"])

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            source = self.stock / (fullname.replace(".", "/") + ".py")
            if fullname == "boltz.main":
                loader = _Loader(fullname, str(source), self.stock)
                return importlib.util.spec_from_file_location(
                    fullname, source, loader=loader
                )
            return importlib.util.spec_from_file_location(fullname, source)
        if fullname in (
            "boltz.model.models.boltz2",
            "boltz.data.feature.featurizerv2",
            "boltz_waste_levers",
            "boltz2_opt.waste",
            "boltz2_opt.affinity_leg",
        ):
            # PathFinder avoids recursing into upstream's after-import hooks.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin:
                spec.loader = _Loader(fullname, spec.origin, self.stock)
            return spec
        return None


def install(stock: Path) -> None:
    conflicts = set(contract()["numeric_modules"]) | {"boltz.data.feature.featurizerv2"}
    if conflicts.intersection(sys.modules) or any(
        name.startswith("boltz.model.") for name in sys.modules
    ):
        raise RuntimeError(
            "upstream compatibility requires a fresh interpreter before model imports"
        )
    sys.meta_path.insert(0, _CoreFinder(stock))
