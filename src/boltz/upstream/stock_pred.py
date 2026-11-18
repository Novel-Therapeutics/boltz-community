"""Clean off-profile subprocess: fork IO with the same reference numerical core."""

import importlib
import json
import os
import sys
from pathlib import Path


def stock_failures(directory: Path, phase: str) -> set[str]:
    """Read the named skips recorded by every writer process in one phase."""
    return {
        json.loads(line)
        for path in directory.glob(f"{phase}-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    }


def track_failures(cli, directory: Path, error_type) -> None:
    """Keep the reference phases running and record failed units for the parent."""

    def track(writer, phase):
        write_batch = writer.write_on_batch_end
        end_phase = writer.on_predict_epoch_end

        def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx,
        ):
            write_batch(
                self,
                trainer,
                pl_module,
                prediction,
                batch_indices,
                batch,
                batch_idx,
                dataloader_idx,
            )
            if prediction["exception"]:
                path = directory / f"{phase}-{os.getpid()}.jsonl"
                with path.open("a", encoding="utf-8") as stream:
                    for record in batch["record"]:
                        stream.write(json.dumps(record.id) + "\n")

        def on_predict_epoch_end(self, trainer, model):
            try:
                end_phase(self, trainer, model)
            except error_type:
                # Native inference raises after the phase; the engine owns
                # per-record accounting and continues with successful units.
                if not stock_failures(directory, phase):
                    raise

        writer.write_on_batch_end = write_on_batch_end
        writer.on_predict_epoch_end = on_predict_epoch_end

    track(cli.BoltzWriter, "structure")
    track(cli.BoltzAffinityWriter, "affinity")
    filter_affinity = cli.filter_inputs_affinity

    def filter_inputs_affinity(manifest, outdir, override=False):
        failed = stock_failures(directory, "structure")
        eligible = cli.Manifest([r for r in manifest.records if r.id not in failed])
        return filter_affinity(eligible, outdir, override)

    cli.filter_inputs_affinity = filter_inputs_affinity


def main(argv=None):
    from .runtime import configure

    _, stock, detail = configure()
    from boltz2_opt import kernels, stock_pred

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        return 2
    split = argv.index("--")
    kopts, remaining = kernels.parse_cli_opts(argv[:split])
    options = dict(zip(remaining[::2], remaining[1::2]))
    # Keep the upstream clean-process/core proof, replacing only its stock-tree
    # assertion with our separately verified compatible-fork source identity.
    proof = stock_pred.env_proof(
        os.environ,
        list(filter(None, options.get("--env-absent", "").split(","))),
        list(filter(None, options.get("--kit-modules", "").split(","))),
        list(filter(None, options.get("--kit-dirs", "").split(","))),
        pins=None,
    )
    if not proof["ok"]:
        print("[boltz2-opt] NOT ACTIVE: " + "; ".join(proof["problems"]), flush=True)
        return stock_pred.EXIT_NOT_STOCK
    print(
        f"[boltz2-opt] verified fork {detail['version']} with pinned upstream numerical core "
        f"source_contract_sha256={detail['source_contract_sha256']}",
        flush=True,
    )
    if options.get("--proof-only") == "1":
        return 0
    from .bootstrap import install

    install(stock)
    cli = importlib.import_module(stock_pred.STOCK_ENTRY[0])
    entry = getattr(cli, stock_pred.STOCK_ENTRY[1])
    from click import ClickException

    from boltz.data.write.writer import PredictionError
    from boltz2_opt import phase

    if directory := options.get("--prediction-failure-dir"):
        track_failures(cli, Path(directory), PredictionError)
    phase.install()
    if kopts:
        kernels.install(mode="stock", **kopts)
    try:
        entry.main(args=argv[split + 1 :], prog_name="boltz", standalone_mode=True)
    except PredictionError as error:
        ClickException(str(error)).show()
        return 1
    except SystemExit as error:
        return int(error.code or 0) if not isinstance(error.code, str) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
