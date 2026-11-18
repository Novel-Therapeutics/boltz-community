"""Clean off-profile subprocess: fork IO with the same reference numerical core."""

import importlib
import os
import sys


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
        f"[boltz2-opt] verified fork {detail['version']} with pinned upstream numerical core",
        flush=True,
    )
    if options.get("--proof-only") == "1":
        return 0
    from .bootstrap import install

    install(stock)
    entry = getattr(
        importlib.import_module(stock_pred.STOCK_ENTRY[0]), stock_pred.STOCK_ENTRY[1]
    )
    from click import ClickException

    from boltz.data.write.writer import PredictionError
    from boltz2_opt import phase

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
