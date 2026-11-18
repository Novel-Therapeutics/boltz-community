"""Adapt launch/provenance only; original flags, profiles and math stay upstream."""

from __future__ import annotations

import os

from .contract import prepare


def configure(*, overlay=False):
    root, stock, detail = prepare()
    # Do not compose the fork's independent experimental kernels with a profile.
    for name in (
        "BOLTZ_FLASHPAIRFORMER",
        "BOLTZ_DENOISER_CACHE",
    ):
        os.environ[name] = "off"
    if overlay:
        from .bootstrap import install

        install(stock)
    return root, stock, detail


def adapt_engine(detail):
    import boltz2_opt
    from boltz2_opt._autoload import TAG, require_core
    from boltz2_opt._core_gate import gate

    gate(boltz2_opt.__file__, TAG)
    require_core()
    from boltz2_opt import cli, manifest, report, stack

    from .contract import contract, digest

    def pins_check(ckpt=False):
        problems = []
        result = dict(detail)
        if ckpt:
            from pathlib import Path

            path = Path(stack.cache_dir()) / "boltz2_conf.ckpt"
            expected = contract()["checkpoint_sha256"]
            if not path.is_file() or digest(path) != expected:
                problems.append(f"checkpoint mismatch: {path}")
        return problems, result

    original_worker_command = stack.worker_command
    original_stock_command = cli.stock_command
    original_build = manifest.build
    original_say = report.say

    def worker_command(*args, **kwargs):
        command = original_worker_command(*args, **kwargs)
        if "boltz2_opt.worker_launch" not in command:
            raise RuntimeError(
                "unsupported upstream worker launch without its attachment wrapper"
            )
        command[command.index("boltz2_opt.worker_launch")] = (
            "boltz.upstream.worker_launch"
        )
        return command

    def stock_command(*args, **kwargs):
        command, env = original_stock_command(*args, **kwargs)
        command[command.index("boltz2_opt.stock_pred")] = "boltz.upstream.stock_pred"
        return command, env

    def build(*args, **kwargs):
        result = original_build(*args, **kwargs)
        result["fork_compatibility"] = detail
        return result

    def say(line, *args, **kwargs):
        if "mode off: stock — the stock CLI in a clean subprocess" in line:
            line = report.not_active_line(
                "mode off: fork with pinned upstream numerical core; no optimization profile applied"
            )
        return original_say(line, *args, **kwargs)

    report.say = say
    stack.pins_check = pins_check
    stack.worker_command = worker_command
    cli.stock_command = stock_command
    manifest.build = build
    return cli
