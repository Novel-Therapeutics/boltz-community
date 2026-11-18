"""Install the reference core before delegating to the original worker launcher."""

import sys


def main(argv=None):
    from .runtime import configure

    argv = list(sys.argv[1:] if argv is None else argv)
    configure(overlay=True)
    if argv and argv[0] == "--reference-affinity":
        from boltz2_opt.affinity_leg import main as affinity

        return affinity(argv[1:])
    from boltz2_opt.worker_launch import main as launch

    return launch(argv)


if __name__ == "__main__":
    sys.exit(main())
