"""The integrated Anthropic CLI, using the fork launchers."""

import sys


def main(argv=None):
    from .runtime import adapt_engine, configure

    try:
        _, _, detail = configure()
        cli = adapt_engine(detail)
    except RuntimeError as error:
        print(f"[boltz2-opt] NOT ACTIVE: {error}", file=sys.stderr)
        return 3
    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
