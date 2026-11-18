"""The integrated Anthropic CLI, with original flags and profile semantics."""

import sys


def main(argv=None):
    from boltz.upstream.__main__ import main as integrated_main

    return integrated_main(argv)


if __name__ == "__main__":
    sys.exit(main())
