# Modified by boltz-community for packaged engine and runtime compatibility.
# Changes are recorded in boltz/upstream/contract.json and docs/upstream-compatibility.md.
# Original Anthropic code is Apache-2.0; bundled LICENSE/NOTICE and third-party notices apply.
"""The integrated Anthropic CLI, with original flags and profile semantics."""

import sys


def main(argv=None):
    from boltz.upstream.__main__ import main as integrated_main

    return integrated_main(argv)


if __name__ == "__main__":
    sys.exit(main())
