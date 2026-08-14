"""
Command-line entry point: `python -m strux <checkpoint>` describes a saved
npz/safetensors file (see strux.describe).
"""

import sys

from strux.inspector import describe


def main(argv):
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print("usage: python -m strux <checkpoint.npz | checkpoint.safetensors>")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    print(describe(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
