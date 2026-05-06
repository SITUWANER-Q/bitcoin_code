from __future__ import annotations

import argparse

from src.config import FROZEN_DIR
from src.utils.io import read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1.0.0")
    args = parser.parse_args()

    root = FROZEN_DIR / args.version
    manifest = read_json(root / "MANIFEST.json")
    expected = manifest["sha256"]
    for file_name, digest in expected.items():
        path = root / file_name
        got = sha256_file(path)
        if got != digest:
            raise RuntimeError(f"checksum mismatch for {file_name}: {got} != {digest}")
    print(f"[verify_data] checksums valid for {args.version}")


if __name__ == "__main__":
    main()

