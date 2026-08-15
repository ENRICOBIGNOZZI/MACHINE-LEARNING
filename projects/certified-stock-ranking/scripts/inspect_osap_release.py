"""Validate an Open Source Asset Pricing wide file and record its checksum."""

from __future__ import annotations

import argparse
from pathlib import Path

from rankcert.data.osap import write_osap_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/osap_release_manifest.json"),
    )
    arguments = parser.parse_args()
    manifest = write_osap_manifest(arguments.input, arguments.output)
    print(manifest)
