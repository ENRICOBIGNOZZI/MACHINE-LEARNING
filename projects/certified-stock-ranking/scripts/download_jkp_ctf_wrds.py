"""Download the prebuilt JKP Common Task Framework tables from WRDS."""

from __future__ import annotations

import argparse
from pathlib import Path

from rankcert.data.jkp import download_jkp_ctf_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/jkp_ctf"),
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    manifest = download_jkp_ctf_tables(
        arguments.output_dir,
        chunksize=arguments.chunksize,
        overwrite=arguments.overwrite,
    )
    print(arguments.output_dir)
    print(manifest)
