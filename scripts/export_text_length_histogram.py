#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from wackywacky_analysis.config import load_config
from wackywacky_analysis.errors import WackyWackyError
from wackywacky_analysis.length_histogram import (
    DEFAULT_BINS,
    DEFAULT_MAXIMUM,
    DEFAULT_MINIMUM,
    export_character_histogram,
)
from wackywacky_analysis.progress import configure_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exporta o histograma de caracteres de B_clean_v2 sem reler os TSVs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot-id")
    parser.add_argument("--minimum", type=float, default=DEFAULT_MINIMUM)
    parser.add_argument("--maximum", type=float, default=DEFAULT_MAXIMUM)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    arguments = parser.parse_args()
    configure_logging()
    try:
        result = export_character_histogram(
            load_config(arguments.config),
            snapshot_id=arguments.snapshot_id,
            minimum=arguments.minimum,
            maximum=arguments.maximum,
            bins=arguments.bins,
        )
    except (WackyWackyError, OSError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
