"""
run_pipeline.py — RiceEWS Central Execution Engine
====================================================

This is the single authoritative execution engine for RiceEWS.
All other runner scripts (run_all.py, run_module1.py, …) are thin
wrappers that delegate here via subprocess.

Usage examples
--------------
  python run_pipeline.py --all
  python run_pipeline.py --module 1
  python run_pipeline.py --module 2
  python run_pipeline.py --module 3
  python run_pipeline.py --all --input /path/to/raw --output /path/to/results
  python run_pipeline.py --module 2 --force
  python run_pipeline.py --all --duplicate-strategy keep_last --verbose
  python run_pipeline.py --all --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Path setup — allow importing from src/ without installing the package
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.console import (  # noqa: E402
    print_banner,
    print_discovered,
    print_execution_time,
    print_exporting,
    print_generating_leaders,
    print_initializing,
    print_loading_datasets,
    print_pipeline_success,
    print_prerequisite_error,
    print_recovering_dag,
    print_running_module,
    print_section,
    print_summary,
    print_validating_schema,
    info,
    warn,
    error,
    success,
    SEPARATOR,
)

# ──────────────────────────────────────────────────────────────────────────────
# Default paths (relative to repo root)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_MODULE1_OUTPUT = DEFAULT_PROCESSED_DIR / "module_01" / "preprocessed_pilot_data.csv"
DEFAULT_MODULE2_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "module_02"
DEFAULT_MODULE2_JSON = DEFAULT_MODULE2_OUTPUT_DIR / "granger_results.json"
DEFAULT_MODULE3_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "module_03"


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "RiceEWS — Rice Price Early Warning System\n"
            "Central execution engine.  Run the full pipeline or an individual module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python run_pipeline.py --all
              python run_pipeline.py --module 1
              python run_pipeline.py --module 2 --force
              python run_pipeline.py --all --duplicate-strategy keep_last
              python run_pipeline.py --all --input data/raw --output data/processed
            """
        ),
    )

    # Mutually exclusive: --all or --module N
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--all",
        action="store_true",
        help="Execute the complete pipeline (Module 1 → Module 2 → Module 3).",
    )
    scope.add_argument(
        "--module",
        type=int,
        choices=[1, 2, 3],
        metavar="{1,2,3}",
        help="Execute a single module (1, 2, or 3).",
    )

    parser.add_argument(
        "--input",
        metavar="PATH",
        default=None,
        help=(
            "Path to the raw input directory (default: data/raw).  "
            "All *.xlsx files in this directory are discovered automatically."
        ),
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Base path for processed outputs (default: data/processed).  "
            "Module-specific sub-directories are created automatically."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a module even if its output already exists.",
    )
    parser.add_argument(
        "--duplicate-strategy",
        dest="duplicate_strategy",
        choices=["keep_first", "keep_last", "error"],
        default="error",
        help=(
            "Strategy when duplicate (date, market, grade) rows are found in "
            "the raw dataset.  Choices: keep_first, keep_last, error (default: error)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from scientific modules.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging (implies --verbose).",
    )
    return parser




# ──────────────────────────────────────────────────────────────────────────────
# Module runners
# ──────────────────────────────────────────────────────────────────────────────


def _run_module1(args: argparse.Namespace, raw_dir: Path, processed_dir: Path) -> Path:
    """Execute Module 1: Data Preprocessing."""

    from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline

    output_file = processed_dir / "module_01" / "preprocessed_pilot_data.csv"

    if output_file.exists() and not args.force:
        info(f"Module 1 output already exists: {output_file}")
        info("Use --force to re-run.  Skipping.")
        return output_file

    # Discover input datasets
    xlsx_files = sorted(raw_dir.glob("*.xlsx"))
    if not xlsx_files:
        error(f"No Excel datasets found in: {raw_dir}")
        sys.exit(1)

    print_loading_datasets()
    print_discovered(len(xlsx_files))
    for f in xlsx_files:
        info(f"  Dataset: {f.name}")
    print_validating_schema()

    print_running_module(1, "Data Preprocessing")

    config = {
        "detrend_method": "none",
        "differencing_order": 1,
        "duplicate_strategy": args.duplicate_strategy,
    }

    # Pass None so the module auto-discovers from raw_dir, or pass explicit list
    input_arg = [str(f) for f in xlsx_files]

    run_full_preprocessing_pipeline(
        input_file=input_arg,
        output_file=str(output_file),
        config=config,
    )

    print_exporting()
    success(f"Module 1 output saved to: {output_file}")
    return output_file


def _run_module2(args: argparse.Namespace, module1_csv: Path, processed_dir: Path) -> Path:
    """Execute Module 2: Pairwise Granger Analysis."""

    from modules.causality_testing.granger_tester import run_full_granger_analysis

    output_dir = processed_dir / "module_02"
    output_json = output_dir / "granger_results.json"

    if output_json.exists() and not args.force:
        info(f"Module 2 output already exists: {output_json}")
        info("Use --force to re-run.  Skipping.")
        return output_json

    if not module1_csv.exists():
        print_prerequisite_error(2, str(module1_csv))
        sys.exit(1)

    print_running_module(2, "Pairwise Granger Analysis")

    config = {
        "lag_order": 4,
        "price_col": "price_diff",
        "significance_level": 0.05,
        "apply_fdr": True,
    }

    run_full_granger_analysis(
        input_file=str(module1_csv),
        output_dir=str(output_dir),
        config=config,
    )

    print_exporting()
    success(f"Module 2 output saved to: {output_dir}")
    return output_json


def _run_module3(args: argparse.Namespace, module2_json: Path, processed_dir: Path) -> Path:
    """Execute Module 3: Network Recovery."""

    from modules.module_03_network_inference.network_builder import run_module3_network_inference

    output_dir = processed_dir / "module_03"
    output_json = output_dir / "network_inference_results.json"

    if output_json.exists() and not args.force:
        info(f"Module 3 output already exists: {output_json}")
        info("Use --force to re-run.  Skipping.")
        return output_json

    if not module2_json.exists():
        print_prerequisite_error(3, str(module2_json))
        sys.exit(1)

    print_running_module(3, "Network Recovery")
    print_recovering_dag()
    print_generating_leaders()

    run_module3_network_inference(
        granger_json_path=str(module2_json),
        output_dir=str(output_dir),
    )

    print_exporting()
    success(f"Module 3 output saved to: {output_dir}")
    return output_json


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print_banner()
    print_initializing()

    # Resolve paths
    raw_dir = Path(args.input) if args.input else DEFAULT_RAW_DIR
    processed_dir = Path(args.output) if args.output else DEFAULT_PROCESSED_DIR

    info(f"Raw data directory  : {raw_dir}")
    info(f"Output directory    : {processed_dir}")
    info(f"Duplicate strategy  : {args.duplicate_strategy}")
    if args.force:
        warn("--force flag is set.  Existing outputs will be overwritten.")

    start_time = time.monotonic()

    try:
        if args.all:
            module1_csv = _run_module1(args, raw_dir, processed_dir)
            module2_json = _run_module2(args, module1_csv, processed_dir)
            _run_module3(args, module2_json, processed_dir)

        elif args.module == 1:
            _run_module1(args, raw_dir, processed_dir)

        elif args.module == 2:
            module1_csv = processed_dir / "module_01" / "preprocessed_pilot_data.csv"
            _run_module2(args, module1_csv, processed_dir)

        elif args.module == 3:
            module2_json = processed_dir / "module_02" / "granger_results.json"
            _run_module3(args, module2_json, processed_dir)

    except KeyboardInterrupt:
        print()
        error("Pipeline interrupted by user.")
        return 130
    except Exception as exc:
        error(f"Pipeline failed: {exc}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.monotonic() - start_time

    print_pipeline_success()
    print_execution_time(elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
