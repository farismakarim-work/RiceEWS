"""
run_pipeline.py — RiceEWS Central Execution Engine
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    ALLOW_DATE_FILTER,
    ALLOW_GRADE_FILTER,
    ALLOW_MARKET_FILTER,
    CLI_DEFAULT_DUPLICATE_STRATEGY,
    CLI_DEFAULT_INPUT_DIR,
    CLI_DEFAULT_OUTPUT_DIR,
    DEFAULT_DATE_RANGE,
    DEFAULT_GRADES,
    DEFAULT_MARKETS,
    LOG_DATEFMT,
    LOG_FORMAT,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_DEFAULT,
    LOG_LEVEL_VERBOSE,
    M1_DETREND_METHOD,
    M1_DIFFERENCING_MODE,
    M1_DUPLICATE_STRATEGY_OPTIONS,
    M1_LOG_TRANSFORM,
    M1_MANUAL_DIFFERENCING_ORDER,
    M1_MAX_DIFFERENCING_ORDER,
    M1_MISSING_VALUE_MODE,
    M1_OUTLIER_MODE,
    M1_OUTLIER_THRESHOLD,
    M1_REQUIRE_STATIONARITY,
    M1_STANDARDIZATION_ENABLED,
    M1_STANDARDIZATION_METHOD,
    M1_STATIONARITY_SIGNIFICANCE_LEVEL,
    M1_STATIONARITY_TEST,
    M2_APPLY_BH_CORRECTION,
    M2_AUTO_SIGNIFICANCE_LEVEL,
    M2_LAG_CRITERION,
    M2_LAG_ORDER,
    M2_LAG_SELECTION_MODE,
    M2_MAX_LAG,
    M2_PRICE_COL,
    M2_SIGNIFICANCE_LEVEL,
    M2_SIGNIFICANCE_MODE,
    M3_ENFORCE_DAG,
    MODULE_01_DIRNAME,
    MODULE_01_OUTPUT_FILENAME,
    MODULE_02_DIRNAME,
    MODULE_02_MAIN_OUTPUT_FILENAME,
    MODULE_03_DIRNAME,
    MODULE_03_MAIN_OUTPUT_FILENAME,
    RAW_INPUT_FILE_PATTERN,
    SKIP_EXISTING_OUTPUTS,
)
from utils.console import (  # noqa: E402
    error,
    info,
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
    success,
    warn,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "RiceEWS — Rice Price Early Warning System\n"
            "Central execution engine. Run the full pipeline or an individual module."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python run_pipeline.py --all
              python run_pipeline.py --module 1
              python run_pipeline.py --all --grade low1 --market 101-106
              python run_pipeline.py --module 2 --market 101,103 --date-range 2021-01-01:2023-12-31
            """
        ),
    )

    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="Execute Module 1 → Module 2 → Module 3.")
    scope.add_argument("--module", type=int, choices=[1, 2, 3], metavar="{1,2,3}", help="Execute one module.")

    parser.add_argument("--input", metavar="PATH", default=None, help="Raw input directory.")
    parser.add_argument("--output", metavar="PATH", default=None, help="Processed output base directory.")
    parser.add_argument("--force", action="store_true", help="Compatibility flag; forces rerun when skip mode is enabled.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip execution when expected output exists (disabled by default).",
    )
    parser.add_argument(
        "--duplicate-strategy",
        dest="duplicate_strategy",
        choices=list(M1_DUPLICATE_STRATEGY_OPTIONS),
        default=CLI_DEFAULT_DUPLICATE_STRATEGY,
        help="Duplicate handling strategy for raw data.",
    )

    parser.add_argument("--grade", action="append", default=None, help="Grade filter. Can be repeated or comma-separated.")
    parser.add_argument("--market", default=None, help="Market filter: single (101), range (101-120), or CSV (101,103).")
    parser.add_argument("--date-range", dest="date_range", default=None, help="Date range filter: YYYY-MM-DD:YYYY-MM-DD")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    return parser


def _parse_grade_filter(raw_values: Optional[List[str]]) -> Optional[List[str]]:
    values = raw_values if raw_values is not None else DEFAULT_GRADES
    if not values:
        return None

    grades: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                grades.append(token)

    return sorted(set(grades)) if grades else None


def _parse_market_filter(raw_value: Optional[str]) -> Optional[List[int]]:
    value = raw_value if raw_value is not None else DEFAULT_MARKETS
    if value in (None, "", []):
        return None

    if isinstance(value, list):
        return sorted({int(v) for v in value})

    markets: Set[int] = set()
    for chunk in str(value).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"market range start must be <= end: {token}")
            for market in range(start, end + 1):
                markets.add(market)
        else:
            markets.add(int(token))

    return sorted(markets) if markets else None


def _parse_date_range(raw_value: Optional[str]) -> Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]:
    value = raw_value if raw_value is not None else DEFAULT_DATE_RANGE
    if not value:
        return None

    if ":" not in str(value):
        raise ValueError("--date-range must use format YYYY-MM-DD:YYYY-MM-DD")

    start_raw, end_raw = str(value).split(":", 1)
    start = pd.Timestamp(datetime.strptime(start_raw, "%Y-%m-%d")) if start_raw else None
    end = pd.Timestamp(datetime.strptime(end_raw, "%Y-%m-%d")) if end_raw else None
    if start and end and start > end:
        raise ValueError("date-range start must be <= end")
    return start, end


def _build_subset_filters(args: argparse.Namespace) -> dict:
    filters = {
        "grades": _parse_grade_filter(args.grade) if ALLOW_GRADE_FILTER else None,
        "markets": _parse_market_filter(args.market) if ALLOW_MARKET_FILTER else None,
        "date_range": _parse_date_range(args.date_range) if ALLOW_DATE_FILTER else None,
    }
    return filters


def _skip_existing_enabled(args: argparse.Namespace) -> bool:
    return bool(SKIP_EXISTING_OUTPUTS or args.skip_existing)


def _should_skip(output_path: Path, args: argparse.Namespace) -> bool:
    return _skip_existing_enabled(args) and output_path.exists() and not args.force


def _run_module1(args: argparse.Namespace, raw_dir: Path, processed_dir: Path, filters: dict) -> Path:
    from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline

    output_file = processed_dir / MODULE_01_DIRNAME / MODULE_01_OUTPUT_FILENAME

    if _should_skip(output_file, args):
        info(f"Module 1 output already exists: {output_file}")
        info("Skip mode enabled; skipping Module 1.")
        return output_file

    xlsx_files = sorted(raw_dir.glob(RAW_INPUT_FILE_PATTERN))
    if not xlsx_files:
        error(f"No Excel datasets found in: {raw_dir}")
        sys.exit(1)

    print_loading_datasets()
    print_discovered(len(xlsx_files))
    for file_path in xlsx_files:
        info(f"  Dataset: {file_path.name}")

    print_running_module(1, "Data Preprocessing")
    run_full_preprocessing_pipeline(
        input_file=[str(path) for path in xlsx_files],
        output_file=str(output_file),
        config={
            "missing_value_mode": M1_MISSING_VALUE_MODE,
            "outlier_mode": M1_OUTLIER_MODE,
            "outlier_threshold": M1_OUTLIER_THRESHOLD,
            "log_transform": M1_LOG_TRANSFORM,
            "detrend_method": M1_DETREND_METHOD,
            "stationarity_test": M1_STATIONARITY_TEST,
            "stationarity_significance_level": M1_STATIONARITY_SIGNIFICANCE_LEVEL,
            "require_stationarity": M1_REQUIRE_STATIONARITY,
            "differencing_mode": M1_DIFFERENCING_MODE,
            "manual_differencing_order": M1_MANUAL_DIFFERENCING_ORDER,
            "max_differencing_order": M1_MAX_DIFFERENCING_ORDER,
            "standardization_enabled": M1_STANDARDIZATION_ENABLED,
            "standardization_method": M1_STANDARDIZATION_METHOD,
            "duplicate_strategy": args.duplicate_strategy,
            "markets": filters.get("markets"),
            "grades": filters.get("grades"),
            "date_range": filters.get("date_range"),
        },
    )

    print_exporting()
    success(f"Module 1 output saved to: {output_file}")
    return output_file


def _run_module2(args: argparse.Namespace, module1_csv: Path, processed_dir: Path, filters: dict) -> Path:
    from modules.causality_testing.granger_tester import run_full_granger_analysis

    output_dir = processed_dir / MODULE_02_DIRNAME
    output_json = output_dir / MODULE_02_MAIN_OUTPUT_FILENAME

    if _should_skip(output_json, args):
        info(f"Module 2 output already exists: {output_json}")
        info("Skip mode enabled; skipping Module 2.")
        return output_json

    if not module1_csv.exists():
        print_prerequisite_error(2, str(module1_csv))
        sys.exit(1)

    print_running_module(2, "Pairwise Granger Analysis")

    run_full_granger_analysis(
        input_file=str(module1_csv),
        output_dir=str(output_dir),
        config={
            "lag_selection_mode": M2_LAG_SELECTION_MODE,
            "lag_order": M2_LAG_ORDER,
            "price_col": M2_PRICE_COL,
            "significance_mode": M2_SIGNIFICANCE_MODE,
            "significance_level": M2_SIGNIFICANCE_LEVEL,
            "auto_significance_level": M2_AUTO_SIGNIFICANCE_LEVEL,
            "max_lag": M2_MAX_LAG,
            "lag_criterion": M2_LAG_CRITERION,
            "apply_fdr": M2_APPLY_BH_CORRECTION,
            "markets": filters.get("markets"),
            "grades": filters.get("grades"),
            "date_range": filters.get("date_range"),
        },
    )

    print_exporting()
    success(f"Module 2 output saved to: {output_dir}")
    return output_json


def _run_module3(args: argparse.Namespace, module2_json: Path, processed_dir: Path, filters: dict) -> Path:
    from modules.module_03_network_inference.network_builder import run_module3_network_inference

    output_dir = processed_dir / MODULE_03_DIRNAME
    output_json = output_dir / MODULE_03_MAIN_OUTPUT_FILENAME

    if _should_skip(output_json, args):
        info(f"Module 3 output already exists: {output_json}")
        info("Skip mode enabled; skipping Module 3.")
        return output_json

    if not module2_json.exists():
        print_prerequisite_error(3, str(module2_json))
        sys.exit(1)

    if filters.get("date_range") is not None:
        warn("Date filtering is not available for Module 3 standalone runs; ignoring --date-range.")

    print_running_module(3, "Network Recovery")
    print_recovering_dag()
    print_generating_leaders()

    run_module3_network_inference(
        granger_json_path=str(module2_json),
        output_dir=str(output_dir),
        enforce_dag=M3_ENFORCE_DAG,
        markets=filters.get("markets"),
        grades=filters.get("grades"),
    )

    print_exporting()
    success(f"Module 3 output saved to: {output_dir}")
    return output_json


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    level_name = LOG_LEVEL_DEFAULT
    if args.debug:
        level_name = LOG_LEVEL_DEBUG
    elif args.verbose:
        level_name = LOG_LEVEL_VERBOSE

    logging.basicConfig(
        level=getattr(logging, str(level_name).upper(), logging.WARNING),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )

    print_banner()
    print_initializing()

    raw_dir = Path(args.input) if args.input else Path(CLI_DEFAULT_INPUT_DIR)
    processed_dir = Path(args.output) if args.output else Path(CLI_DEFAULT_OUTPUT_DIR)
    filters = _build_subset_filters(args)

    info(f"Raw data directory  : {raw_dir}")
    info(f"Output directory    : {processed_dir}")
    info(f"Duplicate strategy  : {args.duplicate_strategy}")
    info(f"Grade filter        : {filters['grades'] if filters['grades'] else 'all'}")
    info(f"Market filter       : {filters['markets'] if filters['markets'] else 'all'}")
    if filters['date_range'] is None:
        info("Date filter         : all")
    else:
        start_date, end_date = filters['date_range']
        start_text = start_date.date().isoformat() if start_date else ""
        end_text = end_date.date().isoformat() if end_date else ""
        info(f"Date filter         : {start_text}:{end_text}")

    if _skip_existing_enabled(args):
        warn("Skip-existing mode is enabled; modules may be skipped if expected outputs already exist.")

    start_time = time.monotonic()

    try:
        if args.all:
            module1_csv = _run_module1(args, raw_dir, processed_dir, filters)
            module2_json = _run_module2(args, module1_csv, processed_dir, filters)
            _run_module3(args, module2_json, processed_dir, filters)
        elif args.module == 1:
            _run_module1(args, raw_dir, processed_dir, filters)
        elif args.module == 2:
            module1_csv = processed_dir / MODULE_01_DIRNAME / MODULE_01_OUTPUT_FILENAME
            _run_module2(args, module1_csv, processed_dir, filters)
        elif args.module == 3:
            module2_json = processed_dir / MODULE_02_DIRNAME / MODULE_02_MAIN_OUTPUT_FILENAME
            _run_module3(args, module2_json, processed_dir, filters)
    except KeyboardInterrupt:
        print()
        error("Pipeline interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        error(f"Pipeline failed: {exc}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1

    print_pipeline_success()
    print_execution_time(time.monotonic() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
