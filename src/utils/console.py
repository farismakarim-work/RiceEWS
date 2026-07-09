"""
RiceEWS - Shared Console Utility
=================================

Single source of truth for all terminal output formatting used by every
runner script.  Import this module; never duplicate print statements.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import timedelta
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Package metadata
# ──────────────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"
PACKAGE_NAME = "RiceEWS"
PACKAGE_FULL_NAME = "Rice Price Early Warning System"
SEPARATOR = "=" * 60
SEPARATOR_THIN = "-" * 60

# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────


def print_banner() -> None:
    """Print the standard RiceEWS startup banner."""
    print(SEPARATOR)
    print(f"  {PACKAGE_NAME}")
    print(f"  {PACKAGE_FULL_NAME}")
    print(f"  Version : {VERSION}")
    print(SEPARATOR)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Section headers
# ──────────────────────────────────────────────────────────────────────────────


def print_section(title: str) -> None:
    """Print a thin-line section header."""
    print()
    print(SEPARATOR_THIN)
    print(f"  {title}")
    print(SEPARATOR_THIN)


# ──────────────────────────────────────────────────────────────────────────────
# Status messages
# ──────────────────────────────────────────────────────────────────────────────


def info(message: str) -> None:
    """Print an informational message."""
    print(f"  {message}")


def success(message: str) -> None:
    """Print a success message."""
    print(f"  [OK]  {message}")


def warn(message: str) -> None:
    """Print a warning message to stderr."""
    print(f"  [WARN]  {message}", file=sys.stderr)


def error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"  [ERROR]  {message}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline-level messages
# ──────────────────────────────────────────────────────────────────────────────


def print_initializing() -> None:
    info("Initializing RiceEWS...")


def print_loading_datasets() -> None:
    info("Loading datasets...")


def print_discovered(count: int) -> None:
    info(f"Discovered {count} input dataset(s).")


def print_validating_schema() -> None:
    info("Validating dataset schema...")


def print_running_module(number: int, name: str) -> None:
    print()
    print(SEPARATOR_THIN)
    info(f"Running Module {number}: {name}")
    print(SEPARATOR_THIN)


def print_recovering_dag() -> None:
    info("Recovering directed acyclic graph...")


def print_generating_leaders() -> None:
    info("Generating market leader summary...")


def print_exporting() -> None:
    info("Exporting results...")


def print_pipeline_success() -> None:
    print()
    success("Pipeline completed successfully.")


def print_execution_time(elapsed_seconds: float) -> None:
    """Print total execution time formatted as HH:MM:SS."""
    td = timedelta(seconds=int(elapsed_seconds))
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    info(f"Execution finished in {hours:02d}:{minutes:02d}:{seconds:02d}.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Prerequisite checks
# ──────────────────────────────────────────────────────────────────────────────


def print_prerequisite_error(module_number: int, missing_path: str) -> None:
    """Print a formatted error when a prerequisite output file is missing."""
    error(
        f"Module {module_number} prerequisite not satisfied."
    )
    error(
        f"Required file not found: {missing_path}"
    )
    error(
        f"Please run Module {module_number - 1} first before executing Module {module_number}."
    )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Summary box
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(lines: list[str]) -> None:
    """Print a block of summary key-value pairs."""
    print()
    print(SEPARATOR_THIN)
    info("Run Summary")
    print(SEPARATOR_THIN)
    for line in lines:
        info(line)
    print(SEPARATOR_THIN)
    print()
