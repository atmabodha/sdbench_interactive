#!/usr/bin/env python3
"""Batch (re-)costing for completed SDBench runs.

Given one or more results directories, recompute the agent's per-test costs and report
how many ordered tests received a *grounded* cost -- matched to the hospital charge master
via CPT lookup -- versus how many fell back to an LLM estimate.

The default ``unified_cpt_llm`` method prices each test against the charge master and falls
back to an LLM estimate, and the breakdown reports grounded (CPT) vs LLM-fallback vs failed.
``--costing-method llm_only`` / ``batch_llm_only`` price every test with the LLM instead, and
the breakdown simply reports priced vs failed. Either way the breakdown reflects only the
costing method being run, ignoring any other cost analyses already stored on the case.

A high grounded hit rate is what makes a run's costs comparable to the published paper:
grounded (CPT) and LLM-estimated costs are not directly comparable, so the LLM fallback
should be filling gaps rather than carrying the bulk of the tests (see the "Costing"
section of the README).

The grounded ``unified_cpt_llm`` method requires the optional ``cpt_costing`` dependencies
and the configured charge-master spreadsheet; see ``data/costing_data/README.md``. Enable
the extra at run time with ``uv run --extra cpt_costing`` (or ``uv sync --extra cpt_costing``).

Usage::

    # Re-cost a single run and print the grounded vs LLM-fallback breakdown
    uv run --extra cpt_costing python -m sdbench.cli.recost results/my_run

    # Several runs at once (reported individually and in aggregate)
    uv run --extra cpt_costing python -m sdbench.cli.recost results/run_a results/run_b

    # Recompute every case, even ones that already have grounded costs
    uv run --extra cpt_costing python -m sdbench.cli.recost results/my_run --force

    # Only report on costs already stored in the JSON, without recomputing (no extra needed)
    uv run python -m sdbench.cli.recost results/my_run --analyze-only

    # Use the LLM-only costing method instead of grounded CPT (no extra needed)
    uv run python -m sdbench.cli.recost results/my_run --costing-method batch_llm_only

Any unrecognised flags are forwarded to the costing config, so the usual costing knobs
(e.g. ``--clinician-cost``, ``--costing-model``) work here too.
"""

import argparse
import glob
import json
import os
import sys
from typing import Any

import dotenv
from tqdm import tqdm

from config import CostingConfig, SDBenchConfig
from costing.cost_types import CostsEstimate
from harness.harness import compute_agent_costs

# cost_details["method"] values, grouped into the categories we report on. "cpt_lookup"
# is the grounded charge-master match; everything LLM-based is a fallback estimate.
GROUNDED_METHOD = "cpt_lookup"
LLM_METHODS = ("llm_fallback", "llm_only", "batch_llm_only")
FAILURE_METHOD = "failure"


def preflight_unified_costing(config: CostingConfig) -> None:
    """Fail fast if the grounded ``unified_cpt_llm`` backend can't be initialized.

    Without this, a missing charge master (or any other costing misconfiguration) makes
    the CPT matcher raise during initialization, which the costing layer turns into a
    per-test pricing *failure*. The run then completes "successfully" while silently
    writing failures into every case JSON and reporting a 100%-failed breakdown. We'd
    rather stop immediately with an actionable message.
    """
    # Grounded costing needs the optional 'cpt_costing' dependencies (faiss, pandas, numpy,
    # fuzzywuzzy). Check for them first so a missing extra produces a clear, actionable
    # message instead of an import traceback from deep inside the matcher.
    try:
        import costing.cpt_matcher  # noqa: F401
    except ImportError as e:
        sys.exit(
            "Error: grounded ('unified_cpt_llm') costing requires the optional 'cpt_costing' "
            f"dependencies, which aren't installed ({e}).\n"
            "Run the command with the extra enabled, e.g.:\n"
            "    uv run --extra cpt_costing python -m sdbench.cli.recost ...\n"
            "or add the extra to your environment first:\n"
            "    uv sync --extra cpt_costing"
        )

    # The most common failure, by far: the hospital charge master spreadsheet is missing.
    source_file = config.resolved_costing_source_file_path
    if not os.path.exists(source_file):
        sys.exit(
            f"Error: grounded costing data not found at '{source_file}'.\n"
            "Grounded ('unified_cpt_llm') costing needs the hospital charge master "
            "spreadsheet. Download it as described in data/costing_data/README.md (or pass "
            "--costing-source-file-path), then re-run."
        )

    # Catch any other initialization problem up front (missing 'cpt_costing' extras,
    # embedding-build failure, bad costing model / API config). get_cpt_fuzzy_matcher
    # caches on the config, so this also warms the matcher the per-case loop will reuse.
    from costing.costing import get_cpt_fuzzy_matcher

    print("Initializing grounded (unified_cpt_llm) costing backend...")
    if not os.path.exists(source_file + ".faiss.index"):
        # The first run embeds every charge-master procedure name to build a search index.
        print(
            "  First run: building a test-name embedding index from the charge master. "
            "This calls the embedding API and can take a few minutes. The index is cached "
            "next to the charge master, so subsequent runs start quickly. Case (re-)costing "
            "begins once this completes."
        )
    if get_cpt_fuzzy_matcher(config) is None:
        sys.exit(
            "Error: failed to initialize the grounded CPT cost matcher (see traceback above). "
            "Ensure the 'cpt_costing' extras are installed (uv sync --extra cpt_costing) and "
            "your costing model / API settings are correct."
        )


def find_case_files(run_dir: str) -> list[str]:
    """Return the sorted case JSON files for a run directory (``<run_dir>/json/case_k*.json``)."""
    json_dir = os.path.join(run_dir, "json")
    if not os.path.isdir(json_dir):
        print(f"Warning: no 'json' directory found in {run_dir!r}, skipping")
        return []
    return sorted(glob.glob(os.path.join(json_dir, "case_k*.json")))


def has_cost_failures(cost_analysis: CostsEstimate | dict[str, Any] | None) -> bool:
    """True if any test in a stored cost analysis failed to price."""
    if not cost_analysis:
        return False
    return any(d.get("method") == FAILURE_METHOD for d in cost_analysis.get("cost_details", []))


def tally_methods(cost_analysis: CostsEstimate | dict[str, Any]) -> dict[str, int]:
    """Count grounded / LLM-fallback / failed tests in a single case's cost analysis."""
    counts = {"grounded": 0, "llm": 0, "failed": 0}
    for detail in cost_analysis.get("cost_details", []):
        method = detail.get("method")
        if method == GROUNDED_METHOD:
            counts["grounded"] += 1
        elif method in LLM_METHODS:
            counts["llm"] += 1
        elif method == FAILURE_METHOD:
            counts["failed"] += 1
    return counts


def recost_case(
    case_file: str, config: CostingConfig, *, force: bool, analyze_only: bool
) -> dict[str, int]:
    """(Re-)cost a single case file in place and return its per-method tally.

    Recomputes when the case has no grounded cost analysis yet, when an existing one
    contains failures, or when ``force`` is set. With ``analyze_only`` nothing is
    recomputed and we simply report whatever is already stored.
    """
    with open(case_file, "r") as f:
        result = json.load(f)

    cost_key = f"cost_analysis_{config.costing_method}"
    existing = result.get(cost_key)

    if not analyze_only and (force or existing is None or has_cost_failures(existing)):
        compute_agent_costs(result, config)
        with open(case_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        existing = result[cost_key]

    if not existing:
        return {"grounded": 0, "llm": 0, "failed": 0}
    return tally_methods(existing)


def print_breakdown(label: str, totals: dict[str, int], costing_method: str) -> None:
    """Print the per-test cost breakdown for a set of cases, tailored to the costing method.

    For the grounded ``unified_cpt_llm`` method we report grounded (CPT) vs LLM-fallback vs
    failed. For the LLM-only methods there is no grounded component, so we just report how
    many tests were priced vs failed.
    """
    grounded, llm, failed = totals["grounded"], totals["llm"], totals["failed"]
    total = grounded + llm + failed

    print(f"\n{label}")
    print("-" * len(label))
    if total == 0:
        print("  No costed tests found.")
        return

    def pct(n: int) -> float:
        return 100.0 * n / total

    print(f"  Total tests costed:   {total:6d}")
    if costing_method == "unified_cpt_llm":
        # The headline number: share of ordered tests that got a grounded (CPT) cost.
        print(f"  Grounded (CPT):       {grounded:6d}  ({pct(grounded):5.1f}%)")
        print(f"  LLM fallback:         {llm:6d}  ({pct(llm):5.1f}%)")
    else:
        # LLM-only methods: no grounded component, so just success vs failure.
        priced = grounded + llm
        print(f"  Priced (LLM):         {priced:6d}  ({pct(priced):5.1f}%)")
    if failed:
        print(f"  Failed to price:      {failed:6d}  ({pct(failed):5.1f}%)")


def main() -> None:
    dotenv.load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description=(
            "Re-cost completed SDBench runs and report the grounded (CPT) vs LLM-fallback "
            "breakdown of ordered tests."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more results directories, each containing a 'json/' subdirectory.",
    )
    parser.add_argument(
        "--dataset", default="nejm", help="Dataset name for config loading (default: nejm)."
    )
    parser.add_argument(
        "--costing-method",
        choices=["unified_cpt_llm", "llm_only", "batch_llm_only"],
        default="unified_cpt_llm",
        help=(
            "Costing method (default: unified_cpt_llm). 'unified_cpt_llm' prices each test "
            "against the hospital charge master (grounded CPT lookup) and falls back to an "
            "LLM estimate; 'llm_only' / 'batch_llm_only' price every test with the LLM. Only "
            "'unified_cpt_llm' needs the charge master and the 'cpt_costing' extras."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute costs for every case, even ones that already have grounded costs.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Don't recompute anything; just report on costs already stored in the JSON.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    args, remaining_argv = parser.parse_known_args()

    # Drive the costing config off the chosen method. SDBenchConfig parses --costing-method
    # from the forwarded argv, so re-inject the explicit choice (a later --costing-method in
    # the forwarded args still wins, for power users passing extra costing flags).
    remaining_argv.insert(0, f"--costing-method={args.costing_method}")
    sdbench_config = SDBenchConfig.from_command_line(args.dataset, remaining_argv)
    sdbench_config.debug = args.debug
    config = CostingConfig.from_sdbench_config(sdbench_config)

    # Fail fast on a misconfigured grounded backend before touching any case files.
    # In --analyze-only we only read already-stored costs, so the matcher isn't needed.
    if not args.analyze_only and config.costing_method == "unified_cpt_llm":
        preflight_unified_costing(config)

    overall = {"grounded": 0, "llm": 0, "failed": 0}
    per_run_totals: list[tuple[str, dict[str, int]]] = []

    for run_dir in args.run_dirs:
        case_files = find_case_files(run_dir)
        if not case_files:
            continue

        run_totals = {"grounded": 0, "llm": 0, "failed": 0}
        desc = "Analyzing" if args.analyze_only else "Re-costing"
        for case_file in tqdm(case_files, desc=f"{desc} {os.path.basename(run_dir.rstrip('/'))}", unit="case"):
            try:
                counts = recost_case(
                    case_file, config, force=args.force, analyze_only=args.analyze_only
                )
            except Exception as e:  # noqa: BLE001 - keep going across the rest of the run
                print(f"\nError processing {case_file}: {e}")
                continue
            for k in run_totals:
                run_totals[k] += counts[k]

        per_run_totals.append((run_dir, run_totals))
        for k in overall:
            overall[k] += run_totals[k]

    if not per_run_totals:
        print("No case files found in any of the provided run directories.")
        return

    # Per-run breakdowns are only worth printing when there's more than one run.
    if len(per_run_totals) > 1:
        for run_dir, run_totals in per_run_totals:
            print_breakdown(run_dir, run_totals, config.costing_method)
        print_breakdown("ALL RUNS", overall, config.costing_method)
    else:
        print_breakdown(per_run_totals[0][0], overall, config.costing_method)


if __name__ == "__main__":
    main()
