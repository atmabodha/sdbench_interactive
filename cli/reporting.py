import argparse
import json
import os

from config import HarnessConfig
from reporting import (
    DEFAULT_REPORT_CASE_SETS,
    build_accuracy_cost_report,
    load_results_from_json,
    print_accuracy_cost_report,
    process_results,
)


def main():
    """Re-report on one or more completed run folders.

    For each folder, regenerates that folder's summary.txt (with its own per-folder
    accuracy/cost report folded in) and HTML. Then prints a distinct aggregate
    accuracy + cost report pooled by (agent, model) across all folders, and broken
    out by case set, optionally writing it to JSON.
    """
    parser = argparse.ArgumentParser(
        description="Process SDBench results: per-run summaries/HTML plus a grouped "
        "accuracy/cost report (with optional bootstrapped CIs)."
    )
    parser.add_argument("results_dirs", nargs="+", help="One or more run directories (each with json/)")
    parser.add_argument(
        "--ci", action="store_true", help="Compute bootstrapped confidence intervals"
    )
    parser.add_argument("--json", dest="json_out", help="Write the accuracy/cost report to this JSON file")
    parser.add_argument(
        "--case-sets",
        nargs="+",
        default=list(DEFAULT_REPORT_CASE_SETS),
        help=f"Case sets to report (default: {' '.join(DEFAULT_REPORT_CASE_SETS)})",
    )

    args = parser.parse_args()

    valid_dirs = []
    for results_dir in args.results_dirs:
        results_dir = os.path.abspath(results_dir)
        json_dir = os.path.join(results_dir, "json")
        if not os.path.isdir(json_dir):
            print(f"Skipping {results_dir}: no json/ directory")
            continue
        html_dir = os.path.join(results_dir, "html")
        os.makedirs(html_dir, exist_ok=True)
        print(f"Processing results from: {results_dir}")
        results = load_results_from_json(json_dir)
        if not results:
            print("  No JSON result files found, skipping")
            continue
        config = HarnessConfig.restore_run_config(results_dir)
        config.azure_storage_enabled = False
        # Regenerate this folder's summary.txt + HTML, with its own per-folder report folded in.
        process_results(results_dir, html_dir, config)
        valid_dirs.append(results_dir)

    if not valid_dirs:
        print("No valid run directories with results found.")
        return

    # Distinct aggregate report: pooled by (agent, model) across all folders.
    report = build_accuracy_cost_report(valid_dirs, case_sets=tuple(args.case_sets), compute_ci=args.ci)
    title = f"AGGREGATE ACCURACY & COST REPORT  (pooled across {len(valid_dirs)} folder(s) by agent/model)"
    print_accuracy_cost_report(report, title=title)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.json_out}")


if __name__ == "__main__":
    main()
