import argparse
import sys
from typing import Any

import dotenv

from sdbench import case_data, config as sdbench_config
from sdbench.gatekeeper import BatchTestRunner


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise the SDBench BatchTestRunner")

    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("case_id", help="Case ID to run tests against")
    parser.add_argument("tests", nargs="+", help="Tests to run (space-separated)")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Number of tests per batch (default: 2)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args, remaining_argv = parser.parse_known_args()

    config = sdbench_config.SDBenchConfig.from_command_line(args.dataset, remaining_argv)
    config.debug = args.debug

    print("=== SDBench BatchTestRunner Exercise ===")
    print(f"Dataset: {args.dataset}")
    print(f"Case ID: {args.case_id}")
    print(f"Tests: {args.tests}")
    print(f"Batch size: {args.batch_size}")
    print(f"Debug: {args.debug}")
    print(f"Config: {config}")
    print()

    # Validate batch size
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        sys.exit(1)

    # Load the case
    print("Loading case...")
    case = case_data.get_case(args.dataset, args.case_id)
    print(f"Loaded case: {case['case_title']}")
    print()

    # Split tests into batches of specified size
    batch_size = args.batch_size
    test_batches: list[tuple[str, list[str]]] = []
    for i in range(0, len(args.tests), batch_size):
        batch = args.tests[i : i + batch_size]
        test_batches.append((f"Batch {len(test_batches) + 1}", batch))

    print(
        f"Splitting {len(args.tests)} tests into {len(test_batches)} batches of up to {batch_size} tests each"
    )
    for batch_name, tests in test_batches:
        print(f"  {batch_name}: {tests}")
    print()

    runner = BatchTestRunner(
        config=config,
        case=case,
        collector={},
    )

    all_results: list[dict[str, Any]] = []
    for batch_name, tests in test_batches:
        print(f"=== Running {batch_name}: {tests} ===")

        try:
            batch_results = runner(tests)

            print(f"=== {batch_name.upper()} RESULTS ===")
            print(batch_results.format_test_results())

            # Store batch results
            batch_summary: dict[str, Any] = {
                "batch_name": batch_name,
                "tests": tests,
                "results": batch_results.test_results,
                "tests_run": len(batch_results.test_results),
                "turns_used": runner.num_rounds,
                "turns_left": runner.turns_left,
                "completed_tests": len(runner.completed_tests),
            }
            all_results.append(batch_summary)

            print(f"=== {batch_name.upper()} SUMMARY ===")
            print(f"Tests run: {batch_summary['tests_run']}")
            print(f"Turns used: {batch_summary['turns_used']}")
            print(f"Turns left: {batch_summary['turns_left']}")
            print(f"Completed tests: {batch_summary['completed_tests']}")
            print()

        except Exception as e:
            print(f"Error running {batch_name}: {e}")
            all_results.append({"batch_name": batch_name, "tests": tests, "error": str(e)})
            print()

    # Final summary
    print("=== FINAL SUMMARY ===")
    total_tests = sum(r.get("tests_run", 0) for r in all_results)
    total_turns = sum(r.get("turns_used", 0) for r in all_results)
    successful_batches = sum(1 for r in all_results if "error" not in r)

    print(f"Total batches: {len(all_results)}")
    print(f"Successful batches: {successful_batches}/{len(all_results)}")
    print(f"Total tests run: {total_tests}")
    print(f"Total turns used: {total_turns}")

    # Show any errors
    failed_batches = [r for r in all_results if "error" in r]
    if failed_batches:
        print("\nFailed batches:")
        for batch in failed_batches:
            print(f"  {batch['batch_name']}: {batch['error']}")

    # Exit with error code if any batches failed
    if failed_batches:
        sys.exit(1)


if __name__ == "__main__":
    main()
