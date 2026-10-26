import argparse
import sys

import dotenv

import case_data
from config import SDBenchConfig
from gatekeeper import order_test, order_tests


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Exercise the SDBench gatekeeper directly")

    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("case_id", help="Case ID to run tests against")
    parser.add_argument("tests", nargs="+", help="Tests to run (space-separated)")
    parser.add_argument(
        "--use-order-test",
        action="store_true",
        help="Use order_test for single tests instead of order_tests",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args, remaining_argv = parser.parse_known_args()

    config = SDBenchConfig.from_command_line(args.dataset, remaining_argv)
    config.debug = args.debug

    print("=== SDBench Gatekeeper Exercise ===")
    print(f"Dataset: {args.dataset}")
    print(f"Case ID: {args.case_id}")
    print(f"Tests: {args.tests}")
    print(f"Use order_test for singles: {args.use_order_test}")
    print(f"Debug: {args.debug}")
    print(f"Config: {config}")
    print()

    # Load the case
    try:
        print("Loading case...")
        case = case_data.get_case(args.dataset, args.case_id)
        print(f"Loaded case: {case['case_title']}")
        print()
    except Exception as e:
        print(f"Error loading case: {e}")
        sys.exit(1)

    # Decide which gatekeeper function to use
    try:
        if len(args.tests) == 1 and args.use_order_test:
            print(f"Using order_test for single test: {args.tests[0]}")
            result = order_test(
                config=config,
                case=case,
                test_name=args.tests[0],
            )
            results = [result]
            function_used = "order_test"
        else:
            print(f"Using order_tests for {len(args.tests)} test(s)")
            results = order_tests(
                config=config,
                case=case,
                test_names=args.tests,
            )
            function_used = "order_tests"

        print()
        print("=== SUMMARY ===")
        print(f"Tests run: {len(results)}")
        print(f"Function used: {function_used}")

        print("\n=== FULL RESULT DETAILS ===")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
            print()

    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
