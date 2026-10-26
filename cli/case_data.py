import argparse
import sys

import dotenv

from sdbench import case_data


def main():
    """Exercise the case data loading functionality."""
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Exercise SDBench case data loading")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--case-id", "-c", help="Specific case ID to load and display")
    parser.add_argument("--case-set", "-s", help="Show contents of specific case set")
    parser.add_argument(
        "--list-only",
        "-l",
        action="store_true",
        help="Only list case sets, don't load individual cases",
    )

    args = parser.parse_args()

    print("=== SDBench Case Data Loader ===")
    print(f"Dataset: {args.dataset}")
    print()

    # Load and report case sets
    print("Loading case sets...")
    case_sets = case_data._get_case_sets(args.dataset)  # pyright: ignore[reportPrivateUsage]
    print()

    print("Available case sets:")
    for set_name, cases in case_sets.items():
        print(f"  {set_name}: {len(cases)} cases")
    print()

    if args.list_only:
        return

    # Show specific case set contents if requested
    if args.case_set:
        if args.case_set not in case_sets:
            print(f"Error: Case set '{args.case_set}' not found")
            print(f"Available sets: {list(case_sets.keys())}")
            sys.exit(1)

        print(f"=== Case Set: {args.case_set} ===")
        cases_in_set = case_sets[args.case_set]
        case_ids = [case["case_id"] for case in cases_in_set]
        print(f"Case IDs ({len(case_ids)}): {', '.join(sorted(case_ids))}")
        print()

    # Load and display specific case if requested
    if args.case_id:
        print(f"=== Loading Case ID: {args.case_id} ===")
        try:
            case = case_data.get_case(args.dataset, args.case_id)

            print(f"Case ID: {case['case_id']}")
            print(f"Dataset: {case['dataset']}")
            print(f"Title: {case['case_title']}")
            print(f"Final Diagnosis: {case['final_diagnosis']}")
            print()
            print("Case Summary:")
            print(f"  {case['case_summary']}")
            print()
            print("Full Text (first 500 chars):")
            print(f"  {case['full_text'][:500]}...")
            print()

        except ValueError as e:
            print(f"Error: {e}")
            # Show available case IDs from first few cases
            all_cases = case_sets["all"]
            sample_ids = [case["case_id"] for case in all_cases[:10]]
            print(f"Sample case IDs: {', '.join(sample_ids)}")
            if len(all_cases) > 10:
                print(f"... and {len(all_cases) - 10} more cases")
            sys.exit(1)

    # If no specific case requested, show summary of first case from each set
    if not args.case_id and not args.case_set:
        print("=== Sample Cases from Each Set ===")
        for set_name, cases in case_sets.items():
            if cases and set_name != "all":  # Skip 'all' to avoid duplication
                first_case = cases[0]
                print(f"{set_name}: Case {first_case['case_id']} - {first_case['case_title']}")
        print()
        print("Use --case-id <id> to load a specific case")
        print("Use --case-set <name> to show all cases in a set")


if __name__ == "__main__":
    main()
