import argparse
import sys

import dotenv

from sdbench import case_data, judge
from sdbench.config import SDBenchConfig


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run check_diagnosis for a given case ID in a given dataset"
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("case_id", help="Case ID to evaluate")
    parser.add_argument("diagnosis", help="Diagnosis to check against the case")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to show all messages"
    )

    args, remaining_argv = parser.parse_known_args()
    config = SDBenchConfig.from_command_line(args.dataset, remaining_argv)
    config.debug = args.debug

    try:
        # Load the case data
        print(f"Loading case {args.case_id} from dataset '{args.dataset}'...")
        case = case_data.get_case(args.dataset, args.case_id)
        print(f"Found case: {case['case_title']}")
        print(f"Reference diagnosis: {case['final_diagnosis']}")
        print()

        # Run diagnosis check
        print(f"Checking diagnosis: '{args.diagnosis}'")
        print(f"Using model: {config.judge_model}")
        print(f"Using prompt version: {config.judge_prompt_version}")
        print()

        passed, response_data = judge.check_diagnosis(
            config,
            args.diagnosis,
            case,
        )

        # Display results
        print("=" * 60)
        print("DIAGNOSIS CHECK RESULTS")
        print("=" * 60)
        print(f"Diagnosis: {args.diagnosis}")
        print(f"Reference: {case['final_diagnosis']}")
        print(f"Score: {response_data.get('diagnosis_score')}")
        print(f"Passed: {'✓' if passed else '✗'} (score >= 4)")
        print()
        print("Judge Response:")
        print("-" * 40)
        print(response_data)
        print()

        # Exit with appropriate code
        sys.exit(0 if passed else 1)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
