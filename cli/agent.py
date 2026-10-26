import argparse
import json
import pprint

import dotenv

import case_data
from config import get_sdbench_config
from harness import run_case


def main():
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(description="Run a single SDBench case evaluation")
    parser.add_argument("dataset", type=str, help="Dataset to use")
    parser.add_argument("case_id", type=str, help="Case ID to evaluate")
    parser.add_argument(
        "--agent",
        type=str,
        default="baseline",
        help="Diagnostic agent to use (default: baseline)",
    )
    parser.add_argument("--model", type=str, default="o4-mini", help="Model for the agent to use")
    parser.add_argument(
        "--model-args", type=json.loads, default=None, help="Model arguments as JSON dict"
    )
    parser.add_argument(
        "--case-set", type=str, default="train", help="Case set to search in (default: train)"
    )
    parser.add_argument(
        "--sdbench-config",
        type=str,
        default="default",
        help="SDBench config name (default: default)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--output", type=str, default=None, help="Write result JSON to file")

    args = parser.parse_args()

    config = get_sdbench_config(args.dataset, config_name=args.sdbench_config)
    config.debug = args.debug

    cases = case_data.get_case_set(args.dataset, args.case_set)
    case = next((c for c in cases if c["case_id"] == args.case_id), None)
    if case is None:
        print(f"Error: Case '{args.case_id}' not found in {args.dataset}/{args.case_set}")
        exit(1)

    print(f"Running case {args.case_id} with agent={args.agent}, model={args.model}")
    result = run_case(config, case, args.agent, args.model, args.model_args)

    correct = result["correct"]
    print(f"\nDiagnosis: {result['model_diagnosis']}")
    print(f"Actual: {result['actual_diagnosis']}")
    print(f"Correct: {correct}")
    print(f"Tests ordered: {result['num_test_rounds']} rounds")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult written to {args.output}")

    if args.debug:
        print("\nFull result:")
        pprint.pprint(result, indent=2, width=120)


if __name__ == "__main__":
    main()
