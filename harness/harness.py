import concurrent.futures
import datetime
import glob
import json
import os
import pprint
import re
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from sdbench import agents, case_data, gatekeeper, judge, models
from sdbench.azure_storage import validate_azure_access, write_output_file
from sdbench.config import AgentConfig, CostingConfig, HarnessConfig, SDBenchConfig
from sdbench.costing.costing import estimate_costs
from sdbench.models import ModelCreateChatCompletion

__all__ = [
    "run_case",
    "run",
    "scan_completed_cases",
]


def run_case(
    config: SDBenchConfig,
    case: case_data.CaseData,
    agent_name: str,
    model: str | ModelCreateChatCompletion,
    model_args: dict[str, Any] | None = None,
    gatekeeper_llm: ModelCreateChatCompletion | None = None,
    judge_llm: ModelCreateChatCompletion | None = None,
) -> dict[str, Any]:
    """Run a diagnostic agent on a single case and return results.

    Args:
        config: SDBench configuration.
        case: The case to evaluate.
        agent_name: Name of the diagnostic agent (e.g., "baseline").
        model: Model name string for the agent LLM, or a pre-built
            `ModelCreateChatCompletion` callable. When a callable is passed,
            `model_args` and `config.agent_model_args` are ignored.
        model_args: Optional model arguments dict.
        gatekeeper_llm: Optional callable for the gatekeeper. When provided,
            `config.gatekeeper_model` and `config.gatekeeper_model_args` are ignored.
        judge_llm: Optional callable for the judge. When provided,
            `config.judge_model` and `config.judge_model_args` are ignored.

    Returns:
        Results dict including correctness, diagnosis, cost analysis, etc.
    """
    if model_args is None:
        model_args = {}

    agent = getattr(agents, agent_name, None)
    if agent is None:
        raise ValueError(
            f"Unknown agent: {agent_name}. Available: {[k for k in dir(agents) if not k.startswith('_')]}"
        )

    if callable(model):
        agent_llm = model
    else:
        all_model_args = {**model_args, **config.agent_model_args}
        agent_llm = models.model_create_chat_completion_fn(
            model, debug=config.debug, **all_model_args
        )

    results: dict[str, Any] = {}

    test_runner = gatekeeper.BatchTestRunner(config, case, results, llm=gatekeeper_llm)

    initial_prompt = case["case_summary"]
    agent_result = agent.run(
        initial_prompt, test_runner, agent_llm, results, AgentConfig.from_sdbench_config(config)
    )

    correct, judge_response = judge.check_diagnosis(
        config,
        agent_result,
        case,
        test_logs=results.get("test_logs"),
        llm=judge_llm,
    )

    results["correct"] = correct
    results |= judge_response
    results["model_diagnosis"] = agent_result
    results["actual_diagnosis"] = case["final_diagnosis"]

    results["requested_tests"] = test_runner.requested_tests
    results["requested_tests_only"] = test_runner.requested_tests_only
    results["num_test_rounds"] = test_runner.num_test_rounds

    _compute_costs(case, results, config)

    return results


def _compute_costs(case: case_data.CaseData, result: dict[str, Any], config: SDBenchConfig):
    if config.compute_costs:
        compute_agent_costs(result, CostingConfig.from_sdbench_config(config))
    if config.compute_actual_workup_costs:
        compute_actual_workup_costs(case, result, CostingConfig.from_sdbench_config(config))


def compute_agent_costs(result: dict[str, Any], config: CostingConfig):
    num_rounds_for_cost = result["num_test_rounds"] + 1

    result[f"cost_analysis_{config.costing_method}"] = estimate_costs(
        non_rejected_diagnostic_tests(result),
        num_rounds_for_cost,
        config,
    )


def compute_actual_workup_costs(case: case_data.CaseData, result: dict[str, Any], config: CostingConfig):
    # Cost of the real-world workup the patient actually received: the case's own tests
    # (excluding those already part of the initial presentation) plus clinician visits.
    result[f"cost_analysis_actual_workup_{config.costing_method}"] = estimate_costs(
        [x["name"] for x in case.get("tests", []) if not x.get("in_initial_presentation", False)],
        case.get("estimated_clinician_visits", 1),
        config,
    )


def _remove_test_prefix(test_name: str) -> str:
    if not test_name:
        return test_name
    prefix_to_remove = "Test to order: "
    cleaned_name = test_name.strip()
    if cleaned_name.lower().startswith(prefix_to_remove.lower()):
        cleaned_name = cleaned_name[len(prefix_to_remove) :].strip()
    return cleaned_name


def non_rejected_diagnostic_tests(case_result: dict[str, Any]) -> list[str]:
    rejected_tests: set[str] = set()
    rejection_phrases = [
        "i can only provide specific test results or answer specific patient questions",
        "not available",
        "please request a specific diagnostic test",
    ]

    tests_only: list[str] = case_result["requested_tests_only"]
    test_logs: list[dict[str, Any]] = case_result.get("test_logs", [])
    for test_log in test_logs:
        test_name: str = test_log.get("test_name", "")
        test_result: str = test_log.get("result", "")
        if any(phrase.lower() in test_result.lower() for phrase in rejection_phrases):
            rejected_tests.add(test_name)

    filtered_tests = [test for test in tests_only if test not in rejected_tests]
    filtered_tests = [_remove_test_prefix(name) for name in filtered_tests]

    return filtered_tests


# --- batch run orchestration -------------------------------------------------------
# Runs many cases (× k repetitions) over run_case, with parallelism, resume support,
# and per-case result persistence. The CLI wiring lives in sdbench.cli.harness.


def _parse_case_ids(raw_arg: list[str] | str) -> list[str]:
    """Parse case IDs from comma-separated string or list."""
    if isinstance(raw_arg, list):
        # Already a list from nargs='+'
        case_ids: list[str] = []
        for item in raw_arg:
            if ',' in str(item):
                # Handle comma-separated values within a single argument
                case_ids.extend([piece.strip() for piece in str(item).split(',') if piece.strip()])
            else:
                case_ids.append(item.strip())
        return case_ids
    else:
        # Single value
        if ',' in str(raw_arg):
            return list([piece.strip() for piece in str(raw_arg).split(',') if piece.strip()])
        else:
            return [raw_arg.strip()]


def _save_result_to_json(result: dict[str, Any], case_id: str, json_dir: str, config: HarnessConfig, case: case_data.CaseData):
    """Save a single result to a JSON file."""
    k = result.get('k', 0)
    filename = f"{json_dir}/case_k{k}_{case_id}.json"

    # Add metadata to the result
    # Note: All the config is already in the run_config.json file
    result_data: dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "case_id": case_id,
        "dataset": case['dataset'],
        "case_title": case.get("case_title", f"Case {case_id}"),
        "model": config.model,
        "model_args": config.model_args,
        "evaluation_method": "single",  # Add evaluation method for compatibility
        "prompt_versions": f"diagnostic_agent:{config.agent}",  # Add prompt versions for compatibility
        "case_set": config.case_set,
        "max_turns": config.sdbench.max_turns,
        "k": k,
        **result  # Include all the original result data
    }

    json_content = json.dumps(result_data, indent=2)

    # Always save locally first (primary storage)
    with open(filename, 'w') as f:
        f.write(json_content)

    if config.debug:
        print(f"Results saved to {filename}")

    if config.azure_storage_enabled:
        # The last 3 parts of the path, are folder/json/file.json, and will be replicated in the
        # Azure blob results folder.
        blob_filename = str(Path(*Path(filename).parts[-3:]))
        blob_url = write_output_file(config, blob_filename, json_content)
        if blob_url:
            print(f"Results uploaded to {blob_url}")
        else:
            print(f"Error: Failed to upload results to Azure Storage")


def _process_single_case(k: int,
                        case: case_data.CaseData,
                        agent_name: str,
                        model: str,
                        model_args: dict[str, Any],
                        harness_config: HarnessConfig,
                        json_dir: str) -> dict[str, Any]:
    case_id = case['case_id']

    try:
        config = harness_config.sdbench
        result = run_case(config, case, agent_name, model, model_args)
        result['k'] = k
        result['max_turns'] = config.max_turns
        _save_result_to_json(result, case_id, json_dir, harness_config, case)

        return result

    except Exception as e:
        error_result: dict[str, Any] = {

            "model_diagnosis": str(e),
            "correct": False,
            "k": k,
        }
        print(f"[_process_single_case] Error with case {case_id}: {e}")
        traceback.print_exc()

        _save_result_to_json(error_result, case_id, json_dir, harness_config, case)

        return error_result


def scan_completed_cases(resume_dir: str) -> set[tuple[str, int]]:
    """
    Scan a results directory to find completed cases.

    Args:
        resume_dir: Path to the results directory to resume from

    Returns:
        set: Set of (case_id, k) tuples that have been completed successfully
    """
    completed: set[tuple[str, int]] = set()
    error_cases: set[tuple[str, int]] = set()

    # Find the json subdirectory
    json_dir = os.path.join(resume_dir, "json")
    if not os.path.exists(json_dir):
        print(f"Warning: JSON directory not found at {json_dir}")
        return completed

    # Pattern to match: case_k{k}_{case_id}.json
    pattern = re.compile(r'case_k(\d+)_(\d+)\.json')

    json_files = glob.glob(os.path.join(json_dir, "case_k*.json"))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        match = pattern.match(filename)

        if match:
            k = int(match.group(1))
            case_id = match.group(2)

            # Verify the file contains valid results by trying to load it
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                    # Check if it has essential fields indicating completion
                    if 'case_id' in data and 'correct' in data:
                        # Check if the result contains an error
                        model_diagnosis = data.get('model_diagnosis', '')
                        has_error = (
                            isinstance(model_diagnosis, str) and
                            ('Error code:' in model_diagnosis or
                             'error' in model_diagnosis.lower() or
                             'exception' in model_diagnosis.lower())
                        )

                        if has_error:
                            error_cases.add((case_id, k))
                            print(f"Warning: Error found in {json_file}, will be re-run")
                        else:
                            completed.add((case_id, k))
                    else:
                        print(f"Warning: Incomplete result file {json_file}, will be re-run")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Corrupted result file {json_file}: {e}, will be re-run")

    if completed:
        print(f"Found completed cases:")
        # Group by case_id and show k values
        case_groups: dict[str, list[int]] = {}
        for case_id, k in completed:
            if case_id not in case_groups:
                case_groups[case_id] = []
            case_groups[case_id].append(k)

        for case_id in sorted(case_groups.keys()):
            k_values = sorted(case_groups[case_id])
            print(f"  Case {case_id}: k={k_values}")

    if error_cases:
        print(f"Found error cases that will be re-run:")
        # Group by case_id and show k values
        error_groups: dict[str, list[int]] = {}
        for case_id, k in error_cases:
            if case_id not in error_groups:
                error_groups[case_id] = []
            error_groups[case_id].append(k)

        for case_id in sorted(error_groups.keys()):
            k_values = sorted(error_groups[case_id])
            print(f"  Case {case_id}: k={k_values}")

    return completed


def _filter_remaining_work(selected_cases: list[case_data.CaseData],
                          k_range: Iterable[int],
                          completed_cases: set[tuple[str, int]]) -> list[tuple[int, case_data.CaseData]]:
    """
    Filter the work to exclude already completed cases.

    Args:
        selected_cases: List of selected cases
        k_range: Range of k values (0 to k-1)
        completed_cases: Set of (case_id, k) tuples that are completed

    Returns:
        List of (k, case) tuples representing remaining work
    """
    remaining_work: list[tuple[int, case_data.CaseData]] = []

    for case in selected_cases:
        case_id = case['case_id']
        for k in k_range:
            if (case_id, k) not in completed_cases:
                remaining_work.append((k, case))

    return remaining_work


def run(config: HarnessConfig, json_dir: str, completed_cases: set[tuple[str, int]]):

    print("Running with configuration:")
    pprint.pprint(asdict(config), indent=2, sort_dicts=False, width=120, compact=True)

    # Validate Azure Storage access if enabled (fail hard on startup)
    if config.azure_storage_enabled:
        if not config.azure_storage_account_name or not config.azure_storage_container_name or not config.azure_storage_result_folder:
            print("Error: When --azure-storage-enabled is set, --azure-storage-account-name, --azure-storage-container-name and --azure-storage-result-folder must be provided")
            exit(1)
        print("\n" + "="*60)
        print("AZURE STORAGE VALIDATION")
        print("="*60)
        validate_azure_access(
            config.azure_storage_account_name,
            config.azure_storage_container_name
        )

    diagnostic_agent_name = config.agent
    if getattr(agents, diagnostic_agent_name, None) is None:
        print(f"Error: Agent '{diagnostic_agent_name}' not found in sdbench.agents")
        exit(1)

    print(f"Using agent: {diagnostic_agent_name}")

    selected_cases = case_data.get_case_set(config.dataset, config.case_set)

    if config.case_id is not None:
        # Handle multiple case IDs mode
        case_ids = _parse_case_ids(config.case_id)
        selected_cases = [case for case in selected_cases if case['case_id'] in case_ids]
        print(f"Filtering to specific case IDs: {case_ids}")
    else:
        # Apply offset and limit
        if config.case_offset:
            selected_cases = selected_cases[config.case_offset:]
        if config.num_cases > 0:
            selected_cases = selected_cases[:config.num_cases]

    if not selected_cases:
        print(f"No cases found in the {config.case_set} set to run.")
        exit(0)

    # Filter remaining work based on completed cases
    k_range = range(config.k)
    remaining_work = _filter_remaining_work(selected_cases, k_range, completed_cases)

    if not remaining_work:
        print("All cases have been completed. Nothing to resume.")
        exit(0)

    print(f"Total selected cases: {len(selected_cases)}")
    print(f"Total tasks (cases × k): {len(selected_cases) * config.k}")
    print(f"Completed tasks: {len(completed_cases)}")
    print(f"Remaining tasks: {len(remaining_work)}")

    if len(completed_cases) > 0:
        # Show remaining work details if resuming
        print(f"Remaining cases to run:")
        # Group by case_id and show k values
        remaining_groups: dict[str, list[int]] = {}
        for k, case in remaining_work:
            case_id = case['case_id']
            if case_id not in remaining_groups:
                remaining_groups[case_id] = []
            remaining_groups[case_id].append(k)

        for case_id in sorted(remaining_groups.keys()):
            k_values = sorted(remaining_groups[case_id])
            print(f"  Case {case_id}: k={k_values}")

    # Setup execution
    workers = min(config.max_workers, len(remaining_work))

    print(f"Running {len(remaining_work)} tasks")
    print(f"Num workers: {workers}")
    print(f"Using model: {config.model}")
    print(f"Case set: {config.case_set}")
    if config.sdbench.max_questions_per_turn:
        print(f"Max questions per turn: {config.sdbench.max_questions_per_turn}")
    if config.sdbench.max_tests_per_turn:
        print(f"Max tests per turn: {config.sdbench.max_tests_per_turn}")
    print(f"Max turns: {config.sdbench.max_turns}")

    print(f"JSON results will be saved to: {json_dir}")

    model_args = {**config.model_args}

    if config.sdbench.costing_method == "unified_cpt_llm":
        from sdbench.costing.costing import get_cpt_fuzzy_matcher

        print("Pre-initializing grounded (unified_cpt_llm) cost matcher before starting workers...")
        if get_cpt_fuzzy_matcher(CostingConfig.from_sdbench_config(config.sdbench)) is None:
            raise RuntimeError(
                "Failed to initialize the grounded CPT cost matcher. Ensure the 'cpt_costing' "
                "extras are installed and the charge master is configured "
                "(see data/costing_data/README.md)."
            )

    # Run cases
    results: list[dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks and create a mapping from future to case info
        future_to_case: dict[concurrent.futures.Future[dict[str, Any]], tuple[str, int]] = {}
        for k, case in remaining_work:
            future = executor.submit(
                _process_single_case,
                k=k,
                case=case,
                agent_name=diagnostic_agent_name,
                model=config.model,
                model_args=model_args,
                harness_config=config,
                json_dir=json_dir,
            )
            future_to_case[future] = (case['case_id'], k)

        # Initialize progress bar for parallel processing
        pbar = tqdm(total=len(remaining_work), desc="Processing cases", unit="case")
        completed_tasks = 0

        # Process results as they complete (not in submission order)
        for future in concurrent.futures.as_completed(future_to_case):
            case_id, k = future_to_case[future]
            result = future.result()
            results.append(result)

            completed_tasks += 1
            status = "✓" if result["correct"] else "✗"

            # Calculate accuracy percentage so far
            correct_count = sum(1 for r in results if r["correct"])
            accuracy_pct = (correct_count / len(results)) * 100 if results else 0

            # Calculate running average cost per case
            def get_cost_analysis_total(result: dict[str, Any]) -> float:
                # Read the agent cost for the configured costing method (the key the
                # harness actually writes; legacy new_cost_analysis / un-suffixed
                # cost_analysis keys are no longer produced).
                cost_analysis = result.get(f"cost_analysis_{config.sdbench.costing_method}", {})
                if cost_analysis and "totals" in cost_analysis:
                    return cost_analysis["totals"].get("total_with_visits", 0)
                return 0

            costs = [get_cost_analysis_total(r) for r in results if get_cost_analysis_total(r) > 0]
            avg_cost = sum(costs) / len(costs) if costs else 0

            # Update progress bar with current stats
            postfix_dict: dict[str, object] = {
                'accuracy': f'{accuracy_pct:.1f}%',
                'last': f'{case_id}:{status}'
            }
            if avg_cost > 0:
                postfix_dict['avg_cost'] = f'${avg_cost:.0f}'

            pbar.set_postfix(ordered_dict=postfix_dict)  # pyright: ignore[reportUnknownMemberType]
            pbar.update(1)

            if not result["correct"] and "model_diagnosis" in result and "Error" in str(result["model_diagnosis"]):
                tqdm.write(f"Error with case {case_id}: {result['model_diagnosis']}")
                # Don't print traceback to avoid cluttering progress bar

        pbar.close()
