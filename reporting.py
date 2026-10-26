#!/usr/bin/env python3
"""
Results processor for SDBench harness.

This module handles summary generation, cost analysis aggregation, 
and HTML report creation from raw diagnostic results stored in JSON files.
"""

import datetime
import os
import json
import glob
import html
from pathlib import Path
from typing import List, Dict, Any, Optional

import case_data
from config import HarnessConfig
from azure_storage import write_output_file


def determine_common_method(results: List[Dict[str, Any]]) -> str:
    """
    Determine the most common costing method across all results.

    This is used to normalize legacy cost_analysis keys that don't have
    a method suffix. We look at the "method" field in cost_analysis dicts
    across all results and pick the most common one.

    Args:
        results: List of results to analyze

    Returns:
        The most common method name (e.g., 'llm_only', 'unified_cpt_llm', 'batch_llm_only')
    """
    from collections import Counter

    methods = []
    for result in results:
        # Check cost_analysis for method field
        cost_analysis = result.get("cost_analysis", {})
        if cost_analysis and isinstance(cost_analysis, dict):
            method = cost_analysis.get("method")
            if method:
                methods.append(method)

        # Also check cost_analysis_actual_workup for method field
        cost_analysis_actual_workup = result.get("cost_analysis_actual_workup", {})
        if cost_analysis_actual_workup and isinstance(cost_analysis_actual_workup, dict):
            method = cost_analysis_actual_workup.get("method")
            if method:
                methods.append(method)

    if methods:
        # Return the most common method
        counter = Counter(methods)
        most_common_method = counter.most_common(1)[0][0]
        return most_common_method
    else:
        # Default to llm_only if no methods found
        return "llm_only"


def normalize_cost_analysis_keys(result: Dict[str, Any], common_method: str = "llm_only") -> Dict[str, Any]:
    """
    Normalize cost analysis keys to new format for backward compatibility.

    Maps legacy keys to new method-specific keys:
    - cost_analysis -> cost_analysis_{common_method} (based on most common method across all results)
    - cost_analysis_actual_workup -> cost_analysis_actual_workup_{common_method}
    - cost_analysis_kp -> cost_analysis_unified_cpt_llm (KP was CPT-based)
    - cost_analysis_actual_workup_kp -> cost_analysis_actual_workup_unified_cpt_llm

    Args:
        result: Result dict to normalize
        common_method: The most common method to use for normalization (determined from all results)

    Returns:
        Normalized result dict with method-specific cost analysis keys
    """
    result = result.copy()

    # Backward compatibility: the real-world workup cost was previously stored under
    # 'cost_analysis_true*'; migrate any legacy keys to 'cost_analysis_actual_workup*'.
    for _legacy_key in [k for k in list(result) if k == "cost_analysis_true" or k.startswith("cost_analysis_true_")]:
        result.setdefault(_legacy_key.replace("cost_analysis_true", "cost_analysis_actual_workup", 1), result.pop(_legacy_key))

    # Map legacy cost_analysis to new format based on common method
    if "cost_analysis" in result:
        # Check if there's already a method-specific key (not KP)
        has_method_specific = any(
            k.startswith("cost_analysis_") and
            k not in ("cost_analysis_kp", "cost_analysis_actual_workup", "cost_analysis_actual_workup_kp")
            for k in result.keys()
        )

        if not has_method_specific:
            cost_analysis = result["cost_analysis"]
            new_key = f"cost_analysis_{common_method}"
            result[new_key] = cost_analysis
            # Keep the old key for backward compatibility in case any code still uses it

    # Map legacy cost_analysis_actual_workup to new format based on common method
    if "cost_analysis_actual_workup" in result:
        # Check if there's already a method-specific true key (not KP)
        has_method_specific_true = any(
            k.startswith("cost_analysis_actual_workup_") and
            k != "cost_analysis_actual_workup_kp"
            for k in result.keys()
        )

        if not has_method_specific_true:
            cost_analysis_actual_workup = result["cost_analysis_actual_workup"]
            new_key = f"cost_analysis_actual_workup_{common_method}"
            result[new_key] = cost_analysis_actual_workup
            # Keep the old key for backward compatibility

    # Map legacy KP costs to unified_cpt_llm (KP was CPT-based)
    if "cost_analysis_kp" in result:
        result["cost_analysis_unified_cpt_llm"] = result["cost_analysis_kp"]
        # Keep the old key for backward compatibility

    if "cost_analysis_actual_workup_kp" in result:
        result["cost_analysis_actual_workup_unified_cpt_llm"] = result["cost_analysis_actual_workup_kp"]
        # Keep the old key for backward compatibility

    return result


def get_cost_analysis_methods(results: List[Dict[str, Any]]) -> List[str]:
    """
    Get all unique costing methods used in the results.

    Returns:
        List of method names (e.g., ['unified_cpt_llm', 'llm_only'])
    """
    methods = set()
    for result in results:
        for key in result.keys():
            if key.startswith("cost_analysis_") and not key.startswith("cost_analysis_actual_workup_") and not key.startswith("cost_analysis_true_"):
                method = key.replace("cost_analysis_", "")
                # Exclude legacy "true" key (cost_analysis_actual_workup without method suffix)
                if method not in ("true", "kp", "actual_workup"):
                    methods.add(method)
    return sorted(list(methods))


def get_cost_analyses_for_method(results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    """
    Get all cost analyses for a specific method.

    Args:
        results: List of results
        method: Costing method name (e.g., 'unified_cpt_llm', 'llm_only')

    Returns:
        List of cost analysis dicts for the given method
    """
    analyses = []
    key = f"cost_analysis_{method}"
    for result in results:
        if key in result:
            analyses.append(result[key])
    return analyses


def get_true_cost_analyses_for_method(results: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    """
    Get all true cost analyses for a specific method.

    Args:
        results: List of results
        method: Costing method name (e.g., 'unified_cpt_llm', 'llm_only')

    Returns:
        List of true cost analysis dicts for the given method
    """
    analyses = []
    key = f"cost_analysis_actual_workup_{method}"
    for result in results:
        if key in result:
            analyses.append(result[key])
    return analyses


def get_method_display_name(method: str) -> str:
    """
    Get short display name for a costing method.

    Args:
        method: Costing method name (e.g., 'unified_cpt_llm', 'llm_only', 'batch_llm_only')

    Returns:
        Short display name (e.g., 'CPT', 'LLM', 'BatchLLM')
    """
    method_short_names = {
        "unified_cpt_llm": "CPT",
        "llm_only": "LLM",
        "batch_llm_only": "BatchLLM"
    }
    return method_short_names.get(method, method.replace("_", " ").title())


def load_results_from_json(json_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load all results from JSON files in the given directory."""
    results : List[Dict[str, Any]] = []
    json_files = glob.glob(os.path.join(json_dir, "case_k*.json"))

    # Sort files for consistent ordering
    json_files.sort()

    # Apply limit if specified
    if limit is not None and limit > 0:
        json_files = json_files[:limit]

    # First pass: load all results without normalization
    raw_results = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                raw_results.append(result)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load {json_file}: {e}")

    # Determine the most common method across all results
    common_method = determine_common_method(raw_results)

    # Second pass: normalize all results using the common method
    for result in raw_results:
        normalized_result = normalize_cost_analysis_keys(result, common_method)
        results.append(normalized_result)

    return results


def generate_summary_file(results: List[Dict[str, Any]], base_dir: str, config: HarnessConfig, report_text: str = "") -> str:
    """Generate a text summary file from results."""
    correct = [r for r in results if r.get("correct", False)]
    summary_file = f"{base_dir}/summary.txt"
    
    parallel_mode = config.max_workers > 1
    
    mode_str = "Parallel" if parallel_mode else "Sequential"
    summary_content = f"{mode_str} processing completed at {datetime.datetime.now()}\n"
    summary_content += f"Total cases processed: {len(results)}\n"
    summary_content += f"Correct diagnoses rate: {len(correct)}/{len(results)}\n"
    summary_content += f"Model: {config.model}\n"
    summary_content += f"Agent: {config.agent}\n"
    summary_content += f"Test runner: BatchTestRunner\n"
    summary_content += f"Case set: {config.case_set}\n"
    summary_content += f"Max turns: {config.sdbench.max_turns}\n"
    summary_content += f"Workers: {config.max_workers}\n"
    
    # Calculate test statistics from results
    all_requested_tests : list[str] = []
    for result in results:
        if 'requested_tests' in result:
            all_requested_tests.extend(result['requested_tests'])
    summary_content += f"Total test requests tracked: {len(all_requested_tests)}\n"
    summary_content += f"Unique tests requested: {len(set(all_requested_tests)) if all_requested_tests else 0}\n"
    
    if config.sdbench.compute_costs:
        summary_content += f"Cost analysis: Enabled\n"
    else:
        summary_content += f"Cost analysis: Disabled\n"

    if report_text:
        summary_content += "\n" + report_text + "\n"

    with open(summary_file, 'w') as f:
        f.write(summary_content)

    if config.azure_storage_enabled:
        # The last 3 parts of the path, are folder/summary.txt, and will be replicated in the
        # Azure blob results folder.
        blob_filename = str(Path(*Path(summary_file).parts[-2:]))
        blob_url = write_output_file(config, blob_filename, summary_content)
        if blob_url:
            print(f"Text summary uploaded to {blob_url}")
        else:
            print(f"Error: Failed to upload text summary to Azure Storage")

    return summary_file


def print_console_summary(summary_file: str, results: List[Dict[str, Any]]) -> None:
    """Print summary to console."""
    correct = [r for r in results if r.get("correct", False)]
    
    print(f"Summary saved to {summary_file}")
    print("\nSummary:")
    print("-" * 50)
    with open(summary_file, 'r') as f:
        print(f.read())
    print("-" * 50)
    print(f"All processing completed. {len(correct)}/{len(results)} correct diagnoses.")


def generate_cost_analysis(results: List[Dict[str, Any]], config: HarnessConfig) -> None:
    """Generate and print cost analysis if enabled."""
    if not config.sdbench.compute_costs:
        print("\nCost analysis disabled (use without --no-compute-costs to enable).")
        return

    print("\nGenerating cost estimation for all requested tests...")

    # Get all costing methods present in results
    methods = get_cost_analysis_methods(results)

    if not methods:
        print("No cost analyses found in results.")
        return

    # Collect all requested tests from results
    all_requested_tests : list[str] = []
    for result in results:
        if 'requested_tests' in result:
            all_requested_tests.extend(result['requested_tests'])

    total_unique_tests_across_cases = len(set(all_requested_tests)) if all_requested_tests else 0
    total_requests_across_cases = len(all_requested_tests)

    # Process cost analysis for each method
    for method in methods:
        all_cost_analyses = get_cost_analyses_for_method(results, method)
        all_cost_analyses_true = get_true_cost_analyses_for_method(results, method)

        if not all_cost_analyses:
            continue

        # Method name for display (use short names)
        method_display = get_method_display_name(method)

        total_cases_with_costs = len(all_cost_analyses)
        total_rounds_across_cases = sum(analysis.get('num_rounds', 1) for analysis in all_cost_analyses)

        # Aggregate totals from individual cases
        aggregate_tests_cost = sum(analysis['totals']['tests_only'] for analysis in all_cost_analyses if 'totals' in analysis)
        aggregate_visits_cost = sum(analysis['totals']['clinician_visits'] for analysis in all_cost_analyses if 'totals' in analysis)
        aggregate_total_cost = sum(analysis['totals']['total_with_visits'] for analysis in all_cost_analyses if 'totals' in analysis)

        # Calculate test method breakdown (for individual test pricing methods within the overall costing method)
        total_cpt_tests = 0
        total_llm_tests = 0
        total_failed_tests = 0
        total_costed_tests = 0

        for analysis in all_cost_analyses:
            cost_details = analysis.get('cost_details', [])
            for detail in cost_details:
                test_method = detail.get('method', 'unknown')
                if test_method == 'failure':
                    total_failed_tests += 1
                elif test_method == 'cpt_lookup':
                    total_cpt_tests += 1
                    total_costed_tests += 1
                elif test_method in ('llm_fallback', 'llm_only', 'batch_llm_only'):
                    total_llm_tests += 1
                    total_costed_tests += 1

        print("\n" + "="*60)
        print(f"AGGREGATE COST SUMMARY ({method_display})")
        print("="*60)
        print(f"Cases with cost analysis: {total_cases_with_costs}")
        print(f"Total unique tests across all cases: {total_unique_tests_across_cases}")
        print(f"Total test requests across all cases: {total_requests_across_cases}")
        print(f"Costed tests: {total_costed_tests}")
        print(f"Test pricing methods - CPT: {total_cpt_tests}, LLM: {total_llm_tests}, Failed: {total_failed_tests}")
        print(f"Total clinician visits across all cases: {total_rounds_across_cases}")
        print(f"\nCost Analysis:")
        print(f"  Aggregate cost - Tests only: ${aggregate_tests_cost:,.0f}")
        print(f"  Aggregate cost - Clinician visits: ${aggregate_visits_cost:,.0f}")
        print(f"  Aggregate total cost: ${aggregate_total_cost:,.0f}")
        print(f"\n  Per-Case Averages:")
        print(f"    Average total cost per case: ${round(aggregate_total_cost / total_cases_with_costs) if total_cases_with_costs else 0:,}")
        print(f"    Average test cost per case: ${round(aggregate_tests_cost / total_cases_with_costs) if total_cases_with_costs else 0:,}")
        print(f"    Average visit cost per case: ${round(aggregate_visits_cost / total_cases_with_costs) if total_cases_with_costs else 0:,}")
        print(f"    Average visits per case: {total_rounds_across_cases / total_cases_with_costs if total_cases_with_costs else 0:.2f}")
        print("="*60)

    print("Individual case cost analyses are saved in each case's JSON file.")



def generate_html_overview(results: List[Dict[str, Any]], html_dir: str, config: HarnessConfig, case_set: Optional[str] = None) -> str:
    """Generate an HTML overview of all results."""
    html_file = f"{html_dir}/overview.html"

    correct = [r for r in results if r.get("correct", False)]
    accuracy = len(correct) / len(results) if results else 0

    # Calculate ICU statistics
    icu_cases = [r for r in results if r.get("icu_metadata", {}).get("current_admission_icu", False)]
    non_icu_cases = [r for r in results if not r.get("icu_metadata", {}).get("current_admission_icu", False)]
    icu_correct = [r for r in icu_cases if r.get("correct", False)]
    non_icu_correct = [r for r in non_icu_cases if r.get("correct", False)]

    icu_accuracy = len(icu_correct) / len(icu_cases) if icu_cases else 0
    non_icu_accuracy = len(non_icu_correct) / len(non_icu_cases) if non_icu_cases else 0
    has_icu_metadata = any(r.get("icu_metadata") for r in results)

    # Get all costing methods present in results
    methods = get_cost_analysis_methods(results)

    # Calculate total costs for ALL methods
    method_stats = {}  # method -> {agent: {...}, true: {...}}

    for method in methods:
        cost_key = f"cost_analysis_{method}"
        cost_true_key = f"cost_analysis_actual_workup_{method}"

        # Initialize stats for this method
        agent_stats = {
            'total_cost': 0,
            'cases_with_costs': 0,
            'total_tests': 0,
            'cpt_tests': 0,
            'llm_tests': 0,
            'failed_tests': 0,
        }

        true_stats = {
            'total_cost': 0,
            'cases_with_costs': 0,
            'total_tests': 0,
            'cpt_tests': 0,
            'llm_tests': 0,
            'failed_tests': 0,
        }

        # Calculate agent cost stats
        for r in results:
            cost_analysis = r.get(cost_key, {})
            if cost_analysis:
                agent_stats['total_cost'] += cost_analysis.get("totals", {}).get("total_with_visits", 0)
                agent_stats['cases_with_costs'] += 1

                # Count test method breakdown
                cost_details = cost_analysis.get("cost_details", [])
                for detail in cost_details:
                    agent_stats['total_tests'] += 1
                    test_method = detail.get("method", "unknown")
                    if test_method == "failure":
                        agent_stats['failed_tests'] += 1
                    elif test_method == "cpt_lookup":
                        agent_stats['cpt_tests'] += 1
                    elif test_method in ("llm_fallback", "llm_only", "batch_llm_only"):
                        agent_stats['llm_tests'] += 1

        # Calculate true cost stats
        for r in results:
            cost_analysis_actual_workup = r.get(cost_true_key, {})
            if cost_analysis_actual_workup:
                true_stats['total_cost'] += cost_analysis_actual_workup.get("totals", {}).get("total_with_visits", 0)
                true_stats['cases_with_costs'] += 1

                # Count test method breakdown
                cost_details = cost_analysis_actual_workup.get("cost_details", [])
                for detail in cost_details:
                    true_stats['total_tests'] += 1
                    test_method = detail.get("method", "unknown")
                    if test_method == "failure":
                        true_stats['failed_tests'] += 1
                    elif test_method == "cpt_lookup":
                        true_stats['cpt_tests'] += 1
                    elif test_method in ("llm_fallback", "llm_only", "batch_llm_only"):
                        true_stats['llm_tests'] += 1

        method_stats[method] = {
            'agent': agent_stats,
            'true': true_stats
        }

    # Add case set info to title if filtering
    title_suffix = f" - {case_set}" if case_set else ""
    
    # Determine dataset name for display
    dataset_display = config.dataset.upper()
    
    html_content : str = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_display} Diagnostic Results - {html.escape(config.agent)}{title_suffix}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .accuracy {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ text-align: center; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }}
        
        /* Table styles */
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 5px;
            overflow: hidden;
        }}
        .results-table th {{
            background-color: #34495e;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: bold;
            font-size: 14px;
        }}
        .results-table td {{
            padding: 12px 8px;
            border-bottom: 1px solid #ecf0f1;
            vertical-align: top;
        }}
        .case-row.correct {{
            background-color: #f8fff8;
        }}
        .case-row.incorrect {{
            background-color: #fff8f8;
        }}
        .case-row:hover {{
            background-color: #f5f5f5;
        }}
        .status-cell {{
            text-align: center;
            font-size: 16px;
            width: 60px;
        }}
        .case-id {{
            font-weight: bold;
            width: 80px;
        }}
        .case-title {{
            font-weight: bold;
            min-width: 200px;
        }}
        .icu-cell {{
            text-align: center;
            width: 60px;
            font-size: 11px;
        }}
        .icu-badge {{
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 10px;
        }}
        .icu-badge.yes {{
            background-color: #e74c3c;
            color: white;
        }}
        .icu-badge.no {{
            background-color: #95a5a6;
            color: white;
        }}
        .diagnosis-cell {{
            max-width: 300px;
            word-wrap: break-word;
            white-space: normal;
            font-size: 13px;
            line-height: 1.3;
        }}
        .score-cell {{
            text-align: center;
            font-weight: bold;
            width: 100px;
        }}
        .tests-cell {{
            text-align: center;
            width: 140px;
        }}
        .cost-cell {{
            text-align: right;
            font-family: monospace;
            width: 100px;
        }}
        .actions-cell {{
            text-align: center;
            width: 240px;
        }}
        .action-btn {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 6px 12px;
            margin: 2px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        .action-btn:hover {{
            background-color: #2980b9;
        }}
        .grader-btn {{
            background-color: #9b59b6;
        }}
        .grader-btn:hover {{
            background-color: #8e44ad;
        }}
        .runner-btn {{
            background-color: #e67e22;
        }}
        .runner-btn:hover {{
            background-color: #d35400;
        }}
        .cost-btn {{
            background-color: #27ae60;
        }}
        .cost-btn:hover {{
            background-color: #229954;
        }}
        .cost-true-btn {{
            background-color: #16a085;
        }}
        .cost-true-btn:hover {{
            background-color: #138d75;
        }}
        .gatekeeper-btn {{
            background-color: #2c3e50;
        }}
        .gatekeeper-btn:hover {{
            background-color: #1a252f;
        }}
        .detail-row {{
            background-color: #f9f9f9;
        }}
        .detail-container {{
            padding: 20px;
        }}
        
        /* Cost Analysis Layout Styles */
        .cost-grid-single {{
            display: block;
        }}
        .cost-panel {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            background-color: #f9f9f9;
        }}
        .cost-panel h5 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}
        
        /* Dialog content styles */
        .content {{
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .content h4 {{
            background-color: #ecf0f1;
            margin: 0;
            padding: 10px 15px;
            border-bottom: 1px solid #ddd;
        }}
        .dialog-section {{
            padding: 15px;
            max-height: 400px;
            overflow-y: auto;
        }}
        
        /* Message styles */
        .message {{
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #ddd;
        }}
        .message.user {{ background-color: #e3f2fd; border-left-color: #2196f3; }}
        .message.assistant {{ background-color: #f3e5f5; border-left-color: #9c27b0; }}
        .message.system {{ background-color: #fff3e0; border-left-color: #ff9800; }}
        .message-role {{ font-weight: bold; margin-bottom: 5px; text-transform: capitalize; }}
        .message-content {{ white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 12px; }}
        .tool-calls {{ background-color: #e8f5e8; padding: 10px; border-radius: 3px; margin: 5px 0; }}
        .tool-call {{ margin-bottom: 10px; }}
        .tool-name {{ font-weight: bold; color: #2e7d32; }}
        .tool-args {{ font-family: 'Courier New', monospace; font-size: 11px; color: #666; }}
        .test-results {{ background-color: #fff9c4; padding: 10px; border-radius: 3px; margin: 5px 0; }}
        .reasoning {{ background-color: #f0f8ff; padding: 10px; border-radius: 3px; margin: 10px 0; }}
        .reasoning-label {{ font-weight: bold; color: #1976d2; margin-bottom: 5px; }}
        .reasoning-content {{ font-size: 14px; line-height: 1.4; }}
        
        /* Test Runner Log styles */
        
        /* Test Execution styles */
        .test-execution {{
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
        }}
        .test-execution-header {{
            background-color: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            color: #2c3e50;
        }}
        .test-reason {{
            font-size: 14px;
            color: #666;
            font-weight: normal;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{dataset_display} Diagnostic Results{title_suffix}</h1>
        <p>Agent: {html.escape(config.agent)} | Model: {html.escape(config.model)} | Case Set: {html.escape(case_set if case_set else config.case_set)}</p>
    </div>
    
    <div class="summary">
        <h2>Overall Performance</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="accuracy">{accuracy:.1%}</div>
                <div>Accuracy</div>
            </div>
            <div class="stat-box">
                <div class="accuracy">{len(correct)}/{len(results)}</div>
                <div>Correct Diagnoses</div>
            </div>
            <div class="stat-box">
                <div class="accuracy">{len(results)}</div>
                <div>Total Cases</div>
            </div>
"""

    # Add cost stat boxes for each method
    # Sort methods to ensure unified_cpt_llm appears first if present
    sorted_methods = sorted(methods, key=lambda m: (m != "unified_cpt_llm", m))

    # Show all Agent Cost panels first
    for method in sorted_methods:
        stats = method_stats.get(method, {})
        agent_stats = stats.get('agent', {})
        method_display = get_method_display_name(method)

        # Determine icon based on method
        icon = "📚" if method == "unified_cpt_llm" else "🤖"

        # Agent Cost Panel (only if agent costs exist)
        if agent_stats.get('cases_with_costs', 0) > 0:
            total_cost = agent_stats['total_cost']
            cases = agent_stats['cases_with_costs']
            avg_cost = round(total_cost / cases) if cases else 0

            all_tests = agent_stats['total_tests']
            cpt_tests = agent_stats['cpt_tests']
            llm_tests = agent_stats['llm_tests']
            failed_tests = agent_stats['failed_tests']

            # Build breakdown string: "All Tests: N [ CPT / LLM : N / N ]"
            breakdown_parts = []
            if cpt_tests > 0:
                breakdown_parts.append(f"CPT")
            if llm_tests > 0:
                breakdown_parts.append(f"LLM")

            breakdown_counts = []
            if cpt_tests > 0:
                breakdown_counts.append(str(cpt_tests))
            if llm_tests > 0:
                breakdown_counts.append(str(llm_tests))

            if breakdown_parts:
                breakdown_str = f"All Tests: {all_tests} [ {' / '.join(breakdown_parts)} : {' / '.join(breakdown_counts)} ]"
            else:
                breakdown_str = f"All Tests: {all_tests}"

            # Add failure info if present
            if failed_tests > 0:
                breakdown_str += f' <strong style="color: #c0392b;">⚠️ {failed_tests} Failed</strong>'

            html_content += f"""
            <div class="stat-box">
                <div class="accuracy">${total_cost:,.0f}</div>
                <div>📊 Agent Cost ({icon} {method_display})</div>
                <div><small style="color: #7f8c8d;">{cases} costed cases • <strong style="color: #2c3e50; font-size: 1.15em;">${avg_cost:,.0f}/case</strong></small></div>
                <div><small style="color: #7f8c8d;">{breakdown_str}</small></div>
            </div>
"""

    # Then show all Actual Workup Cost panels
    for method in sorted_methods:
        stats = method_stats.get(method, {})
        true_stats = stats.get('true', {})
        method_display = get_method_display_name(method)

        # Determine icon based on method
        icon = "📚" if method == "unified_cpt_llm" else "🤖"

        # Actual Workup Cost Panel (only if true costs exist)
        if true_stats.get('cases_with_costs', 0) > 0:
            total_cost = true_stats['total_cost']
            cases = true_stats['cases_with_costs']
            avg_cost = round(total_cost / cases) if cases else 0

            all_tests = true_stats['total_tests']
            cpt_tests = true_stats['cpt_tests']
            llm_tests = true_stats['llm_tests']
            failed_tests = true_stats['failed_tests']

            # Build breakdown string: "All Tests: N [ CPT / LLM : N / N ]"
            breakdown_parts = []
            if cpt_tests > 0:
                breakdown_parts.append(f"CPT")
            if llm_tests > 0:
                breakdown_parts.append(f"LLM")

            breakdown_counts = []
            if cpt_tests > 0:
                breakdown_counts.append(str(cpt_tests))
            if llm_tests > 0:
                breakdown_counts.append(str(llm_tests))

            if breakdown_parts:
                breakdown_str = f"All Tests: {all_tests} [ {' / '.join(breakdown_parts)} : {' / '.join(breakdown_counts)} ]"
            else:
                breakdown_str = f"All Tests: {all_tests}"

            # Add failure info if present
            if failed_tests > 0:
                breakdown_str += f' <strong style="color: #c0392b;">⚠️ {failed_tests} Failed</strong>'

            html_content += f"""
            <div class="stat-box">
                <div class="accuracy">${total_cost:,.0f}</div>
                <div>💰 Actual Workup Cost ({icon} {method_display})</div>
                <div><small style="color: #7f8c8d;">{cases} costed cases • <strong style="color: #2c3e50; font-size: 1.15em;">${avg_cost:,.0f}/case</strong></small></div>
                <div><small style="color: #7f8c8d;">{breakdown_str}</small></div>
            </div>
"""

    html_content += f"""
        </div>
    </div>
"""

    # Add ICU statistics section if ICU metadata is available
    if has_icu_metadata:
        html_content += f"""
    <div class="summary">
        <h2>ICU Status Breakdown</h2>
        <div class="stats">
            <div class="stat-box" style="background-color: #ffe6e6;">
                <div class="accuracy">{len(icu_cases)}</div>
                <div>ICU Cases</div>
                <div style="margin-top: 5px;"><small>Accuracy: {icu_accuracy:.1%}</small></div>
            </div>
            <div class="stat-box" style="background-color: #e6f3ff;">
                <div class="accuracy">{len(non_icu_cases)}</div>
                <div>Non-ICU Cases</div>
                <div style="margin-top: 5px;"><small>Accuracy: {non_icu_accuracy:.1%}</small></div>
            </div>
            <div class="stat-box">
                <div class="accuracy">{len(icu_correct)}/{len(icu_cases)}</div>
                <div>ICU Correct</div>
            </div>
            <div class="stat-box">
                <div class="accuracy">{len(non_icu_correct)}/{len(non_icu_cases)}</div>
                <div>Non-ICU Correct</div>
            </div>
        </div>
    </div>
"""

    html_content += f"""
    <div class="summary">
        <h2>Case Results</h2>
        <div class="table-container">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Case ID</th>"""

    # Add ICU column if ICU metadata is available
    if has_icu_metadata:
        html_content += """
                        <th>ICU</th>"""

    html_content += """
                        <th>Case Title</th>
                        <th>Model Diagnosis</th>
                        <th>Actual Diagnosis</th>
                        <th>Grader Score</th>
                        <th>Tests (Synthetic/Real)</th>
                        <th>Cost</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add individual case rows with detailed information
    for _, result in enumerate(results):
        is_correct = result.get("correct", False)
        case_id = result.get("case_id", "Unknown")
        case_title = result.get("case_title", f"Case {case_id}")
        model_diagnosis = result.get("model_diagnosis", "No diagnosis")
        actual_diagnosis = result.get("actual_diagnosis", "Unknown")
        model_reasoning = result.get("model_reasoning", "")

        # Get grader information if available
        grader_score = result.get("diagnosis_score", "N/A")
        grader_messages = result.get("check_diagnosis_messages", [])

        # Get new simple test logs
        test_logs = result.get("test_logs", [])

        # Count synthetic vs non-synthetic tests
        synthetic_count = sum(1 for log in test_logs if log.get("is_synthetic", False))
        real_count = len(test_logs) - synthetic_count

        # Get test and cost information for ALL methods
        requested_tests = result.get("requested_tests", [])
        num_tests = len(requested_tests)

        # Get all cost analysis methods available for this result
        result_methods = []
        for key in result.keys():
            # Skip legacy keys without method suffix
            if key in ("cost_analysis", "cost_analysis_actual_workup", "cost_analysis_kp", "cost_analysis_actual_workup_kp"):
                continue
            if key.startswith("cost_analysis_") and not key.startswith("cost_analysis_actual_workup_") and not key.startswith("cost_analysis_true_"):
                method = key.replace("cost_analysis_", "")
                result_methods.append(method)

        # Sort methods: unified first, then others
        def method_sort_key(m):
            if m == "unified_cpt_llm":
                return 0
            elif m == "llm_only":
                return 1
            elif m == "batch_llm_only":
                return 2
            else:
                return 3

        result_methods.sort(key=method_sort_key)

        # Build cost display with all methods
        cost_display_lines = []
        cost_display_lines.append('<div style="font-size: 0.85em; color: #7f8c8d; margin-bottom: 0;">Agent Cost</div>')

        for method in result_methods:
            cost_key = f"cost_analysis_{method}"
            cost_analysis = result.get(cost_key, {})
            total_cost = cost_analysis.get("totals", {}).get("total_with_visits", 0) if cost_analysis else 0

            # Check for failures
            cost_details = cost_analysis.get("cost_details", [])
            failed_tests = [d for d in cost_details if d.get("method") == "failure"]
            has_failures = len(failed_tests) > 0

            # Get display name and icon
            method_display = get_method_display_name(method)
            icon = "📚" if method == "unified_cpt_llm" else "🤖"

            cost_line = f'<div style="white-space: nowrap;">{icon} {method_display}: ${total_cost:,.0f}'
            if has_failures:
                cost_line += f' <span style="color: #c0392b;">⚠️{len(failed_tests)}</span>'
            cost_line += '</div>'

            cost_display_lines.append(cost_line)

        # Add horizontal rule
        cost_display_lines.append('<hr style="margin: 5px 0; border: none; border-top: 1px solid #ddd;">')

        # True costs section
        has_true_costs = any(f"cost_analysis_actual_workup_{m}" in result for m in result_methods)

        if has_true_costs:
            cost_display_lines.append('<div style="font-size: 0.85em; color: #7f8c8d; margin-bottom: 0;">Actual Workup Cost</div>')

            for method in result_methods:
                cost_true_key = f"cost_analysis_actual_workup_{method}"
                cost_analysis_actual_workup = result.get(cost_true_key, {})
                total_cost_true = cost_analysis_actual_workup.get("totals", {}).get("total_with_visits", 0) if cost_analysis_actual_workup else 0

                if total_cost_true > 0:
                    # Check for failures in true costs
                    cost_true_details = cost_analysis_actual_workup.get("cost_details", [])
                    failed_true_tests = [d for d in cost_true_details if d.get("method") == "failure"]
                    has_true_failures = len(failed_true_tests) > 0

                    method_display = get_method_display_name(method)
                    icon = "📚" if method == "unified_cpt_llm" else "🤖"

                    cost_line = f'<div style="white-space: nowrap;">💰{icon} {method_display}: ${total_cost_true:,.0f}'
                    if has_true_failures:
                        cost_line += f' <span style="color: #c0392b;">⚠️{len(failed_true_tests)}</span>'
                    cost_line += '</div>'

                    cost_display_lines.append(cost_line)
        else:
            cost_display_lines.append('<div style="font-size: 0.85em; color: #7f8c8d;">No true costs computed</div>')

        cost_display = ''.join(cost_display_lines) if cost_display_lines else "N/A"

        status_icon = "✅" if is_correct else "❌"
        row_class = "correct" if is_correct else "incorrect"

        # Prepare ICU badge if ICU metadata is available
        icu_badge = ""
        if has_icu_metadata:
            icu_metadata = result.get("icu_metadata", {})
            has_icu = icu_metadata.get("current_admission_icu", False)
            icu_badge = f'<span class="icu-badge yes">ICU</span>' if has_icu else f'<span class="icu-badge no">No</span>'

        html_content += f"""
                    <tr class="case-row {row_class}">
                        <td class="status-cell">{status_icon}</td>
                        <td class="case-id">{case_id}</td>"""

        # Add ICU cell if ICU metadata is available
        if has_icu_metadata:
            html_content += f"""
                        <td class="icu-cell">{icu_badge}</td>"""

        html_content += f"""
                        <td class="case-title">{html.escape(case_title)}</td>
                        <td class="diagnosis-cell">{html.escape(model_diagnosis)}</td>
                        <td class="diagnosis-cell">{html.escape(actual_diagnosis)}</td>
                        <td class="score-cell">{grader_score}</td>
                        <td class="tests-cell">{num_tests} (Synthetic: {synthetic_count}, Real: {real_count})</td>
                        <td class="cost-cell">{cost_display}</td>
                        <td class="actions-cell">
                            <button class="action-btn agent-btn collapsible" data-original-text="Agent Dialog">Agent Dialog</button>
                            <button class="action-btn grader-btn collapsible" data-original-text="Grader Dialog">Grader Dialog</button>
                            <button class="action-btn runner-btn collapsible" data-original-text="Test Runner Log">Test Runner Log</button>
"""


        # Add cost analysis buttons for each method available in this result
        result_methods_buttons = []
        for key in result.keys():
            # Skip legacy keys without method suffix
            if key in ("cost_analysis", "cost_analysis_actual_workup", "cost_analysis_kp", "cost_analysis_actual_workup_kp"):
                continue
            if key.startswith("cost_analysis_") and not key.startswith("cost_analysis_actual_workup_") and not key.startswith("cost_analysis_true_"):
                method = key.replace("cost_analysis_", "")
                result_methods_buttons.append(method)

        for method in result_methods_buttons:
            # Map method to short display name
            method_display = get_method_display_name(method)

            cost_key = f"cost_analysis_{method}"
            cost_true_key = f"cost_analysis_actual_workup_{method}"

            # Add model cost analysis button if exists
            if cost_key in result and result.get(cost_key):
                button_text = f"Agent Cost ({method_display})"
                html_content += f"""
                            <button class="action-btn cost-btn collapsible" data-method="{method}" data-original-text="{button_text}">{button_text}</button>
"""

            # Add true cost analysis button if exists
            if cost_true_key in result and result.get(cost_true_key):
                button_text = f"Actual Workup Cost ({method_display})"
                html_content += f"""
                            <button class="action-btn cost-true-btn collapsible" data-method="{method}" data-original-text="{button_text}">{button_text}</button>
"""

        # Add gatekeeper button if gatekeeper messages exist
        gatekeeper_messages = result.get("gatekeeper_messages", [])
        if gatekeeper_messages:
            html_content += """
                            <button class="action-btn gatekeeper-btn collapsible" data-original-text="Gatekeeper Dialog">Gatekeeper Dialog</button>
"""

        html_content += """
                        </td>
                    </tr>"""

        # Calculate colspan: base 9 + 1 if ICU
        detail_colspan = 9
        if has_icu_metadata:
            detail_colspan += 1

        html_content += f"""
                    <tr class="detail-row" style="display: none;">
                        <td colspan="{detail_colspan}">
                            <div class="detail-container">
"""
        
        # Add reasoning if available
        if model_reasoning:
            html_content += f"""
                                <div class="reasoning">
                                    <div class="reasoning-label">Model Reasoning:</div>
                                    <div class="reasoning-content">{html.escape(model_reasoning)}</div>
                                </div>
"""
        
        # Agent dialog section
        agent_messages = result.get("agent_messages", [])
        panel_discussions = result.get("panel_discussions", [])
        model_mapping = result.get("model_mapping", {})
        
        # Calculate total messages including panel discussions
        total_panel_responses = sum(len(discussion.get("panel_responses", [])) for discussion in panel_discussions)
        
        # Add panel info to title if applicable
        title_suffix = ""
        if panel_discussions:
            title_suffix = f" + {len(panel_discussions)} panel rounds ({total_panel_responses} member responses)"
        
        html_content += f"""
                                <div class="content agent-content" style="display: none;">
                                    <h4>Agent Dialog ({len(agent_messages)} messages{title_suffix})</h4>
                                    <div class="dialog-section">
"""
        
        # Show model configuration if this is a multi-agent panel
        if model_mapping:
            html_content += """
                        <div class="model-mapping" style="background-color: #f8f9fa; padding: 12px; border-radius: 5px; margin-bottom: 15px; border-left: 3px solid #3498db;">
                            <div style="font-weight: bold; margin-bottom: 8px;">🤖 Multi-Agent Panel Configuration:</div>
"""
            default_model = result.get("model", "Unknown")
            for member, model in model_mapping.items():
                html_content += f'<div style="margin: 2px 0; font-size: 13px;"><strong>{html.escape(member.title())}:</strong> {html.escape(model)}</div>'
            html_content += f'<div style="margin: 2px 0; font-size: 13px;"><strong>Default Model:</strong> {html.escape(default_model)}</div>'
            html_content += """
                        </div>
"""
        
        if agent_messages:
            # Create mapping of assistant messages to panel discussions for inline display
            panel_discussions_by_round = {d.get("round", i+1): d for i, d in enumerate(panel_discussions)}
            panel_round_counter = 1
            
            for _, message in enumerate(agent_messages):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                # Handle tool calls
                tool_calls_html = ""
                if message.get("tool_calls"):
                    tool_calls_html = '<div class="tool-calls">'
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")
                        tool_calls_html += f'''
                            <div class="tool-call">
                                <div class="tool-name">🔧 {html.escape(tool_name)}</div>
                                <div class="tool-args">{html.escape(tool_args)}</div>
                            </div>
                        '''
                    tool_calls_html += '</div>'
                
                # Handle tool responses
                tool_response_html = ""
                if role == "tool":
                    tool_name = message.get("name", "unknown_tool")
                    tool_response_html = f'<div class="test-results"><strong>🧪 {html.escape(tool_name)} Result:</strong></div>'
                
                html_content += f"""
                        <div class="message {role}">
                            <div class="message-role">{html.escape(role)}</div>
                            {tool_response_html}
                            <div class="message-content">{html.escape(content)}</div>
                            {tool_calls_html}
                        </div>
"""
                
                # Insert panel discussion after assistant messages (coordinator responses)
                if role == "assistant" and panel_round_counter <= len(panel_discussions):
                    discussion = panel_discussions_by_round.get(panel_round_counter)
                    if discussion:
                        panel_responses = discussion.get("panel_responses", [])
                        coordinator_model = discussion.get("coordinator_model", "Unknown")
                        
                        html_content += f"""
                        <div class="panel-discussion" style="margin: 15px 0; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background-color: #fafafa;">
                            <div class="panel-discussion-header" style="background-color: #34495e; color: white; padding: 10px 15px; font-weight: bold; font-size: 14px;">
                                🏛️ Panel Discussion Round {panel_round_counter}
                            </div>
                            <div class="panel-discussion-content" style="padding: 15px;">
"""
                        
                        # Display individual panel member responses
                        for response in panel_responses:
                            # Parse the response format: "**Dr. Name (model):** content"
                            if response.startswith("**") and ":**" in response:
                                header_end = response.find(":**")
                                header = response[2:header_end]  # Remove ** from start
                                content_text = response[header_end + 3:].strip()  # Remove :** and whitespace
                                
                                # Extract member name and model
                                if "(" in header and header.endswith(")"):
                                    paren_start = header.rfind("(")
                                    member_name = header[:paren_start].strip()
                                    member_model = header[paren_start+1:-1].strip()
                                else:
                                    member_name = header
                                    member_model = "Unknown"
                                
                                html_content += f"""
                                <div class="panel-member-response" style="margin-bottom: 12px; padding: 10px; background-color: white; border-left: 4px solid #3498db; border-radius: 4px;">
                                    <div class="panel-member-header" style="font-weight: bold; color: #2c3e50; margin-bottom: 6px; font-size: 13px;">
                                        {html.escape(member_name)} <span style="font-size: 11px; color: #7f8c8d; font-weight: normal;">({html.escape(member_model)})</span>
                                    </div>
                                    <div class="panel-member-content" style="line-height: 1.4; font-size: 13px;">
                                        {html.escape(content_text)}
                                    </div>
                                </div>
"""
                            else:
                                # Fallback for unexpected format
                                html_content += f"""
                                <div class="panel-member-response" style="margin-bottom: 12px; padding: 10px; background-color: white; border-left: 4px solid #95a5a6; border-radius: 4px;">
                                    <div class="panel-member-content" style="line-height: 1.4; font-size: 13px;">
                                        {html.escape(response)}
                                    </div>
                                </div>
"""
                        
                        # Display coordinator decision summary
                        html_content += f"""
                                <div class="coordinator-summary" style="margin-top: 10px; padding: 8px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; font-size: 12px;">
                                    <div style="font-weight: bold; color: #856404; margin-bottom: 4px;">
                                        👔 Coordinator Decision ({html.escape(coordinator_model)})
                                    </div>
                                    <div style="color: #856404;">
                                        ↑ Decision reflected in assistant message above
                                    </div>
                                </div>
"""
                        
                        html_content += """
                            </div>
                        </div>
"""
                        panel_round_counter += 1
        else:
            html_content += """
                        <p>No agent messages recorded for this case.</p>
"""
        
        html_content += """
                                    </div>
                                </div>
"""
        
        # Grader dialog section
        html_content += f"""
                                <div class="content grader-content" style="display: none;">
                                    <h4>Grader Dialog ({len(grader_messages)} messages)</h4>
                                    <div class="dialog-section">
"""
        
        if grader_messages:
            for _, message in enumerate(grader_messages):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                
                html_content += f"""
                        <div class="message {role}">
                            <div class="message-role">{html.escape(role)}</div>
                            <div class="message-content">{html.escape(content)}</div>
                        </div>
"""
        else:
            html_content += """
                        <p>No grader messages recorded for this case.</p>
"""
        
        html_content += """
                                    </div>
                                </div>
"""

        # Gatekeeper dialog section
        if gatekeeper_messages:
            # Calculate total messages across all conversations
            total_gatekeeper_messages = sum(len(conversation) for conversation in gatekeeper_messages)

            html_content += f"""
                                <div class="content gatekeeper-content" style="display: none;">
                                    <h4>Gatekeeper Dialog ({len(gatekeeper_messages)} conversations, {total_gatekeeper_messages} messages)</h4>
                                    <div class="dialog-section">
"""

            for conv_idx, conversation in enumerate(gatekeeper_messages, 1):
                html_content += f"""
                        <div class="test-execution">
                            <div class="test-execution-header">
                                <strong>Conversation {conv_idx}</strong> - {len(conversation)} messages
                            </div>
"""

                for message in conversation:
                    role = message.get("role", "unknown")
                    content = message.get("content", "")

                    html_content += f"""
                            <div class="message {role}">
                                <div class="message-role">{html.escape(role)}</div>
                                <div class="message-content">{html.escape(content)}</div>
                            </div>
"""

                html_content += """
                        </div>
"""

            html_content += """
                                    </div>
                                </div>
"""

        
        # Test Runner Log section
        html_content += f"""
                                <div class="content runner-content" style="display: none;">
                                    <h4>Test Execution Log ({len(test_logs)} tests)</h4>
                                    <div class="dialog-section">
"""
        
        if test_logs:
            for _, test_log in enumerate(test_logs):
                test_name = test_log.get("test_name", "Unknown Test")
                reason = test_log.get("reason", "No reason provided")
                is_synthetic = test_log.get("is_synthetic", False)
                timestamp = test_log.get("timestamp", "Unknown")
                messages = test_log.get("messages", [])
                
                synthetic_badge = " 🤖" if is_synthetic else " 🔬"
                
                # Display test header
                html_content += f"""
                        <div class="test-execution">
                            <div class="test-execution-header">
                                <strong>{html.escape(test_name)}</strong>{synthetic_badge} - {html.escape(timestamp)}
                                <div class="test-reason">Reason: {html.escape(reason)}</div>
                            </div>
"""
                
                # Display the messages (system, user, assistant) like agent/grader dialogs
                for _, message in enumerate(messages):
                    role = message.get("role", "unknown")
                    content = message.get("content", "")
                    
                    html_content += f"""
                            <div class="message {role}">
                                <div class="message-role">{html.escape(role)}</div>
                                <div class="message-content">{html.escape(content)}</div>
                            </div>
"""
                
                html_content += """
                        </div>
"""
        else:
            html_content += """
                        <p>No test execution logs available.</p>
"""
        
        html_content += """
                                    </div>
                                </div>
"""

        # Generate method-specific cost content divs for each costing method
        for method in result_methods:
            method_display = get_method_display_name(method)
            method_class = method.replace("_", "-")  # unified_cpt_llm -> unified-cpt-llm

            # Agent cost content div
            cost_key = f"cost_analysis_{method}"
            if cost_key in result and result.get(cost_key):
                cost_analysis_method = result[cost_key]
                cost_details = cost_analysis_method.get("cost_details", [])
                totals = cost_analysis_method.get("totals", {})

                html_content += f"""
                                <div class="content cost-content-{method_class}" style="display: none;">
                                    <h4>Agent Cost Analysis ({method_display})</h4>
                                    <div class="dialog-section">
"""
                if cost_details:
                    html_content += """
                        <div class="reasoning">
                            <div class="reasoning-label">Cost Details:</div>
                            <div class="reasoning-content">
"""
                    for detail in cost_details:
                        test_name = detail.get("test_name", "Unknown")
                        test_cost = detail.get("cost", 0) or 0
                        explanation = detail.get("explanation", "")
                        test_method = detail.get("method", "unknown")

                        # Style based on method - highlight failures in red
                        if test_method == "failure":
                            border_color = "#c0392b"
                            icon = "⚠️ "
                        else:
                            border_color = "#3498db"
                            icon = ""

                        html_content += f"""
                                <div style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-left: 3px solid {border_color};">
                                    <strong>{icon}{html.escape(test_name)}</strong>: ${test_cost:,.0f} <span style="font-size: 11px; color: #888; font-weight: normal;">({html.escape(test_method)})</span><br>
                                    <small style="color: #666;">{html.escape(explanation)}</small>
                                </div>
"""
                    html_content += """
                            </div>
                        </div>
"""

                if totals:
                    html_content += f"""
                        <div class="reasoning">
                            <div class="reasoning-label">Cost Summary:</div>
                            <div class="reasoning-content">
                                <p><strong>Total Tests Cost:</strong> ${totals.get('tests_only', 0):,}</p>
                                <p><strong>Clinician Visits Cost:</strong> ${totals.get('clinician_visits', 0):,}</p>
                                <p><strong>Total Cost:</strong> ${totals.get('total_with_visits', 0):,}</p>
                                <p><strong>Number of Rounds:</strong> {cost_analysis_method.get('num_rounds', 1)}</p>
                                <p><strong>Unique Tests:</strong> {cost_analysis_method.get('total_unique_tests', 0)}</p>
                            </div>
                        </div>
"""

                html_content += """
                                    </div>
                                </div>
"""

            # True cost content div
            cost_true_key = f"cost_analysis_actual_workup_{method}"
            if cost_true_key in result and result.get(cost_true_key):
                cost_analysis_actual_workup_method = result[cost_true_key]
                cost_true_details = cost_analysis_actual_workup_method.get("cost_details", [])
                totals_true = cost_analysis_actual_workup_method.get("totals", {})

                html_content += f"""
                                <div class="content cost-true-content-{method_class}" style="display: none;">
                                    <h4>Actual Workup Cost Analysis ({method_display})</h4>
                                    <div class="dialog-section">
"""
                if cost_true_details:
                    html_content += """
                        <div class="reasoning">
                            <div class="reasoning-label">Actual Workup Cost Details:</div>
                            <div class="reasoning-content">
"""
                    for detail in cost_true_details:
                        test_name = detail.get("test_name", "Unknown")
                        test_cost = detail.get("cost", 0) or 0
                        explanation = detail.get("explanation", "")
                        test_method = detail.get("method", "unknown")

                        # Style based on method - highlight failures in red
                        if test_method == "failure":
                            border_color = "#c0392b"
                            icon = "⚠️ "
                        else:
                            border_color = "#9b59b6"
                            icon = ""

                        html_content += f"""
                                <div style="margin-bottom: 10px; padding: 8px; background-color: #f8f9fa; border-left: 3px solid {border_color};">
                                    <strong>{icon}{html.escape(test_name)}</strong>: ${test_cost:,} <span style="font-size: 11px; color: #888; font-weight: normal;">({html.escape(test_method)})</span><br>
                                    <small style="color: #666;">{html.escape(explanation)}</small>
                                </div>
"""
                    html_content += """
                            </div>
                        </div>
"""

                if totals_true:
                    html_content += f"""
                        <div class="reasoning">
                            <div class="reasoning-label">Actual Workup Cost Summary:</div>
                            <div class="reasoning-content">
                                <p><strong>Total Tests Cost:</strong> ${totals_true.get('tests_only', 0):,}</p>
                                <p><strong>Clinician Visits Cost:</strong> ${totals_true.get('clinician_visits', 0):,}</p>
                                <p><strong>Total Cost:</strong> ${totals_true.get('total_with_visits', 0):,}</p>
                                <p><strong>Number of Rounds:</strong> {cost_analysis_actual_workup_method.get('num_rounds', 1)}</p>
                                <p><strong>Unique Tests:</strong> {cost_analysis_actual_workup_method.get('total_unique_tests', 0)}</p>
                            </div>
                        </div>
"""

                html_content += """
                                    </div>
                                </div>
"""

        html_content += """
                            </div>
                        </td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <script>
        // Helper function to reset all buttons in a row to their original text
        function resetAllButtons(parentRow) {
            var allButtons = parentRow.querySelectorAll('.action-btn[data-original-text]');
            allButtons.forEach(function(btn) {
                btn.textContent = btn.getAttribute('data-original-text');
            });
        }


        function toggleDialogContent(button, contentType) {
            // Find the parent row and the detail row
            var parentRow = button.closest('tr');
            var detailRow = parentRow.nextElementSibling;

            if (detailRow && detailRow.classList.contains('detail-row')) {
                // Show the detail row if hidden
                if (detailRow.style.display === 'none') {
                    detailRow.style.display = 'table-row';
                }

                // For cost buttons with data-method, construct method-specific selector
                var effectiveContentType = contentType;
                if ((contentType === 'cost' || contentType === 'cost-true') && button.hasAttribute('data-method')) {
                    var method = button.getAttribute('data-method');
                    // Convert method to selector format: unified_cpt_llm -> unified-cpt-llm
                    var methodClass = method.replace(/_/g, '-');
                    effectiveContentType = contentType + '-' + methodClass;
                }

                // Map content types to their selectors
                var contentMap = {
                    'agent': '.agent-content',
                    'grader': '.grader-content',
                    'gatekeeper': '.gatekeeper-content',
                    'runner': '.runner-content',
                    'panel': '.panel-content',
                    'cost': '.cost-section',
                    'cost-true': '.cost-true-content',
                };

                // For method-specific cost content, add to map dynamically
                if (effectiveContentType.startsWith('cost-') && effectiveContentType !== 'cost-true') {
                    contentMap[effectiveContentType] = '.cost-content-' + effectiveContentType.substring(5); // Remove 'cost-' prefix
                }
                if (effectiveContentType.startsWith('cost-true-')) {
                    contentMap[effectiveContentType] = '.cost-true-content-' + effectiveContentType.substring(10); // Remove 'cost-true-' prefix
                }

                // Also find ALL method-specific cost content divs in the detail row to ensure we hide them all
                var allCostContentDivs = detailRow.querySelectorAll('[class*="cost-content-"], [class*="cost-true-content-"]');

                // Get all content sections from the contentMap
                var allContents = Object.keys(contentMap).map(function(key) {
                    return detailRow.querySelector(contentMap[key]);
                }).filter(function(el) { return el !== null; });

                // Add all method-specific cost divs to allContents
                allCostContentDivs.forEach(function(div) {
                    if (allContents.indexOf(div) === -1) {
                        allContents.push(div);
                    }
                });

                // Get the target content
                var targetContent = detailRow.querySelector(contentMap[effectiveContentType]);

                if (targetContent) {
                    // Check if content is currently visible
                    var isCurrentlyVisible = targetContent.style.display !== 'none';

                    // Hide all content sections
                    allContents.forEach(function(content) {
                        content.style.display = 'none';
                    });

                    // Reset all buttons to original text
                    resetAllButtons(parentRow);

                    // Toggle the target content
                    if (isCurrentlyVisible) {
                        // Was visible, now hiding it
                        targetContent.style.display = 'none';
                        button.textContent = button.getAttribute('data-original-text');
                    } else {
                        // Was hidden, now showing it
                        targetContent.style.display = 'block';
                        button.textContent = 'Hide';
                    }

                    // Check if all contents are hidden, if so hide detail row
                    var allHidden = allContents.every(function(content) {
                        return content.style.display === 'none';
                    });

                    if (allHidden) {
                        detailRow.style.display = 'none';
                    }
                }
            }
        }

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', function() {
            // Add event listeners to agent dialog buttons
            var agentButtons = document.querySelectorAll('.agent-btn');
            agentButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'agent');
                });
            });
            
            // Add event listeners to grader dialog buttons
            var graderButtons = document.querySelectorAll('.grader-btn');
            graderButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'grader');
                });
            });
            
            // Add event listeners to test runner log buttons
            var runnerButtons = document.querySelectorAll('.runner-btn');
            runnerButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'runner');
                });
            });
            
            // Add event listeners to cost estimator buttons
            var costButtons = document.querySelectorAll('.cost-btn');
            costButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'cost');
                });
            });
            
            
            // Add event listeners to cost estimator true buttons
            var costTrueButtons = document.querySelectorAll('.cost-true-btn');
            costTrueButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'cost-true');
                });
            });
            

            // Add event listeners to gatekeeper dialog buttons
            var gatekeeperButtons = document.querySelectorAll('.gatekeeper-btn');
            gatekeeperButtons.forEach(function(button) {
                button.addEventListener('click', function() {
                    toggleDialogContent(this, 'gatekeeper');
                });
            });
        });
    </script>
</body>
</html>
"""
    
    with open(html_file, 'w') as f:
        f.write(html_content)

    if config.azure_storage_enabled:
        # The last 3 parts of the path, are folder/html/file.html, and will be replicated in the
        # Azure blob results folder.
        blob_filename = str(Path(*Path(html_file).parts[-3:]))
        blob_url = write_output_file(config, blob_filename, html_content)
        if blob_url:
            print(f"HTML summary uploaded to {blob_url}")
        else:
            print(f"Error: Failed to upload HTML summary to Azure Storage")

    
    return html_file


def process_results(base_dir: str, html_dir: str, config: HarnessConfig, case_set: Optional[str] = None) -> None:
    """
    Main function to process results and generate all outputs.
    
    Args:
        base_dir: Base directory for outputs
        html_dir: Directory for HTML files  
        config: Config that created the results
        case_set: Optional case set name to filter results
    """
    # Load results from JSON files
    json_dir = os.path.join(base_dir, "json")
    results = load_results_from_json(json_dir)
    
    if not results:
        print("Warning: No results found in JSON files")
        return
    
    # Filter by case set if specified
    if case_set:
        case_set_cases = case_data.get_case_set(config.dataset, case_set)
        case_ids_in_set = {case['case_id'] for case in case_set_cases}
        original_count = len(results)
        results = [r for r in results if r.get('case_id') in case_ids_in_set]
        
        print(f"Loaded {original_count} results from JSON files")
        print(f"Filtered to {len(results)} results matching case set '{case_set}'")
        
        if not results:
            print(f"Warning: No results found for case set '{case_set}'")
            return
    else:
        print(f"Loaded {len(results)} results from JSON files")
    
    # Accuracy + cost report for this run (grouped by agent/model, broken out by case set).
    # Built before the summary so the numbers can be folded into summary.txt.
    report_text = ""
    try:
        report = build_accuracy_cost_report([base_dir])
        report_text = format_accuracy_cost_report(report)
    except Exception as e:
        print(f"Warning: could not generate accuracy/cost report: {e}")

    # Generate summary file (with the accuracy/cost report folded in)
    summary_file = generate_summary_file(results, base_dir, config, report_text)

    # Print console summary
    print_console_summary(summary_file, results)

    # Generate cost analysis
    generate_cost_analysis(results, config)

    # Generate HTML overview
    html_file = generate_html_overview(results, html_dir, config, case_set)
    print(f"\nHTML overview saved to: {html_file}")


# ---------------------------------------------------------------------------
# Accuracy / cost reporting (grouped by agent + model, broken out by case set)
# ---------------------------------------------------------------------------

# Case sets reported by default; everything else (e.g. "hard_validation", legacy
# "validation_validation") is ignored unless explicitly requested.
DEFAULT_REPORT_CASE_SETS = ("train", "validation", "test")

# Bootstrap defaults for confidence intervals.
BOOTSTRAP_RESAMPLES = 1000
CI_CONFIDENCE = 0.95


def case_is_errored(result: Dict[str, Any]) -> bool:
    """True if a case looks like a failed/incomplete run rather than a real diagnosis.

    Mirrors the harness completion check: an error string in ``model_diagnosis`` or any
    test that failed to price in a stored cost analysis.
    """
    diagnosis = result.get("model_diagnosis", "")
    if isinstance(diagnosis, str):
        low = diagnosis.lower()
        if "error code:" in low or "exception" in low or low.startswith("error"):
            return True
    for key, value in result.items():
        if key.startswith("cost_analysis_") and isinstance(value, dict):
            for detail in value.get("cost_details", []) or []:
                if detail.get("method") == "failure":
                    return True
    return False


def case_cost(result: Dict[str, Any], method: str) -> Optional[float]:
    """total_with_visits for a case under a given costing method, or None if absent."""
    analysis = result.get(f"cost_analysis_{method}")
    if not isinstance(analysis, dict):
        return None
    totals = analysis.get("totals")
    if not isinstance(totals, dict):
        return None
    value = totals.get("total_with_visits")
    return float(value) if value is not None else None


def bootstrap_ci(data: List[float], n_bootstrap: int = BOOTSTRAP_RESAMPLES,
                 confidence_level: float = CI_CONFIDENCE) -> List[float]:
    """Percentile bootstrap CI for a mean."""
    import numpy as np

    if len(data) < 2:
        only = data[0] if data else 0.0
        return [only, only]
    arr = np.asarray(data, dtype=float)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    alpha = 1 - confidence_level
    lower = np.percentile(means, (alpha / 2) * 100)
    upper = np.percentile(means, (1 - alpha / 2) * 100)
    return [float(lower), float(upper)]


def accuracy_binomial_ci(accuracy: float, n: int, confidence_level: float = CI_CONFIDENCE) -> List[float]:
    """Normal-approximation binomial CI for accuracy."""
    import numpy as np

    if n <= 0:
        return [0.0, 0.0]
    z = 1.96  # 95%; default confidence
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    return [max(0.0, accuracy - z * se), min(1.0, accuracy + z * se)]


def _summarize_cases(cases: List[Dict[str, Any]], methods: List[str], compute_ci: bool) -> Dict[str, Any]:
    """Compute accuracy + per-method cost stats for one (group, case set) bucket.

    Errored cases are excluded from stats but counted. Uneven repetitions across cases are
    flagged. Costs are reported as mean per case and run total, per costing method present.
    """
    errored = [c for c in cases if case_is_errored(c)]
    valid = [c for c in cases if not case_is_errored(c)]

    n_correct = sum(1 for c in valid if c.get("correct"))
    n = len(valid)
    accuracy = n_correct / n if n else 0.0

    reps_per_case: Dict[str, int] = {}
    for c in valid:
        reps_per_case[str(c.get("case_id"))] = reps_per_case.get(str(c.get("case_id")), 0) + 1
    uneven_reps = len(set(reps_per_case.values())) > 1 if reps_per_case else False

    costs: Dict[str, Any] = {}
    for method in methods:
        vals = [v for v in (case_cost(c, method) for c in valid) if v is not None]
        if not vals:
            continue
        entry: Dict[str, Any] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "total": sum(vals),
        }
        if compute_ci:
            entry["ci"] = bootstrap_ci(vals)
        costs[method] = entry

    summary: Dict[str, Any] = {
        "cases": len(cases),
        "valid_cases": n,
        "errored": len(errored),
        "unique_cases": len(reps_per_case),
        "uneven_reps": uneven_reps,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "costs": costs,
    }
    if compute_ci:
        summary["accuracy_ci"] = accuracy_binomial_ci(accuracy, n)
    return summary


def build_accuracy_cost_report(run_dirs: List[str], *, case_sets: tuple[str, ...] = DEFAULT_REPORT_CASE_SETS,
                               compute_ci: bool = False) -> Dict[str, Any]:
    """Build a grouped accuracy/cost report over one or more run folders.

    Groups cases by (agent, model) read from each run's run_config.json, pooling folders,
    and breaks each group down by case set. Aborts if folders span multiple datasets.
    """
    datasets: set[str] = set()
    methods: set[str] = set()
    # group_key -> case_set -> list[result]
    groups: Dict[tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}

    for run_dir in run_dirs:
        json_dir = os.path.join(run_dir, "json")
        if not os.path.isdir(json_dir):
            print(f"Warning: no 'json' directory in {run_dir!r}, skipping")
            continue
        cfg = HarnessConfig.restore_run_config(run_dir)
        datasets.add(cfg.dataset)
        results = load_results_from_json(json_dir)
        methods.update(get_cost_analysis_methods(results))
        key = (cfg.agent, cfg.model)
        bucket = groups.setdefault(key, {})
        for r in results:
            cs = r.get("case_set", cfg.case_set)
            if cs not in case_sets:
                continue
            bucket.setdefault(cs, []).append(r)

    if len(datasets) > 1:
        raise SystemExit(f"Cannot report across multiple datasets: {sorted(datasets)}")

    dataset = next(iter(datasets), None)
    method_list = sorted(methods)

    report: Dict[str, Any] = {"dataset": dataset, "methods": method_list, "groups": []}
    for (agent, model), by_set in sorted(groups.items()):
        group_entry: Dict[str, Any] = {"agent": agent, "model": model, "case_sets": {}}
        for cs in case_sets:
            cases = by_set.get(cs)
            if not cases:
                continue
            summ = _summarize_cases(cases, method_list, compute_ci)
            if dataset:
                try:
                    summ["expected"] = len(case_data.get_case_set(dataset, cs))
                    summ["incomplete"] = summ["unique_cases"] < summ["expected"]
                except Exception:
                    summ["expected"] = None
                    summ["incomplete"] = False
            group_entry["case_sets"][cs] = summ
        report["groups"].append(group_entry)
    return report


def format_accuracy_cost_report(report: Dict[str, Any], title: str = "ACCURACY & COST REPORT") -> str:
    """Render the grouped accuracy/cost report as text (shared by console + summary.txt)."""
    lines = ["=" * 72, f"{title}  (dataset: {report.get('dataset')})", "=" * 72]
    if not report["groups"]:
        lines.append("  No reportable cases (train/validation/test).")
        return "\n".join(lines)
    for group in report["groups"]:
        lines.append(f"\n{group['agent']} | {group['model']}")
        for cs, s in group["case_sets"].items():
            acc = f"{100 * s['accuracy']:.1f}% ({s['n_correct']}/{s['valid_cases']})"
            if "accuracy_ci" in s:
                lo, hi = s["accuracy_ci"]
                acc += f"  CI[{100 * lo:.1f}-{100 * hi:.1f}]"
            lines.append(f"  {cs:<11} acc {acc}")
            for method, c in s["costs"].items():
                line = f"      cost[{method}] mean ${c['mean']:.2f}  total ${c['total']:.2f}"
                if "ci" in c:
                    line += f"  CI[{c['ci'][0]:.2f}-{c['ci'][1]:.2f}]"
                lines.append(line)
            flags = []
            if s["errored"]:
                flags.append(f"{s['errored']} errored (excluded)")
            if s.get("incomplete"):
                flags.append(f"incomplete {s['unique_cases']}/{s['expected']}")
            if s["uneven_reps"]:
                flags.append("uneven reps")
            if flags:
                lines.append(f"      \u26a0 {', '.join(flags)}")
    return "\n".join(lines)


def print_accuracy_cost_report(report: Dict[str, Any], title: str = "ACCURACY & COST REPORT") -> None:
    """Print the grouped accuracy/cost report to the console."""
    print("\n" + format_accuracy_cost_report(report, title=title))
