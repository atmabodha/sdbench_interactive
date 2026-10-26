"""SDBench CLI Commands

This module provides command line interfaces for various SDBench functionality.

Available CLI commands:
- harness: Run the batch harness over a case set (parallel, resumable) + reporting
- agent: Run a single case evaluation
- case_data: Exercise case data loading functionality
- judge: Run diagnosis checking for a specific case
- prompts: Print prompt templates
- models: Test model chat completion functionality directly
- test_runner: Run batches of tests against the gatekeeper
- gatekeeper: Exercise the gatekeeper functions directly
- costing: Estimate costs for medical tests using CPT lookup and LLM fallback
- azure_storage: Test Azure Blob Storage connectivity and permissions

Usage:
    python -m sdbench.cli.harness --dataset <dataset> [--num-cases N] [--max-workers N] [options]
    python -m sdbench.cli.agent <dataset> <case_id> [--agent NAME] [--model NAME] [options]
    python -m sdbench.cli.case_data <dataset> [options]
    python -m sdbench.cli.judge <dataset> <case_id> <diagnosis> [options]
    python -m sdbench.cli.prompts <prompt_id> <version>
    python -m sdbench.cli.models <model> <user_message> [options]
    python -m sdbench.cli.test_runner <dataset> <case_id> <test1> <test2> ... [--batch-size N] [--debug]
    python -m sdbench.cli.gatekeeper <dataset> <case_id> <test1> <test2> ... [--use-order-test] [--debug]
    python -m sdbench.cli.costing <test1> <test2> ... [--num-rounds N] [--output-format FORMAT] [--debug]
    python -m sdbench.cli.azure_storage --account-name <name> --container-name <name> [--test-upload] [--list-blobs]
"""

__all__ = []
