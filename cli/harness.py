"""CLI entry point for the batch SDBench harness runner.

Wires command-line arguments and environment to the batch runner in
``sdbench.harness`` (``run``) and the reporting in ``sdbench.reporting``
(``process_results``). Run with ``python -m sdbench.cli.harness [...]``.
"""

import datetime
import getpass
import os
import subprocess
from pathlib import Path

import dotenv

from sdbench.azure_storage import write_output_file
from sdbench.config import HarnessConfig
from sdbench.harness import run, scan_completed_cases
from sdbench.reporting import process_results


def _resolve_run_owner() -> str:
    """Best-effort owner label for the results directory name.

    Tries the git global user's first name, then the OS login, then 'unknown'.
    This is a CLI convenience for human-readable run folders, not core harness logic.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--global", "user.name"],
            capture_output=True,
            text=True,
            check=True,
        )
        name = result.stdout.strip()
        if name:
            return name.split()[0]
    except Exception:
        pass
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def create_results_directory(config: HarnessConfig, prefix: str = "") -> str:
    """Create a directory structure for results.

    `prefix` is an optional owner/run label prepended to the directory name.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use label if provided, otherwise fall back to agent name
    label_to_use = config.label if config.label else config.agent

    # Use provided result_folder as base directory, otherwise default to "results"
    base_results_dir = config.result_folder if config.result_folder else "results"

    prefix_part = f"{prefix}_" if prefix else ""
    base_dir = f"{base_results_dir}/{prefix_part}{config.dataset}_{label_to_use}_{config.model}_{config.case_set}_{timestamp}_k{config.k}"
    os.makedirs(base_dir)

    return base_dir


def main():
    dotenv.load_dotenv()

    harness_config = HarnessConfig.from_command_line()
    if harness_config.resume:
        resume_dir = harness_config.resume

        if not os.path.exists(resume_dir):
            print(f"Error: Resume directory {resume_dir} does not exist")
            exit(1)

        print(f"Resuming from: {resume_dir}")

        # If resuming, we ignore the config provided, and use the run_config.json from the initial run
        command_line_harness_config = harness_config
        harness_config = HarnessConfig.restore_run_config(resume_dir)

        # Allow these to be overridden on resume, any others that are set are ignored
        harness_config.debug = command_line_harness_config.debug
        harness_config.max_workers = command_line_harness_config.max_workers

        completed_cases = scan_completed_cases(resume_dir)
        print(f"Found {len(completed_cases)} completed cases")

        base_dir = resume_dir
    else:
        completed_cases: set[tuple[str, int]] = set()
        base_dir = create_results_directory(harness_config, _resolve_run_owner())
        run_config_filename, run_config_json = harness_config.save_run_config(base_dir)
        if harness_config.azure_storage_enabled:
            # The last 3 parts of the path, are folder/run_config.json, and will be replicated in the
            # Azure blob results folder.
            blob_filename = str(Path(*Path(run_config_filename).parts[-2:]))
            blob_url = write_output_file(harness_config, blob_filename, run_config_json)
            if blob_url:
                print(f"Run config uploaded to {blob_url}")
            else:
                print(f"Error: Failed to upload run config to Azure Storage")

    json_dir = os.path.join(base_dir, "json")
    html_dir = os.path.join(base_dir, "html")

    # Ensure directories exist
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)

    run(harness_config, json_dir, completed_cases)

    # Process results and generate summaries/HTML
    process_results(base_dir, html_dir, harness_config)


if __name__ == "__main__":
    main()
