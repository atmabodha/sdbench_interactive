"""SDBench Case Data

This module provides utilities for loading and validating case data for the Sequential Diagnosis
Benchmark (SDBench). It loads medical case data from a single-file dataset package (JSON or CSV)
or a directory of per-case JSON files, and organizes them into case sets for evaluation.

Dataset Requirements:
--------------------
A dataset lives under data/processed_case_data/{dataset_name}/ and may be packaged in one of
three layouts. When more than one is present the first match below wins:

    data/processed_case_data/{dataset_name}/
    ├── dataset.json     # (1) Single-file JSON dataset: all cases and their case-set
    or                   #     membership in one JSON array. Preferred when present.
    ├── dataset.csv      # (2) Single-file CSV dataset: the same, one case per row.
    or                   #     Used only when dataset.json is absent.
    ├── case_sets.json   # (3) Individual JSON cases: each case is provided as a separate
    ├── case_001.json    #     JSON file as well as a case sets file defining the splits.
    ├── case_002.json    #     Useful when building a dataset on a new source.
    ├── case_003.json    #     Only used when dataset.json or dataset.csv is absent.
    ├── case_???.json    #     Note: Cases must be a .json file starting with "case_" but
    └── case_00n.json    #     don't require an id in the filename.

All three layouts describe the same case object — see "Case Format" below. The single-file
dataset.json is a JSON array of those objects; dataset.csv holds the same fields with one case
per row (lists flattened to comma-separated, csv-escaped cells); the individual-cases layout
stores one object per file. JSON is preferred over CSV (CSV can't represent a full_text larger
than a spreadsheet cell, and stores tests / case_sets as escaped strings rather than native
arrays).

Dataset Location:
-----------------
The data directory location is determined by:
1. Environment variable SDBENCH_DATA_PATH (if set)
2. Default: "data" directory relative to current working directory
3. Fallback: datasets bundled inside the installed package (currently the `synthetic`
   demo dataset), used when a dataset is not found under the above location.

Case Format:
------------
A case is a JSON object of the form below (shown with comments for documentation only).
The single-file dataset.json is a JSON array of these objects.
Each individual case_*.json file holds exactly one of these objects.
The single-file dataset.csv carries the same fields flattened into CSV escaped columns
(case_sets becomes a comma-separated, csv-escaped cell). Required fields must be non-empty strings:

    {
        // ---- required ----
        "case_id": "1",                   // unique id within the dataset (ints are coerced to str)
        "case_title": "...",              // title, typically indicates the final diagnosis
        "final_diagnosis": "...",         // ground-truth diagnosis used for evaluation
        "case_summary": "...",            // brief intro; must NOT reveal the diagnosis
        "full_text": "...",               // full clinical detail (history, exam, results, course)

        // ---- single-file only ----
        "case_sets": ["validation"]       // splits this case belongs to (e.g. "validation",
                                          // "test"). In the individual-cases layout this is
                                          // omitted and provided centrally in case_sets.json.
    }

Additionally there are optional legacy fields for "estimated_clinician_visits" and "tests" that were
used to compute real-world workup costs of a case, but don't make up the final sdbench benchmark.

Case Sets:
----------
In the single-file datasets each case carries its own "case_sets" list (above). In the
individual-cases layout, membership instead lives in a separate case_sets.json that maps each
split name to a list of case ids:
{
    "validation": ["50", "51", ...],
    "test": ["100", "101", ...]
}
An "all" set containing every case is always available regardless of layout. A
"validationandtest" set (the deduped union of "validation" and "test") is also available
whenever both of those sets exist.

Command Line Interface:
----------------------
A CLI is provided to exercise the case data loading functionality:

    python -m sdbench.cli.case_data <dataset> [--case-id <id>] [--case-set <set>] [--list-only]

This will load the specified dataset, list available case sets, and optionally display
details for a specific case or case set, i.e.:

    # List all case sets for the dataset
    python -m sdbench.cli.case_data <dataset> --list-only

    # Show sample from each case set in the dataset
    python -m sdbench.cli.case_data <dataset>

    # Load a specific case
    python -m sdbench.cli.case_data <dataset> --case-id <case_id>

    # List all case IDs in a case set
    python -m sdbench.cli.case_data <dataset> --case-set <case_set>

Supported Datasets:
-------------------
See main README.md file for details sourcing the "nejm" and "mimic" datasets.

Bring Your Own Dataset:
-----------------------
You can run SDBench on your own dataset as long as it follows the required directory structure,
file format and contents as described above.
"""

import csv
import functools
import json
import os
import pathlib
from typing import Any, TypedDict

from sdbench._paths import sdbench_data_path


class CaseData(TypedDict):
    """Represents the data for an individual SDBench case.

    Is a TypedDict over the original JSON structure loaded from file.
    All fields that are directly used in the harness are included here.

    Attributes:
        dataset: The dataset this case belongs to.
        case_id: The unique identifier for the case within the dataset. Normalised to a string.
        case_title: The title of the case, which typically indicates the final diagnosis.
        final_diagnosis: The final diagnosis for the case to use as the ground truth.
        case_summary: A brief summary of the case, typically one or two sentences in the style of a
            NEJM case introduction: "A XX-year-old man presented with XX. He underwent XX."
            This is used to provide initial context to the agent and should not disclose the
            final diagnosis or key details required to arrive at the diagnosis.
        full_text: The full text of the case, including all clinical details, history,
            examination findings, test results, and hospital course.
            This is used provided as context to the gatekeeper and the judge calls.
    """

    dataset: str
    case_id: str
    case_title: str
    final_diagnosis: str
    case_summary: str
    full_text: str


# Datasets shipped inside the installed package (currently the synthetic demo dataset),
# used as a fallback so `pip install sdbench` works out-of-the-box without a local ./data.
_BUNDLED_DATA_DIR = pathlib.Path(__file__).parent / "_bundled_data"


def _dataset_dir(dataset: str) -> str:
    """Resolve the directory for a dataset.

    Prefers ``<SDBENCH_DATA_PATH>/processed_case_data/<dataset>``; if that does not exist,
    falls back to the dataset bundled inside the package. The primary (user-facing) path
    is returned when neither exists so error messages reference the expected location.
    """
    primary = os.path.join(sdbench_data_path(), "processed_case_data", dataset)
    if os.path.exists(primary):
        return primary
    bundled = _BUNDLED_DATA_DIR / "processed_case_data" / dataset
    if bundled.exists():
        return str(bundled)
    return primary


def _validate_case_data(case: dict[Any, Any], filename: str) -> None:
    required_fields = {"case_id", "case_title", "final_diagnosis", "case_summary", "full_text"}

    missing_fields = required_fields - set(case.keys())
    if missing_fields:
        raise ValueError(f"Case file {filename} is missing required fields: {missing_fields}")

    # Validate field types (all should be strings, case_id can be converted)
    string_fields = required_fields - {"case_id"}
    non_string_fields = [field for field in string_fields if not isinstance(case[field], str)]
    if non_string_fields:
        raise ValueError(f"Case file {filename} has non-string fields: {non_string_fields}")

    # Validate that required string fields are not empty
    empty_fields = [field for field in string_fields if not case[field] or not case[field].strip()]
    if empty_fields:
        raise ValueError(f"Case file {filename} has empty required fields: {empty_fields}")

    # Validate case_id can be converted to string
    if case["case_id"] is None:
        raise ValueError(f"Case file {filename} has null case_id")


def _load_case_data(dataset: str) -> list[CaseData]:
    directory = _dataset_dir(dataset)
    if not os.path.exists(directory):
        raise ValueError(f"Data directory for dataset '{dataset}' does not exist: {directory}")

    all_cases: list[CaseData] = []
    for filename in os.listdir(directory):
        if (
            filename.startswith("case_")
            and filename.endswith(".json")
            and filename not in ["case_sets.json", "case_id_mapping.json"]
        ):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    case = json.load(f)
                    _validate_case_data(case, filename)
                    case["dataset"] = dataset
                    all_cases.append(case)
            except Exception as e:
                raise ValueError(f"Failed to load case file {filename}") from e
    all_cases.sort(key=lambda c: c["case_id"])  # Sort by original case_id for consistent ordering
    for case in all_cases:
        case["case_id"] = str(
            case["case_id"]
        )  # Ensure case_id is a string (as it might originally be int in the JSON)
    return all_cases


def _load_case_set_ids(dataset: str) -> dict[str, list[str]]:
    case_set_file = os.path.join(_dataset_dir(dataset), "case_sets.json")
    if not os.path.exists(case_set_file):
        raise ValueError(f"Case sets file for dataset '{dataset}' does not exist: {case_set_file}")
    try:
        with open(case_set_file, "r") as f:
            case_set_ids = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load case sets file {case_set_file}") from e
    for k, v in case_set_ids.items():
        if not isinstance(v, list):
            raise ValueError(f"Case set '{k}' in dataset '{dataset}' should be a list of case IDs")
    return {case_set: list(map(str, case_ids)) for case_set, case_ids in case_set_ids.items()}


# Single-file case set filenames: a fixed "dataset.json" / "dataset.csv" inside the
# dataset directory (the dataset is identified by the directory name, the file inside
# is always the same). When one is present it takes priority over the directory of
# per-case JSON files (see _get_case_sets). JSON is preferred over CSV (CSV can't
# represent cases whose full_text exceeds a spreadsheet cell limit, and JSON keeps
# tests/case_sets as native arrays). The fixed name avoids colliding with the legacy
# all_cases.json combined dump.
def _single_file_case_set_paths(dataset: str) -> tuple[str, str]:
    base = _dataset_dir(dataset)
    return os.path.join(base, "dataset.json"), os.path.join(base, "dataset.csv")


def _load_cases_from_json(dataset: str, json_path: str) -> tuple[list[CaseData], dict[str, list[str]]]:
    """Load all cases and their case-set membership from a single JSON file.

    Returns (cases, case_set_ids)
    """
    cases: list[CaseData] = []
    case_set_ids: dict[str, list[str]] = {}
    try:
        with open(json_path, "r") as f:
            records = json.load(f)
        for record in records:
            case: dict[str, Any] = dict(record)
            case_sets = case.pop("case_sets", [])
            _validate_case_data(case, f"{json_path} (case_id={record.get('case_id')})")
            case["case_id"] = str(case["case_id"])
            case["dataset"] = dataset
            case.setdefault("tests", [])
            for set_name in case_sets:
                case_set_ids.setdefault(set_name, []).append(case["case_id"])
            cases.append(case)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Failed to load JSON case set file {json_path}") from e

    cases.sort(key=lambda c: c["case_id"])
    return cases, case_set_ids


def _decode_list_field(cell: str) -> list[str]:
    """Decode a CSV cell holding a comma-separated, csv-escaped list of strings.

    The cell is itself parsed as a single CSV row so that values containing commas or
    quotes (already escaped on write) round-trip correctly. Empty cells yield [].
    """
    cell = (cell or "").strip()
    if not cell:
        return []
    # next() is safe: csv.reader over a single non-empty line always yields one row.
    return [item.strip() for item in next(csv.reader([cell]))]


def _load_cases_from_csv(dataset: str, csv_path: str) -> tuple[list[CaseData], dict[str, list[str]]]:
    """Load all cases and their case-set membership from a single CSV file.

    Returns (cases, case_set_ids)
    """
    cases: list[CaseData] = []
    case_set_ids: dict[str, list[str]] = {}
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            has_initial_presentation = "initial_presentation_tests" in (reader.fieldnames or [])
            for row_num, row in enumerate(reader, start=2):  # start=2: row 1 is the header
                case: dict[str, Any] = {
                    field: (row.get(field) or "")
                    for field in ("case_title", "final_diagnosis", "case_summary", "full_text")
                }
                case["case_id"] = row.get("case_id")
                _validate_case_data(case, f"{csv_path} (row {row_num})")
                case["case_id"] = str(case["case_id"])
                case["dataset"] = dataset

                visits = (row.get("estimated_clinician_visits") or "").strip()
                if visits:
                    case["estimated_clinician_visits"] = int(visits)

                test_names = _decode_list_field(row.get("tests", ""))
                if has_initial_presentation:
                    initial = set(_decode_list_field(row.get("initial_presentation_tests", "")))
                    case["tests"] = [
                        {"name": name, "in_initial_presentation": name in initial}
                        for name in test_names
                    ]
                else:
                    case["tests"] = [{"name": name} for name in test_names]

                for set_name in _decode_list_field(row.get("case_sets", "")):
                    case_set_ids.setdefault(set_name, []).append(case["case_id"])

                cases.append(case)  # type: ignore[arg-type]
    except Exception as e:
        raise ValueError(f"Failed to load CSV case set file {csv_path}") from e

    cases.sort(key=lambda c: c["case_id"])
    return cases, case_set_ids



@functools.lru_cache(maxsize=None)
def _get_case_sets(dataset: str) -> dict[str, list[CaseData]]:
    # A single <dataset>.json or <dataset>.csv in the dataset directory takes priority
    # over the directory of per-case JSON files + case_sets.json (JSON before CSV).
    # Fall back to the per-case JSON layout otherwise.
    json_path, csv_path = _single_file_case_set_paths(dataset)
    if os.path.exists(json_path):
        cases, case_set_ids = _load_cases_from_json(dataset, json_path)
    elif os.path.exists(csv_path):
        cases, case_set_ids = _load_cases_from_csv(dataset, csv_path)
    else:
        cases = _load_case_data(dataset)
        case_set_ids = _load_case_set_ids(dataset)
    print(f"Loaded {len(cases)} cases for dataset '{dataset}'")
    case_sets = {
        name: [case for case in cases if case["case_id"] in case_ids]
        for name, case_ids in case_set_ids.items()
    }
    case_sets["all"] = cases  # all is always defined

    # Convenience union (mirrors "all"): run validation + test together in one run.
    # Only defined when both component sets exist; deduped, order preserved.
    if "validation" in case_sets and "test" in case_sets:
        seen: set[str] = set()
        union: list[CaseData] = []
        for case in case_sets["validation"] + case_sets["test"]:
            if case["case_id"] not in seen:
                seen.add(case["case_id"])
                union.append(case)
        case_sets["validationandtest"] = union

    case_set_info = ", ".join(
        [f"{name} ({len(case_list)})" for name, case_list in case_sets.items()]
    )
    print(f"Loaded {len(case_sets)} case sets for dataset '{dataset}': {case_set_info}")

    return case_sets


@functools.lru_cache(maxsize=None)
def _get_case_lookup(dataset: str) -> dict[str, CaseData]:
    all_cases = _get_case_sets(dataset)["all"]
    return {str(c["case_id"]): c for c in all_cases}


def get_case_set(dataset: str, case_set: str) -> list[CaseData]:
    """Get all cases in a specific case set for the specified dataset.
    File accesses are cached for efficiency so repeated calls are fast.

    Args:
        dataset: The dataset name.
        case_set: The case set name (e.g., "validation", "test", or "all" for all cases).

    Returns:
        List of CaseData objects in the specified case set ordered by case_id.

    Raises:
        ValueError if the case_set is not found.
    """
    case_sets_map = _get_case_sets(dataset)
    if case_set not in case_sets_map:
        available_sets = list(case_sets_map.keys())
        raise ValueError(
            f"Case set '{case_set}' not found in dataset '{dataset}'. Available case sets: {available_sets}"
        )
    return case_sets_map[case_set]


def get_case(dataset: str, case_id: str) -> CaseData:
    """Get a specific case by its ID from the specified dataset.
    File accesses are cached for efficiency so repeated calls are fast.

    Args:
        dataset: The dataset name.
        case_id: The case ID to retrieve (will be converted to string).

    Returns:
        The CaseData object for the specified case ID.

    Raises:
        ValueError if the case_id is not found.
    """
    case_id = str(case_id)  # no-op for properly typed code, but just in case
    casedata = _get_case_lookup(dataset).get(case_id)
    if not casedata:
        raise ValueError(f"Case ID '{case_id}' not found in dataset '{dataset}'")
    return casedata
