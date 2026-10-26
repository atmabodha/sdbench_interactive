import argparse
from dataclasses import asdict, dataclass, field, fields, replace
import datetime
import functools
import importlib.util
import json
import os
from typing import Any

from sdbench._paths import local_overrides_base, sdbench_data_path


def costing_data_dir() -> str:
    """Directory holding the grounded-costing charge master, under SDBENCH_DATA_PATH."""
    return os.path.join(sdbench_data_path(), "costing_data")


def resolve_costing_source_file_path(costing_source_file_path: str) -> str:
    """Resolve a charge-master file path under ``<SDBENCH_DATA_PATH>/costing_data``.

    ``costing_source_file_path`` is interpreted relative to the ``costing_data``
    directory; an absolute path is used as-is (``os.path.join`` keeps it). This lets
    the charge master track ``SDBENCH_DATA_PATH`` while still allowing an explicit
    out-of-tree override via ``--costing-source-file-path /abs/path.xlsx``.
    """
    return os.path.join(costing_data_dir(), costing_source_file_path)


@dataclass(kw_only=True)
class SDBenchConfig:
    """Configuration for running a set of cases for SDBench.

    Everything for SDBench to reproducibly run an agent is in SDBenchConfig.
    The actual mechanics of running the cases, including the agent and underlying model to use,
    is configured in HarnessConfig.
    """

    debug: bool = False

    compute_costs: bool = True
    # Cost of the real-world workup the patient actually received.
    # Off by default: it requires in_initial_presentation flags to be meaningful and
    # aligned to the gatekeeper behaviour.
    compute_actual_workup_costs: bool = False
    clinician_cost: int = 300
    costing_method: str = "batch_llm_only"
    costing_parallelism: int = (
        20  # Number of processes/threads for parallel cost estimation. If > 1, is parallelized.
    )
    costing_multiprocess: bool = (
        False  # If True, use ProcessPoolExecutor, otherwise ThreadPoolExecutor
    )
    # Hospital charge master spreadsheet, loaded directly from
    # https://healthy.kaiserpermanente.org/content/dam/kporg/final/documents/health-plan-documents/coverage-information/910511770-central-hospital-standard-charges-wa-en.xlsx
    # Relative to <SDBENCH_DATA_PATH>/costing_data (see resolve_costing_source_file_path);
    # an absolute path overrides that location. Resolved lazily so it tracks the data dir.
    costing_source_file_path: str = (
        "910511770-central-hospital-standard-charges-wa-en.xlsx"
    )
    costing_model: str = "gpt-5-mini"
    # Costing doesn't need ZDR restrictions as it only sends test names to the model
    costing_model_args: dict[str, Any] = field(
        default_factory=lambda: {"openrouter_zdr_only": False}
    )
    costing_cache_cosmosdb_enabled: bool = False
    costing_cache_cosmosdb_endpoint: str | None = None
    costing_cache_cosmosdb_database: str | None = None
    costing_cache_cosmosdb_container: str | None = None
    costing_cache_use_managed_identity: bool = False

    test_runner_enforce_single_turn_type: bool = True

    max_turns: int
    max_questions_per_turn: int | None = None
    max_tests_per_turn: int | None = None
    agent_model_args: dict[str, Any] = field(default_factory=dict)
    agent_extra_args: dict[str, str] = field(default_factory=dict)

    gatekeeper_prompt_version: str
    gatekeeper_model: str
    gatekeeper_model_args: dict[str, Any] = field(default_factory=dict)
    gatekeeper_chunk_size: int = 10  # Max number of tests to request from the gatekeeper at a time, distinct from size of the test batch passed to the BatchTestRunner

    judge_prompt_version: str
    judge_model: str
    judge_model_args: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _build_parser(add_help: bool = True) -> argparse.ArgumentParser:
        """Build the argparse parser defining all SDBench config overrides."""
        parser = argparse.ArgumentParser(description="Configuration for SDBench", add_help=add_help)

        # The command line argument to select a base configuration, from which to override with the rest of the command line args
        # This argument doesn't exist in SDBenchConfig itself
        parser.add_argument(
            "--sdbench-config", type=str, default="default", help="SDBench configuration to run"
        )

        # All SDBench arguments match exactly the fields in SDBenchConfig and are suppressed by default
        # So we will override with all the values that are supplied
        parser.add_argument(
            "--test-runner-enforce-single-turn-type",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument("--clinician-cost", type=int, default=argparse.SUPPRESS)
        parser.add_argument(
            "--compute-costs",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--compute-actual-workup-costs",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--costing-method",
            type=str,
            default=argparse.SUPPRESS,
            choices=["unified_cpt_llm", "llm_only", "batch_llm_only"],
        )
        parser.add_argument("--costing-parallelism", type=int, default=argparse.SUPPRESS)
        parser.add_argument(
            "--costing-multiprocess",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument("--costing-source-file-path", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--costing-model", type=str, default=argparse.SUPPRESS)
        parser.add_argument(
            "--costing-model-args",
            type=_parse_json_dict,
            default=argparse.SUPPRESS,
            help="Model arguments for costing as a JSON dictionary",
        )
        parser.add_argument(
            "--costing-cache-cosmosdb-enabled",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--costing-cache-cosmosdb-endpoint", type=str, default=argparse.SUPPRESS
        )
        parser.add_argument(
            "--costing-cache-cosmosdb-database", type=str, default=argparse.SUPPRESS
        )
        parser.add_argument(
            "--costing-cache-cosmosdb-container", type=str, default=argparse.SUPPRESS
        )
        parser.add_argument(
            "--costing-cache-use-managed-identity",
            type=bool,
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
        )
        parser.add_argument("--max-turns", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--max-questions-per-turn", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--max-tests-per-turn", type=int, default=argparse.SUPPRESS)
        parser.add_argument(
            "--agent-arg",
            dest="agent_extra_args",
            type=_validate_agent_arg,
            action="append",
            default=argparse.SUPPRESS,
            help="Extra arguments to pass to the agent in key=value format. Can be specified multiple times.",
        )
        parser.add_argument(
            "--agent_model_args",
            type=_parse_json_dict,
            default=argparse.SUPPRESS,
            help="Additional model arguments for the main agent as a JSON dictionary",
        )
        parser.add_argument("--gatekeeper-prompt-version", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--gatekeeper-model", type=str, default=argparse.SUPPRESS)
        parser.add_argument(
            "--gatekeeper-model-args",
            type=_parse_json_dict,
            default=argparse.SUPPRESS,
            help="Model arguments for gatekeeper as a JSON dictionary",
        )
        parser.add_argument("--gatekeeper-chunk-size", type=int, default=argparse.SUPPRESS)
        parser.add_argument("--judge-prompt-version", type=str, default=argparse.SUPPRESS)
        parser.add_argument("--judge-model", type=str, default=argparse.SUPPRESS)
        parser.add_argument(
            "--judge-model-args",
            type=_parse_json_dict,
            default=argparse.SUPPRESS,
            help="Model arguments for judge as a JSON dictionary",
        )

        return parser

    @staticmethod
    def from_command_line(dataset: str, argv: list[str]) -> "SDBenchConfig":
        parser = SDBenchConfig._build_parser()
        args = parser.parse_args(argv)
        sdbench_config_name = vars(args).pop("sdbench_config")
        config = get_sdbench_config(dataset, config_name=sdbench_config_name)
        print(f"Using base SDBench config: {config} for dataset '{dataset}'")

        overrides = vars(args)
        if "agent_extra_args" in overrides:
            # Convert agent_extra_args from list of tuples to dict if present
            overrides["agent_extra_args"] = dict(overrides.get("agent_extra_args", []))

        if overrides:
            print(f"Overriding config with command line arguments: {overrides}")
            config = replace(config, **overrides)

        return config


def _validate_agent_arg(value: str) -> tuple[str, str]:
    """Validate --agent-arg argument format and return (key, value) tuple."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid format '{value}'. --agent-arg values must themselves be in key=value format "
            f"(e.g., '--agent-arg=foo=bar')"
        )
    key, val = value.split("=", 1)
    key = key.strip()
    val = val.strip()
    if not key:
        raise argparse.ArgumentTypeError(
            f"Invalid format '{value}'. --agent-arg values must themselves be in key=value format "
            f"(e.g., '--agent-arg=foo=bar')"
        )
    return (key, val)


def _parse_json_dict(value: str) -> dict[str, Any]:
    """Parse JSON string into a dictionary for model args."""
    if not value.strip():
        return {}

    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise argparse.ArgumentTypeError(
                f"JSON value must be a dictionary/object, got {type(parsed).__name__}: {value}"
            )
        return parsed
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}. Value: {value}") from e


@dataclass(kw_only=True)
class AgentConfig:
    """A subset of the SDBenchConfig that is passed to the agent."""

    debug: bool
    clinician_cost: int
    max_turns: int
    max_questions_per_turn: int | None
    max_tests_per_turn: int | None
    agent_extra_args: dict[str, str]

    @staticmethod
    def from_sdbench_config(sdbench_config: SDBenchConfig) -> "AgentConfig":
        return AgentConfig(
            **{
                k: v
                for k, v in vars(sdbench_config).items()
                if k in {f.name for f in fields(AgentConfig)}
            }
        )


@dataclass(frozen=True, kw_only=True)
class CostingConfig:
    """A subset of the SDBenchConfig that is passed to costing functions.
    Frozen dataclass that is hashable and can be used for caching.
    """

    clinician_cost: int = 300
    costing_method: str
    costing_parallelism: int
    costing_multiprocess: bool
    costing_source_file_path: str
    costing_model: str
    costing_model_args_tuple: tuple[tuple[str, Any], ...]
    debug: bool
    costing_cache_cosmosdb_enabled: bool
    costing_cache_cosmosdb_endpoint: str | None
    costing_cache_cosmosdb_database: str | None
    costing_cache_cosmosdb_container: str | None
    costing_cache_use_managed_identity: bool

    @staticmethod
    def from_sdbench_config(sdbench_config: SDBenchConfig) -> "CostingConfig":
        # Dynamically copy all matching fields from SDBenchConfig to CostingConfig
        costing_config_fields = {f.name for f in fields(CostingConfig)}
        kwargs: dict[str, Any] = {}
        # Special case for costing_model_args to convert dict to sorted tuple of items for hashability
        kwargs["costing_model_args_tuple"] = tuple(
            sorted(sdbench_config.costing_model_args.items())
        )
        for field_name in costing_config_fields:
            if hasattr(sdbench_config, field_name):
                kwargs[field_name] = getattr(sdbench_config, field_name)
        return CostingConfig(**kwargs)

    @property
    def costing_model_args(self) -> dict[str, Any]:
        """Dictionary accessor for costing_model_args."""
        return dict(self.costing_model_args_tuple)

    @property
    def resolved_costing_source_file_path(self) -> str:
        """Charge-master path resolved under ``<SDBENCH_DATA_PATH>/costing_data``.

        ``costing_source_file_path`` is stored relative to the ``costing_data`` directory
        (an absolute path overrides that); this resolves it for actual file access.
        """
        return resolve_costing_source_file_path(self.costing_source_file_path)


@dataclass(kw_only=True)
class HarnessConfig:
    """Configuration for the diagnostic harness.

    This dataclass contains all the command line arguments and configuration
    options available to the harness.

    It is stored with each run, and restored when that run if that run is resumed.
    """

    # Harness specific configuration
    resume: str | None = None
    max_workers: int = 5
    debug: bool = False
    result_folder: str | None = None
    label: str | None = None

    # Azure Storage configuration
    azure_storage_enabled: bool = False
    azure_storage_account_name: str | None = None
    azure_storage_container_name: str | None = None
    azure_storage_result_folder: str | None = None

    # Main configuration for the run
    dataset: str = "synthetic"
    agent: str = "baseline"
    model: str = "gpt-5"
    model_args: dict[str, Any] = field(default_factory=dict)
    k: int = 1

    # Case selection configuration
    case_set: str = "train"
    case_id: list[str] | None = None
    num_cases: int = -1
    case_offset: int = 0

    # SDBench configuration
    sdbench_config: SDBenchConfig | None = None

    @property
    def sdbench(self) -> SDBenchConfig:
        """The run's SDBenchConfig, narrowed to non-None.

        `sdbench_config` is Optional because it is populated after construction (by
        from_command_line / restore_run_config). This accessor returns it once set and
        raises if read before initialisation — removing the Optional warnings at the
        many read sites.
        """
        if self.sdbench_config is None:
            raise ValueError("HarnessConfig.sdbench_config has not been initialised")
        return self.sdbench_config

    @staticmethod
    def from_command_line() -> "HarnessConfig":
        # Include the SDBench config options as a parent parser so that `--help` on the harness
        # lists them all. They are parsed into a single namespace and split back out below.
        sdbench_parser = SDBenchConfig._build_parser(add_help=False)
        parser = argparse.ArgumentParser(
            description="Configuration for the SDBench Harness",
            parents=[sdbench_parser],
        )

        # Resume configuration, overrides all other arguments to load from config saved from original run
        parser.add_argument(
            "--resume",
            type=str,
            help="Resume from interrupted run by specifying the results directory path. All original arguments will be automatically restored from the saved configuration.",
        )

        # Harness specific configuration
        # Control how the harness runs, can be changed on resume
        parser.add_argument(
            "--max-workers",
            type=int,
            default=argparse.SUPPRESS,
            help="Maximum number of parallel workers",
        )
        parser.add_argument("--debug", action="store_true", help="Debug mode")
        # Controls where it stores results, are ignored on resume
        parser.add_argument(
            "--result-folder",
            type=str,
            help="Base directory for results (default: 'results'). Used by some agents for input and changes output location for all agents",
        )
        parser.add_argument(
            "--label",
            type=str,
            help="Label to use for results (defaults to agent name if not specified)",
        )

        # Azure Storage configuration
        parser.add_argument(
            "--azure-storage-enabled",
            action="store_true",
            help="Enable Azure Blob Storage upload for results",
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--azure-storage-account-name",
            type=str,
            help="Azure storage account name (required if azure-storage-enabled)",
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--azure-storage-container-name",
            type=str,
            help="Azure container name (required if azure-storage-enabled)",
            default=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--azure-storage-result-folder",
            type=str,
            help="Azure result folder name (required if azure-storage-enabled)",
            default=argparse.SUPPRESS,
        )

        # Case selection configuration to run on a subset, not for full runs
        parser.add_argument(
            "--case-set",
            type=str,
            default=argparse.SUPPRESS,
            help="Set of cases to run. Available sets depend on dataset.",
        )
        parser.add_argument(
            "--case-id",
            nargs="+",
            type=str,
            help="Case ID(s) to run. Can specify multiple IDs separated by spaces or comma-separated. If not specified, processes multiple cases.",
        )
        parser.add_argument(
            "--num-cases",
            type=int,
            default=argparse.SUPPRESS,
            help="Number of cases to run (-1 for all) if --case-id not set",
        )
        parser.add_argument(
            "--case-offset",
            type=int,
            default=argparse.SUPPRESS,
            help="Starting case offset if --case-id not set",
        )

        # Agent running configuration
        parser.add_argument(
            "--dataset",
            type=str,
            default=argparse.SUPPRESS,
            help="Dataset to use (default: synthetic)",
        )
        parser.add_argument(
            "--agent", type=str, default=argparse.SUPPRESS, help="The diagnostic agent to use"
        )
        parser.add_argument(
            "--model", type=str, default=argparse.SUPPRESS, help="Model for the agent to use"
        )
        parser.add_argument(
            "--model-args",
            type=_parse_json_dict,
            default=argparse.SUPPRESS,
            help='Model arguments for the agent as JSON dictionary (e.g., \'{"reasoning_effort": "high"}\')',
        )
        parser.add_argument(
            "--k", type=int, default=argparse.SUPPRESS, help="Number of repetitions per case"
        )

        args = parser.parse_args()

        # Split the combined namespace into harness-level vs SDBench-level overrides by which
        # parser owns each destination. --sdbench-config selects the base SDBench config; it is
        # not itself an SDBenchConfig field.
        all_overrides = vars(args)
        sdbench_config_name = all_overrides.pop("sdbench_config")
        sdbench_dests = {action.dest for action in sdbench_parser._actions if action.dest != "help"}
        harness_overrides = {k: v for k, v in all_overrides.items() if k not in sdbench_dests}
        sdbench_overrides = {k: v for k, v in all_overrides.items() if k in sdbench_dests}

        harness_config = get_harness_config()
        print(f"Using base Harness config: {harness_config}")
        if harness_overrides:
            print(f"Overriding harness config with command line arguments: {harness_overrides}")
            harness_config = replace(harness_config, **harness_overrides)

        sdbench_config = get_sdbench_config(harness_config.dataset, config_name=sdbench_config_name)
        print(f"Using base SDBench config: {sdbench_config} for dataset '{harness_config.dataset}'")
        if "agent_extra_args" in sdbench_overrides:
            # Convert agent_extra_args from list of (key, value) tuples to a dict
            sdbench_overrides["agent_extra_args"] = dict(sdbench_overrides["agent_extra_args"])
        if sdbench_overrides:
            print(f"Overriding SDBench config with command line arguments: {sdbench_overrides}")
            sdbench_config = replace(sdbench_config, **sdbench_overrides)
        sdbench_config.debug = harness_config.debug

        harness_config.sdbench_config = sdbench_config
        return harness_config

    def save_run_config(self, base_dir: str):
        """Save the run configuration to allow proper resuming.

        Args:
            base_dir: Base results directory
            args: Command line arguments namespace
        """
        config_file = os.path.join(base_dir, "run_config.json")

        config_dict = asdict(self)
        config_dict["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        json_content = json.dumps(config_dict, indent=2)

        with open(config_file, "w") as f:
            f.write(json_content)

        print(f"Run configuration saved to {config_file}")
        return config_file, json_content

    @staticmethod
    def restore_run_config(base_dir: str) -> "HarnessConfig":
        """Load the original run configuration."""
        config_file = os.path.join(base_dir, "run_config.json")

        if not os.path.exists(config_file):
            print(f"ERROR: No run_config.json found in {base_dir}")
            print("Cannot resume without the original run configuration.")
            exit(1)

        try:
            with open(config_file, "r") as f:
                raw_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"ERROR: Could not read run configuration: {e}")
            exit(1)

        print(
            f"Loading original run configuration from {raw_config.pop('timestamp', 'unknown time')}:"
        )

        sdbench_dict = raw_config.pop("sdbench_config", None)
        if sdbench_dict:
            # Drop unknown fields (options renamed or removed since the run) so older
            # run_config.json files still load instead of crashing SDBenchConfig(**...).
            known_fields = {f.name for f in fields(SDBenchConfig)}
            dropped = sorted(k for k in sdbench_dict if k not in known_fields)
            if dropped:
                print(f"Warning: ignoring unknown SDBenchConfig fields from run config: {dropped}")
            sdbench_dict = {k: v for k, v in sdbench_dict.items() if k in known_fields}
        sdbench_config = SDBenchConfig(**sdbench_dict) if sdbench_dict else None
        return HarnessConfig(**raw_config, sdbench_config=sdbench_config)


def _get_override_config_module():
    """Load a user-supplied ``config.py`` override, if present.

    The file is looked up at ``<SDBENCH_LOCAL_DIR or CWD>/config.py`` and loaded by
    explicit file path (so it never collides with an unrelated ``config`` module on
    ``sys.path``). Returns ``None`` when local overrides are disabled
    (``SDBENCH_LOCAL_DIR=disable``) or no such file exists.
    """
    local_base = local_overrides_base()
    if local_base is None:
        return None

    config_path = local_base / "config.py"
    if not config_path.is_file():
        return None

    # Memoized by resolved path so the override is executed (and logged) once, matching
    # the single-execution semantics of a normal module import.
    return _load_config_override(str(config_path.resolve()))


@functools.lru_cache(maxsize=None)
def _load_config_override(config_path: str):
    try:
        spec = importlib.util.spec_from_file_location("sdbench_local_config", config_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001 - surface a warning, fall back to defaults
        print(f"Warning: Could not load config override '{config_path}': {e}")
        return None

    # An override config.py must implement the full hook contract. We recognize the file as
    # an *intended* override if it defines at least one known hook, so an unrelated config.py
    # in the working directory is ignored rather than treated as a (broken) override. But once
    # recognized, all hooks are required so call sites can invoke them unconditionally and a
    # partially-written override fails loudly instead of silently skipping behavior.
    required_hooks = (
        "get_harness_config",
        "update_harness_config",
        "get_sdbench_config",
        "update_sdbench_config",
    )
    present = [name for name in required_hooks if callable(getattr(module, name, None))]
    if not present:
        return None
    missing = [name for name in required_hooks if name not in present]
    if missing:
        raise ValueError(
            f"Config override '{config_path}' is missing required hook(s): "
            f"{', '.join(missing)}. An SDBench config override must define all of: "
            f"{', '.join(required_hooks)}."
        )

    print(f"Loading external config module: {config_path}")
    return module


def get_harness_config() -> HarnessConfig:
    """Load the harness configuration, applying any overrides from an external config module."""
    harness_config = None
    override_module = _get_override_config_module()
    if override_module is not None:
        harness_config = override_module.get_harness_config()
    if not harness_config:
        harness_config = HarnessConfig()
    if override_module is not None:
        override_module.update_harness_config(harness_config)
    return harness_config


def get_sdbench_config(dataset: str, config_name: str = "default") -> SDBenchConfig:
    """Load and return the SDBench configuration for the given dataset and config name."""
    sdbench_config = _get_sdbench_config(dataset, config_name)
    override_module = _get_override_config_module()
    if override_module is not None:
        override_module.update_sdbench_config(sdbench_config)
    return sdbench_config


def _get_sdbench_config(dataset: str, config_name: str = "default") -> SDBenchConfig:
    override_module = _get_override_config_module()
    if override_module is not None:
        sdbench_config = override_module.get_sdbench_config(dataset, config_name)
        if sdbench_config:
            return sdbench_config

    if config_name == "science":
        # The config that is used in the peer reviewed Science paper
        # The baseline agent used for the Science paper is --agent baseline (default).
        if dataset == "mimic":
            gatekeeper_prompt_version = "v13_mimic"
            gatekeeper_model="gpt-5"
            judge_prompt_version = "v4_reasoning_with_tests"
        elif dataset == "nejm":
            gatekeeper_prompt_version = "v6_fixed"
            gatekeeper_model="o4-mini"
            judge_prompt_version = "v3_reasoning"
        else:
            raise ValueError(f"science config is only valid for nejm or mimic datasets, got dataset={dataset}")

        return SDBenchConfig(
            max_turns=70,
            gatekeeper_prompt_version=gatekeeper_prompt_version,
            gatekeeper_model=gatekeeper_model,
            judge_prompt_version=judge_prompt_version,
            judge_model="gpt-5",
        )

    if config_name == "default":
        # The current default config
        if dataset == "mimic":
            gatekeeper_prompt_version = "v13_mimic"
        else:
            # Use nejm gatekeeper by default for all other datasets
            gatekeeper_prompt_version = "v6_fixed"
        return SDBenchConfig(
            max_turns=70,
            gatekeeper_prompt_version=gatekeeper_prompt_version,
            gatekeeper_model="gpt-5",
            judge_prompt_version="v4_reasoning_with_tests",
            judge_model="gpt-5",
        )

    raise ValueError(f"Unknown configuration name: {config_name}")
