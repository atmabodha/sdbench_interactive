import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Protocol, runtime_checkable

from _paths import local_overrides_base
from config import AgentConfig
from gatekeeper import TestRunner
from models import ModelCreateChatCompletion


@runtime_checkable
class DiagnosticAgent(Protocol):
    """Protocol defining the expected interface for diagnostic agent modules."""

    def run(
        self,
        case_summary: str,
        test_runner: TestRunner,
        agent_llm: ModelCreateChatCompletion,
        collector: dict[str, Any],
        config: AgentConfig,
    ) -> str:
        """Run the agent on one case and return its final diagnosis.

        Called once per case. The agent proceeds turn by turn, and each single turn is
        one of two mutually exclusive actions: gather more information, or commit a
        diagnosis. A turn that gathers information orders diagnostic tests and/or asks the
        patient questions -- submitted as a batch via ``test_runner`` -- while a turn that
        commits returns the final diagnosis. The turn budget is bounded by
        ``test_runner.turns_left`` / ``test_runner.max_turns``.

        How the agent decides between these actions, and the format it uses to do so, is
        entirely up to the implementation.

        Args:
            case_summary: Short case presentation that does not reveal the diagnosis.
            test_runner: Callable used to order a turn's batch of tests / questions; also
                exposes the turn budget (``turns_left`` / ``max_turns``).
            agent_llm: Chat-completion callable for the model under test.
            collector: Mutable dict the agent populates with run artifacts (e.g.
                ``agent_messages``) for later reporting.
            config: Agent configuration (model args, debug flag, etc.).

        Returns:
            The agent's final diagnosis string.
        """
        ...


_SDBENCH_AGENTS_DIR = Path(__file__).parent


def _import_agents_from(directory: Path, package: str | None = None):
    """Import all .py agent modules from a directory into this namespace."""
    if not directory.is_dir():
        return
    is_package = (directory / "__init__.py").exists()
    for file_path in directory.glob("*.py"):
        if file_path.name == "__init__.py":
            continue
        module_name = file_path.stem
        try:
            if package:
                module = importlib.import_module(f".{module_name}", package=package)
            elif is_package:
                pkg_name = directory.name
                if pkg_name not in sys.modules:
                    importlib.import_module(pkg_name)
                module = importlib.import_module(f"{pkg_name}.{module_name}")
            else:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    continue
            globals()[module_name] = module
        except ImportError as e:
            print(f"Warning: Could not import {module_name} from {directory}: {e}")


# Load packaged agents first, then local overrides from
# <SDBENCH_LOCAL_DIR or CWD>/diagnostic_agents (skipped when SDBENCH_LOCAL_DIR=disable).
_import_agents_from(_SDBENCH_AGENTS_DIR, package=__name__)
_local_base = local_overrides_base()
if _local_base is not None:
    _import_agents_from((_local_base / "diagnostic_agents").resolve())

__all__ = ["DiagnosticAgent"]
