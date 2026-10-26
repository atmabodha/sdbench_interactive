import functools
import pathlib
from typing import Any, TypedDict
import yaml

from _paths import local_overrides_base


class PromptTemplate(TypedDict):
    """Versioned system and user prompt templates with metadata.

    Prompts can include placeholders like {CASE_CONTEXT} which should be replaced with actual
    content at runtime by calling str.format() on the prompt template.

    Metadata provides configuration information for how the prompt should be used.
    """

    system: str
    user: str
    # Metadata fields
    metadata: dict[str, Any]


_SDBENCH_PROMPTS_DIR = pathlib.Path(__file__).parent.resolve()


def _load(prompt_id: str, version: str) -> dict[str, Any]:
    # Local overrides (<SDBENCH_LOCAL_DIR or CWD>/prompts) take priority over the
    # packaged prompts; SDBENCH_LOCAL_DIR=disable skips them entirely. Resolved on
    # demand as the env var may be set after import (i.e. from .env).
    base_dirs = []
    local_base = local_overrides_base()
    if local_base is not None:
        base_dirs.append(local_base / "prompts")
    base_dirs.append(_SDBENCH_PROMPTS_DIR)

    for base_dir in base_dirs:
        yaml_path = base_dir / prompt_id / version / "prompt.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                print(f"Loading prompt {prompt_id} version {version} from {yaml_path}")
                return yaml.safe_load(f)
    dirs = [str(d) for d in base_dirs]
    raise FileNotFoundError(f"Prompt {prompt_id} version {version} not found. Looked in: {dirs}")


@functools.lru_cache()
def load_template(prompt_id: str, version: str) -> PromptTemplate:
    """Load a prompt template, returning system/user components with metadata.

    Args:
        prompt_id (str): Identifier for the prompt
        version (str): Version string (e.g., 'v1').

    Returns:
        PromptTemplate: A dict with 'system', 'user', and metadata fields

    Raises:
        FileNotFoundError: If the prompt or version doesn't exist
    """
    prompt_data = _load(prompt_id, version)
    if not prompt_data or "template" not in prompt_data:
        raise ValueError(f"Prompt '{prompt_id}' version '{version}' missing 'template' field")

    template = prompt_data["template"]
    if not isinstance(template, dict):
        raise ValueError(f"Invalid template format for '{prompt_id}' version '{version}'")

    if "system" not in template or "user" not in template:
        raise ValueError(
            f"Template for '{prompt_id}' version '{version}' must contain 'system' and 'user' keys"
        )

    # Extract metadata with defaults
    metadata = prompt_data.get("metadata", {})
    return {
        "system": template["system"],
        "user": template["user"],
        "metadata": metadata,
    }
