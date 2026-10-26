"""SDBench Prompts

This module manages the prompts used for SDBench.
Prompts are organized by ID and version in a structured directory format.

Directory Structure:
    prompts/
    ├── prompt_id/
    │   ├── v1/
    │   │   └── prompt.yaml
    │   ├── v2/
    │   │   └── prompt.yaml
    │   └── v3/
    │       └── prompt.yaml
    └── another_prompt/
        └── v1/
            └── prompt.yaml

YAML Format:
    Each prompt.yaml file must contain:
    -----
    template:
        system: "System prompt content..."
        user: "User prompt content..."
    -----
    Additional metadata can be included as additional fields, but will not be surfaced when
    loading a template.

Search Locations:
    1. Local: `<SDBENCH_LOCAL_DIR or CWD>/prompts/` (skipped when SDBENCH_LOCAL_DIR=disable)
    2. SDBench: sdbench/prompts/ directory in this package (i.e. sibling to this file)

    Prompt locations are searched in order, with the first match being returned.

API Usage:
----------
    # Load a specific prompt version
    template = sdbench.prompts.load_template("prompt_id", "v1")

    # Returns PromptTemplate dict with 'system' and 'user' keys
    print(template["system"])  # System prompt content
    print(template["user"])    # User prompt content

Command Line Interface:
-----------------------
    A CLI is provided to print prompt templates:

    > python -m sdbench.cli.prompts <prompt_id> <version>

    This will print both the system and user templates for the specified prompt.
"""

from prompts.registry import PromptTemplate, load_template

__all__ = ["PromptTemplate", "load_template"]
