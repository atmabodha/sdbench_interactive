# SDBench (Interactive Mode Edition)

This repository contains an updated version of [microsoft/sdbench](https://github.com/microsoft/sdbench/tree/main) (Sequential Diagnosis Benchmark) and is distributed under the same **MIT License**.

## Summary of Changes

- **Interactive Human-in-the-Loop Mode (`cli/interactive.py`):**
  - Added a new CLI entry point (`python -m cli.interactive`) that bypasses the automated `Gatekeeper` and `Judge` components, enabling real-time diagnostic interaction with a human user via standard terminal I/O (`input()` / `print()`).
  - Preserves existing benchmark entry points (`cli/harness.py` and `cli/agent.py`) and agent state/message structures without modification.
- **Action Handling & Error Recovery:**
  - Implemented `InteractiveTestRunner` and `InteractiveAgent` to drive sequential turns (`ask_question`, `order_test`, and `commit_diagnosis`), supporting both SDBench XML tags (`<question>`, `<test>`, `<diagnosis>`) and JSON action payloads.
  - Added graceful handling for malformed JSON or invalid actions, prompting the model to self-correct while preserving turn budgets and conversation history.
- **Environment-Driven Configuration (`.env`):**
  - Added automatic default resolution for `--model` (`SDBENCH_MODEL` / `MODEL` in `.env`) and `--agent` (defaulting to `baseline` or `SDBENCH_AGENT` in `.env`), plus direct `OPENAI_API_KEY` support alongside OpenRouter, Azure OpenAI, and local model endpoints.
- **Unit Tests (`cli/test_interactive.py`):**
  - Added unit tests covering action parsing, prefix cleaning, Gatekeeper-compatible observation formatting, multi-turn interactive loops, and CLI/environment resolution.

## Quick Start (Interactive Mode)

Configure your API key and default model in `.env`:

```env
OPENAI_API_KEY=<your-api-key>
SDBENCH_MODEL=o4-mini
```

Run the interactive diagnostic session:

```bash
uv run python -m cli.interactive
```

Or override the agent and model via CLI flags:

```bash
uv run python -m cli.interactive --agent baseline --model <model>
```

## License

Distributed under the **MIT License**, matching the upstream [microsoft/sdbench](https://github.com/microsoft/sdbench/tree/main) project.
