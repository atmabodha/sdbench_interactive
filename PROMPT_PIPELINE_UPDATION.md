# Antigravity CLI (agy) Instructions: SDBench Interactive Mode

## Goal
Modify the SDBench codebase to support a real-time, interactive diagnostic mode. The standard SDBench framework evaluates an `Agent` against an automated `Gatekeeper` (which reads from a static case file). This task requires bypassing the `Gatekeeper` and `Judge` entirely, allowing a human user to interact with the `Agent` via standard terminal I/O.

## Implementation Requirements

1. **Create a New Entry Point:**
   - Create a new file at `sdbench/cli/interactive.py`.
   - Do not modify the existing `sdbench/cli/harness.py` or `sdbench/cli/agent.py` files to preserve the original benchmark functionality.

2. **Isolate the Agent:**
   - Instantiate the diagnostic agent (defaulting to the `baseline` agent).
   - Ensure the script accepts command-line arguments for the `--model` (e.g., to pass an OpenRouter or local model string) and API keys if necessary, mirroring the argument parsing in the existing CLIs.

3. **Replace the Gatekeeper with Standard I/O (Human-in-the-Loop):**
   - Prompt the human user for an initial `case_summary` (e.g., "Enter your initial symptoms: ").
   - Initialize the agent's state with this summary.
   - Implement a standard `while` loop to drive the agent's turns.

4. **Handle Agent Actions:**
   - On each turn, generate the agent's next action (`ask_question`, `order_test`, or `commit_diagnosis`).
   - Parse the structured output from the agent.
   - **If `ask_question` or `order_test`:** 
     - Print the agent's question or test request clearly to the terminal.
     - Use Python's `input()` to pause execution and capture the human user's real-time response or test result.
     - Format the human's input as the required observation format (mimicking what the Gatekeeper would normally return) and append it to the agent's message history.
   - **If `commit_diagnosis`:**
     - Print the final diagnosis and recommended tests/treatments to the terminal.
     - Exit the loop gracefully.

5. **Error Handling & State Management:**
   - Ensure proper error handling if the agent outputs malformed JSON or invalid actions, prompting the agent to correct itself (similar to how the standard harness handles parsing errors).
   - Preserve the `Agent` class's expected internal state and message history structures so it functions identically to how it would in the benchmark.

## Usage 
Once implemented, the user should be able to run this mode using:
`uv run python -m sdbench.cli.interactive --agent baseline --model <model>`
