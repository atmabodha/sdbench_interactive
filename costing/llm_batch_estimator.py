import re
import time
from typing import TYPE_CHECKING

from config import CostingConfig
from models import (
    ChatCompletionMessageParam,
    ModelCreateChatCompletion,
    as_assistant_message,
    create_chat_completion,
)

if TYPE_CHECKING:
    from costing.cost_types import CostMethodEstimate


def cost_tests_batch(
    test_names: list[str],
    config: CostingConfig | None,
    llm: ModelCreateChatCompletion | None = None,
) -> dict[str, "CostMethodEstimate"]:
    """Use LLM to estimate costs for multiple tests in a single batch call.

    Args:
        test_names: List of test names to price (should all be uncached)
        config: CostingConfig containing costing parameters

    Returns:
        Dict mapping test_name -> CostMethodEstimate
        The test_name keys are normalized (case-insensitive, stripped).
    """
    if not test_names:
        return {}

    start_time = time.time()

    system_prompt = """You are a medical billing expert. Given a list of medical tests, provide cost estimates in USD for typical hospital/lab pricing in the United States.

For each test, respond in this EXACT format:
Test: [test name]
Cost: $[estimate]
Explanation: [brief explanation]

IMPORTANT: Respond with estimates in the SAME ORDER as the tests are provided in the numbered list. Do not reorder or skip any tests.

Be consistent with the format and use realistic pricing. Provide a single cost estimate (not a range)."""

    numbered_tests = "\n".join(f"{i + 1}. {test}" for i, test in enumerate(test_names))
    user_prompt = f"""Please estimate the costs for the following {len(test_names)} medical tests:

{numbered_tests}

Please provide cost estimates in the specified format for each test, in the same order."""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        if llm is not None:
            response = llm(messages=messages)
        else:
            assert config is not None
            response = create_chat_completion(
                model=config.costing_model,
                messages=messages,
                debug=config.debug,
                **config.costing_model_args,
            )
        messages.append(as_assistant_message(response.choices[0].message))
        cost_analysis_text = response.choices[0].message.content or ""
        cost_analysis_text = cost_analysis_text.strip()
        if not cost_analysis_text:
            print(f"Warning: Empty response from LLM for batch cost estimation: {response}")

        elapsed_time = time.time() - start_time

        # Parse the response to extract costs for each test
        results: dict[str, CostMethodEstimate] = {}

        # Parse each test block and match by order
        test_blocks = cost_analysis_text.split("\n\n")
        parsed_costs: list[tuple[float, str] | None] = []

        for block in test_blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 2:
                test_line = next((line for line in lines if line.startswith("Test:")), None) or ""
                cost_line = next((line for line in lines if line.startswith("Cost:")), None)

                if cost_line:
                    test_name = test_line.replace("Test:", "").strip()

                    # Extract cost using regex for single estimate
                    cost_match = re.search(r"\$(\d+(?:,\d+)?(?:\.\d+)?)", cost_line)
                    if cost_match:
                        cost = float(cost_match.group(1).replace(",", ""))
                        parsed_costs.append((cost, test_name))
                    else:
                        parsed_costs.append(None)
                else:
                    parsed_costs.append(None)

        # Match parsed costs to test names by order
        for i, test_name in enumerate(test_names):
            normalised_test_name = test_name.lower().strip()

            if i < len(parsed_costs) and parsed_costs[i] is not None:
                cost_value, parsed_test_name = parsed_costs[i]  # type: ignore
                results[normalised_test_name] = {
                    "messages": messages,
                    "cost": cost_value,
                    "batch_test_name": parsed_test_name,
                    "time": elapsed_time,  # Total time for the batch call
                }
            else:
                if config is not None and config.debug:
                    print(
                        f"Could not parse cost for test '{test_name}' at position {i + 1} from LLM response."
                    )
                results[normalised_test_name] = {
                    "messages": messages,
                    "error": f"Could not parse cost from batch response for '{test_name}' at position {i + 1}",
                    "time": elapsed_time,
                }

        return results

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Warning: Batch LLM cost estimation failed: {e}")

        # Return error for all tests
        return {
            test_name: {
                "messages": messages,
                "error": str(e),
                "time": elapsed_time,
            }
            for test_name in test_names
        }
