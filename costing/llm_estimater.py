import re
import time
from typing import TYPE_CHECKING

from config import CostingConfig
from models import ChatCompletionMessageParam, as_assistant_message, create_chat_completion

if TYPE_CHECKING:
    from costing.cost_types import CostMethodEstimate


def cost_test(test_name: str, config: CostingConfig) -> "CostMethodEstimate":
    """Use LLM to estimate cost for a single test."""
    start_time = time.time()

    system_prompt = """You are a medical billing expert. Given a medical test name, provide a cost estimate in USD for typical hospital/lab pricing in the United States.

Respond with ONLY the dollar amount (no $ symbol, no explanation). For example: 150"""

    user_prompt = f"What is the typical cost for this medical test: {test_name}"

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = create_chat_completion(
            model=config.costing_model,
            messages=messages,
            debug=config.debug,
            **config.costing_model_args,
        )
        messages.append(as_assistant_message(response.choices[0].message))
        cost_text = response.choices[0].message.content or ""
        cost_text = cost_text.strip()

        # Remove $ signs and commas just in case, then extract number (including decimals)
        cost_text = cost_text.replace("$", "").replace(",", "").strip()
        cost_match = re.search(r"(\d+\.?\d*)", cost_text)

        elapsed_time = time.time() - start_time

        if cost_match:
            return {
                "messages": messages,
                "cost": float(cost_match.group(1)),
                "time": elapsed_time,
            }
        else:
            return {
                "messages": messages,
                "error": f"Could not parse cost from response: '{cost_text}'",
                "time": elapsed_time,
            }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Warning: LLM cost estimation failed for '{test_name}': {e}")
        return {
            "messages": messages,
            "error": str(e),
            "time": elapsed_time,
        }
