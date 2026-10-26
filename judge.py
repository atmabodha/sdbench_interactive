from dataclasses import dataclass
import re
from typing import Any, TypedDict

import case_data
import models
import prompts
from config import SDBenchConfig


@dataclass(frozen=True)
class PromptConfig:
    """Configuration derived from prompt metadata."""

    min_score: int
    max_score: int
    supports_reasoning: bool


def parse_prompt_config(prompt_template: prompts.PromptTemplate) -> PromptConfig:
    """Parse prompt configuration from template metadata."""
    score_range = prompt_template["metadata"].get(
        "score_range", "1-5"
    )  # Default to legacy 1-5 range
    supports_reasoning = prompt_template["metadata"].get(
        "supports_reasoning", False
    )  # Default to False for backwards compatibility

    # Parse score range (e.g., "1-5" or "2-5")
    min_score, max_score = map(int, score_range.split("-"))

    return PromptConfig(
        min_score=min_score, max_score=max_score, supports_reasoning=supports_reasoning
    )


class CheckDiagnosisResponse(TypedDict, total=False):
    """Response data from the judge.

    Attributes:
       check_diagnosis_messages: The list of messages exchanged with the model during diagnosis checking.
       check_diagnosis_output: The raw output text from the model.
       diagnosis_score: The numeric score assigned by the model (range determined by prompt metadata).
       diagnosis_reasoning: The reasoning provided by the model (optional, only present when reasoning is enabled).
    """

    check_diagnosis_messages: Any
    check_diagnosis_output: str
    diagnosis_score: int
    diagnosis_reasoning: str  # Optional field for backward compatibility


def _format_test_logs_for_prompt(test_logs: list[dict[str, Any]]) -> str:
    """Render ordered tests into a concise, readable block for the judge prompt."""
    if not test_logs:
        return ""

    # Sort alphabetically by test name (case-insensitive) to hide original ordering
    sorted_logs = sorted(
        test_logs,
        key=lambda log: (str(log.get("test_name") or "").lower(), str(log.get("result") or "")),
    )

    formatted_entries: list[str] = []
    for idx, log in enumerate(sorted_logs, start=1):
        name = log.get("test_name") or "Unknown test"
        result = log.get("result") or "Result not documented."
        is_synth = log.get("is_synthetic", False)
        reason = log.get("reason")
        annotations: list[str] = []
        if is_synth:
            annotations.append("synthetic")
        if reason:
            annotations.append(reason.strip())

        header = f"{idx}. {name}"
        if annotations:
            header += f" ({'; '.join(annotations)})"

        formatted_entries.append(f"{header}\n    Result: {result.strip()}")

    return "\n".join(formatted_entries)


def check_diagnosis(
    config: SDBenchConfig,
    diagnosis: str,
    case: case_data.CaseData,
    test_logs: list[dict[str, Any]] | None = None,
    llm: models.ModelCreateChatCompletion | None = None,
) -> tuple[bool, CheckDiagnosisResponse]:
    """Check if a diagnosis is correct using an AI judge.

    This function evaluates a candidate diagnosis against the reference diagnosis
    for a given case using a language model as a judge. The judge considers both
    the semantic similarity and clinical accuracy of the diagnosis.

    Args:
        config: Configuration for the judge, including model, prompt version, and options.
        diagnosis: The candidate diagnosis to evaluate.
        case: Complete case information.
        test_logs: Optional list of ordered test logs to include in the evaluation.
        llm: Optional pre-built chat completion callable. Overrides config.judge_model.

    Returns:
        A tuple containing:
        - bool: Whether the diagnosis passed (score >= 4 out of 5)
        - CheckDiagnosisResponse: Detailed response data

    Raises:
        ValueError: If the judge response is invalid.
    """
    if config.debug:
        model_str = config.judge_model
        if config.judge_model_args:
            model_str += f" ({', '.join(f'{k}={v}' for k, v in config.judge_model_args.items())})"
        print(
            f"Using model {model_str} and prompt version {config.judge_prompt_version} for diagnosis checking"
        )

    prompt_template = prompts.load_template("check_diagnosis", config.judge_prompt_version)

    # Parse configuration from template metadata
    prompt_config = parse_prompt_config(prompt_template)

    # The pass threshold below (score >= 4) is hardcoded for the legacy 1-5 Likert
    # scale. Fail loudly rather than silently mis-scoring a prompt that declares a
    # different range.
    if (prompt_config.min_score, prompt_config.max_score) != (1, 5):
        raise ValueError(
            f"check_diagnosis only supports a 1-5 score range, but judge prompt "
            f"'{config.judge_prompt_version}' declares a "
            f"{prompt_config.min_score}-{prompt_config.max_score} range. The pass "
            "threshold (score >= 4) is hardcoded for the 1-5 scale."
        )

    case_context = case["full_text"]

    tests_section = ""
    if test_logs:
        tests_formatted = _format_test_logs_for_prompt(test_logs)
        if tests_formatted:
            tests_section = f"Tests Ordered by the Agent and Their Results:\n{tests_formatted}"

    system_prompt = prompt_template["system"]
    system_prompt = system_prompt.replace("{{CASE_CONTEXT}}", case_context)
    system_prompt = system_prompt.replace("{{TEST_LOGS}}", tests_section)

    user_prompt_template = prompt_template["user"]
    user_prompt = user_prompt_template.replace("{{CANDIDATE_DIAGNOSIS}}", diagnosis)
    user_prompt = user_prompt.replace("{{REFERENCE_DIAGNOSIS}}", case["final_diagnosis"])

    messages: list[models.ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if llm is not None:
        output = models.call_llm_with_retry(llm, messages=messages)
    else:
        output = models.create_chat_completion(
            model=config.judge_model,
            messages=messages,
            debug=config.debug,
            **config.judge_model_args,
        )
    completion_message = output.choices[0].message

    if config.debug:
        models.print_messages(messages, completion_message)

    messages.append(models.as_assistant_message(completion_message))

    if not completion_message.content:
        raise ValueError(f"No content in model response: {output}")

    text = completion_message.content.strip()

    # Parse response based on whether reasoning was requested
    reasoning = ""
    if prompt_config.supports_reasoning:
        # Look for structured format: **REASONING**: ... **SCORE**: X
        reasoning_match = re.search(
            r"\*\*REASONING\*\*:\s*(.*?)\s*\*\*SCORE\*\*:", text, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Extract score from SCORE section or fallback to any digit
        score_pattern = f"[{prompt_config.min_score}-{prompt_config.max_score}]"
        score_match = re.search(rf"\*\*SCORE\*\*:\s*({score_pattern})", text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        else:
            # Fallback to any digit in the valid range
            digit_match = re.search(score_pattern, text)
            if digit_match:
                score = int(digit_match.group(0))
            else:
                raise ValueError(f"Could not extract score from response: {text}")
    else:
        # Original parsing for non-reasoning responses
        score_pattern = f"[{prompt_config.min_score}-{prompt_config.max_score}]"
        match = re.search(score_pattern, text)
        if match:
            score = int(match.group(0))
        else:
            raise ValueError(f"Could not extract score from response: {text}")

    response_data: CheckDiagnosisResponse = {
        "check_diagnosis_messages": messages,
        "check_diagnosis_output": text,
        "diagnosis_score": score,
    }

    # Add reasoning field only if reasoning was provided
    if prompt_config.supports_reasoning and reasoning:
        response_data["diagnosis_reasoning"] = reasoning
    passed = score >= 4
    return passed, response_data
