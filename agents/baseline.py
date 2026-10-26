import re
from typing import Any

from sdbench.config import AgentConfig
from sdbench.gatekeeper import TestRunner
from sdbench.models import ChatCompletionMessageParam, ModelCreateChatCompletion


def extract_test_list_from_response(response_content: str) -> list[str]:
    """Extract a list of test names and questions from the agent's response.
    Looks for <test>test_name</test> and <question>question_text</question> tags.
    """
    # Single pattern to match both test and question tags in order
    pattern = r"<(test|question)>(.*?)</\1>"

    matches = re.findall(pattern, response_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    # Process matches in the order they appear
    all_items = []

    for tag_type, content in matches:
        item = content.strip()
        if item:  # Only check if item exists, no length limit
            if tag_type.lower() == "test":
                all_items.append("Test to order: " + item)
            elif tag_type.lower() == "question":
                all_items.append("Question for the patient: " + item)

    return all_items


def extract_diagnosis_from_response(response_content: str) -> str | None:
    """Extract diagnosis from <diagnosis>...</diagnosis> tags."""
    pattern = r"<diagnosis>(.*?)</diagnosis>"
    match = re.search(pattern, response_content, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def run(case_summary: str, test_runner: TestRunner, agent_llm: ModelCreateChatCompletion, collector: dict[str, Any], config: AgentConfig) -> str:
    if case_summary is None:
        case_summary = "A new patient case is available. You need to ask questions and order tests to reach a diagnosis."

    system_prompt = """\
You are a diagnostic assistant. Order tests and ask patient questions to determine the diagnosis.

To order tests use <test></test> tags:
<test>CBC</test>
<test>Chest X-ray</test>
...more tests...

You can also ask questions directly (make sure to put each question in a separate <question> tag):
<question>What are your symptoms?</question>
<question>What is your medical history?</question>
...more questions...

You cannot mix <test> and <question> tags in the same turn, just use all <test> tags or all <question> tags. Make sure to ask for enough questions and tests to reach a diagnosis.

When ready to diagnose, use <diagnosis></diagnosis> tags to provide ONE final diagnosis:
<diagnosis>Your diagnosis here</diagnosis>
"""

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Patient presentation:\n\n{case_summary}\n\nWhat questions would you like to ask?\nOr what tests would you like to order?",
        },
    ]

    while test_runner.turns_left > 0:
        # Get model response
        response = agent_llm(messages=messages)
        content = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        if config.debug:
            print(f"\nAssistant: {content}")

        # Check for diagnosis
        diagnosis = extract_diagnosis_from_response(content)
        if diagnosis:
            collector["agent_messages"] = messages
            return diagnosis

        # Extract and run tests
        test_names = extract_test_list_from_response(content)
        if test_names:
            # Run tests
            batch_results = test_runner(test_names)
            results = batch_results.test_results
            limit_messages = batch_results.limit_messages

            # Format results
            results_text = "Results:\n"
            for result in results:
                results_text += f"- {result['test_name']}: {result['results']}\n"

            # Add any limit messages
            if limit_messages:
                results_text += f"\nSYSTEM NOTICE: {'; '.join(limit_messages)}\n"

            messages.append({"role": "user", "content": results_text})

            if config.debug:
                print(f"\nUser: {results_text}")
        else:
            # No tests found, prompt for action
            messages.append(
                {
                    "role": "user",
                    "content": "Please order tests using <test> tags, questions using <question> tags, or provide diagnosis using <diagnosis> tags.",
                }
            )

    # Out of turns - force diagnosis
    messages.append(
        {
            "role": "user",
            "content": "No more tests available. Please provide your diagnosis using <diagnosis> tags.",
        }
    )

    if config.debug:
        print(
            "\nUser: No more tests available. Please provide your diagnosis using <diagnosis> tags."
        )

    response = agent_llm(messages=messages)
    content = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": content})

    if config.debug:
        print(f"\nAssistant: {content}")

    diagnosis = extract_diagnosis_from_response(content)
    collector["agent_messages"] = messages

    return diagnosis if diagnosis else "Unable to provide diagnosis"
