"""SDBench Gatekeeper module for responding to requests for a diagnostic test result or patient questions.

For legacy reasons the term "test" is used in the API and code to refer to both "diagnostic tests" and
"patient questions". When meaning only diagnostic tests, the term "diagnostic test" is used.
The terminology of "ordering" a test is also used, and refers to both requesting a test result or asking a
patient a question.

When the term "test name" is used it refers to the instruction to order a specific diagnostic test
(ie: chest CT scan) or ask a specific patient question (ie: have you felt short of breath).
The test name for a patient question must be prefixed with "Question for the patient: ".
The test name for a diagnostic test should be prefixed with "Test to order: ".
If a test name has neither prefix, it is treated as a diagnostic test.
"""

from dataclasses import dataclass
import datetime
import re
from typing import Any, Iterable, Protocol, TypedDict

import prompts
from case_data import CaseData
from config import SDBenchConfig
from models import (
    ChatCompletionMessageParam,
    ModelCreateChatCompletion,
    as_assistant_message,
    call_llm_with_retry,
    create_chat_completion,
    print_messages,
)


class OrderTestResponse(TypedDict):
    """Response for a single ordered test or patient question."""

    test_name: str
    results: str
    is_synthetic: bool


def _parse_xml_test_response(
    test_names: list[str], raw_response: str, debug: bool = False
) -> tuple[list[OrderTestResponse], int]:
    """Parse XML gatekeeper prompt response containing test_result and/or patient_response blocks.

    Args:
        test_names: List of test names requested (for matching order).
        raw_response: Raw XML response from LLM
        debug: Whether to print debug information

    Returns:
        A list of OrderTestResponse in matching order to test_names, and a count of those that were successful.
    """
    # Try test_result pattern first (for diagnostic tests) - handle both quoted and unquoted IDs
    test_block_pattern = r'<test_result id="?(\d+)"?>(.*?)</test_result>'
    test_block_matches = re.findall(test_block_pattern, raw_response, re.DOTALL)

    # Try patient_response pattern (for patient questions) - handle both quoted and unquoted IDs
    patient_block_pattern = r'<patient_response id="?(\d+)"?>(.*?)</patient_response>'
    patient_block_matches = re.findall(patient_block_pattern, raw_response, re.DOTALL)

    # Combine both types of matches
    all_matches = test_block_matches + patient_block_matches

    if debug and len(all_matches) == 0:
        print("[DEBUG] No XML blocks found in response")
        print(f"[DEBUG] Raw response preview: {raw_response[:500]}")
        print(f"[DEBUG] Looking for test pattern: {test_block_pattern}")
        print(f"[DEBUG] Looking for patient pattern: {patient_block_pattern}")

    response_by_id: dict[int, OrderTestResponse] = {}
    for raw_test_id, raw_test_content in all_matches:
        test_id = int(raw_test_id)
        test_content = raw_test_content.strip()

        # Try <result> tag first (for diagnostic tests)
        result_pattern = r"<result>\s*(.*?)\s*</result>"
        result_match = re.search(result_pattern, test_content, re.DOTALL)
        if result_match:
            content = result_match.group(1).strip()
        else:
            # Try <response> tag (for patient questions)
            response_pattern = r"<response>\s*(.*?)\s*</response>"
            response_match = re.search(response_pattern, test_content, re.DOTALL)
            if response_match:
                content = response_match.group(1).strip()
            else:
                # No <result> or <response> tag found, skip this block
                if debug:
                    print(
                        f"[DEBUG] No <result> or <response> tag found in block ID {test_id} content: {test_content}"
                    )
                continue

        # Extract is_synthetic flag from within the block
        synthetic_pattern = r"<is_synthetic>\s*(true|false)\s*</is_synthetic>"
        synthetic_match = re.search(synthetic_pattern, test_content, re.IGNORECASE)
        if synthetic_match:
            is_synthetic = synthetic_match.group(1).lower() == "true"
        else:
            if debug:
                print(
                    f"[DEBUG] No <is_synthetic> tag found in block ID {test_id}, defaulting to False"
                )
            is_synthetic = False  # Default to False if not specified

        response_by_id[test_id] = OrderTestResponse(
            {
                "test_name": "",  # Placeholder, will fill in below
                "results": content,
                "is_synthetic": is_synthetic,
            }
        )

    results: list[OrderTestResponse] = []
    num_successful = 0
    for test_id, test_name in enumerate(test_names, 1):
        result = response_by_id.get(test_id)
        if result:
            result["test_name"] = test_name
            num_successful += 1
        else:
            # No result found for this test, so return the placeholder "Results not available" results
            result = OrderTestResponse(
                {
                    "test_name": test_name,
                    "results": f"Results not available for {test_name}",
                    "is_synthetic": False,
                }
            )
            if debug:
                print(f"[DEBUG] No result found for test '{test_name}' with expected ID {test_id}")
        results.append(result)
    return results, num_successful


def order_test(
    config: SDBenchConfig,
    case: CaseData,
    test_name: str,
    previous_tests: Iterable[str] | None = None,
    collector: dict[str, Any] | None = None,
) -> OrderTestResponse:
    """Request a single diagnostic test or patient question.
    Delegates to order_tests().
    See order_tests() for method details.
    """
    test_responses = order_tests(
        config=config,
        case=case,
        test_names=[test_name],
        previous_tests=previous_tests,
        collector=collector,
    )
    return test_responses[0]


def order_tests(
    config: SDBenchConfig,
    case: CaseData,
    test_names: list[str],
    previous_tests: Iterable[str] | None = None,
    collector: dict[str, Any] | None = None,
    llm: ModelCreateChatCompletion | None = None,
) -> list[OrderTestResponse]:
    """Order a batch of diagnostic tests.

    Args:
        config: Configuration for the gatekeeper.
        case: The case data to source responses from.
        test_names: List of test names to return results/responses for. See module docstring for details.
        previous_tests: List of test names of previously completed tests.
        collector: Optional dictionary to store gatekeeper LLM messages and other metadata.
        llm: Optional pre-built chat completion callable. Overrides config.gatekeeper_model.

    Returns:
        List of OrderTestResponse, exactly one for each test.
            If the gatekeeper failed to process an individual test then the "results" for that test will be
            "Results not available for {test_name}".
    """
    if not test_names:
        return []

    # Use provided collector dict or create a dummy one if not provided
    if collector is None:
        collector = {}

    # Helper function to split list into chunks of maximum size
    def _chunk_list(all_tests: list[str], chunk_size: int):
        for i in range(0, len(all_tests), chunk_size):
            yield all_tests[i : i + chunk_size]

    # Split test_names into chunks of at most chunk_size tests
    test_names_chunks = list(_chunk_list(test_names, config.gatekeeper_chunk_size))

    if config.debug:
        print(
            f"[order_tests] Splitting {len(test_names)} tests into {len(test_names_chunks)} chunks"
        )

    # Initialization or defensive copy of initial previous_tests to append to
    previous_tests = list(previous_tests) if previous_tests else []
    all_results: list[OrderTestResponse] = []

    for chunk_idx, test_names_chunk in enumerate(test_names_chunks, 1):
        if config.debug:
            print(
                f"[order_tests] Processing chunk {chunk_idx}/{len(test_names_chunks)} with {len(test_names_chunk)} tests"
            )

        all_results.extend(
            _order_tests_chunk(
                config, case, chunk_idx, test_names_chunk, previous_tests, collector, llm=llm
            )
        )
        previous_tests.extend(test_names_chunk)

    if config.debug:
        print(
            f"[order_tests] Completed processing all {len(test_names_chunks)} chunks, returning {len(all_results)} results"
        )

    return all_results


def _order_tests_chunk(
    config: SDBenchConfig,
    case: CaseData,
    chunk_idx: int,
    test_names: list[str],
    previous_tests: Iterable[str],
    collector: dict[str, Any],
    llm: ModelCreateChatCompletion | None = None,
) -> list[OrderTestResponse]:
    # Initialize gatekeeper messages collection
    gatekeeper_messages = collector.setdefault("gatekeeper_messages", [])

    if config.debug:
        print(
            f"[order_tests] Using model {config.gatekeeper_model} ({config.gatekeeper_model_args}) and prompt version {config.gatekeeper_prompt_version}"
        )

    case_text = case["full_text"]
    prompt_template = prompts.load_template(
        "gatekeeper_synthetic_tests", config.gatekeeper_prompt_version
    )

    # Format previously requested tests for the template
    if previous_tests:
        # Format as a simple numbered list of previously requested tests
        formatted_previous_tests = "\n".join(
            [f"{i}. {test}" for i, test in enumerate(previous_tests, 1)]
        )
    else:
        formatted_previous_tests = "None - this is the first request."

    system_prompt = prompt_template["system"].format(
        case_text=case_text,
        previous_tests=formatted_previous_tests,
        final_diagnosis=case["final_diagnosis"],
    )

    # Format the test names as a numbered list for this chunk
    formatted_requested_tests = "\n".join(
        [f"ID {i}: {test_name}" for i, test_name in enumerate(test_names, 1)]
    )

    user_prompt = prompt_template["user"].format(requested_tests=formatted_requested_tests)

    # Try up to 3 times (original + 2 retries)
    raw_response = None
    for attempt in range(3):
        # For retries, add stronger emphasis on the expected format
        if attempt > 0:
            # Add format emphasis to user prompt for retries
            format_emphasis = f"""

IMPORTANT: You MUST respond in the exact XML format specified. Do not ask questions or provide any other text.

For each of the {len(test_names)} tests or questions listed, provide exactly one <test_result> block with sequential IDs 1 through {len(test_names)}.

Example format:
<test_result id="1">
<result>
[test result here]
</result>
<is_synthetic>true</is_synthetic>
</test_result>
<test_result id="2">
<result>
[test result here]
</result>
<is_synthetic>false</is_synthetic>
</test_result>

Provide results for ALL {len(test_names)} tests in this exact format."""

            retry_user_prompt = user_prompt + format_emphasis

            # Include the previous failed response in conversation history only on first retry
            if attempt == 1:
                # First retry - include previous response context
                messages: Iterable[ChatCompletionMessageParam] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw_response},  # Previous failed response
                    {"role": "user", "content": retry_user_prompt},
                ]
                if config.debug:
                    print(
                        f"[order_tests] Chunk {chunk_idx}: First retry - adding format emphasis and previous response context"
                    )
            else:
                # Second retry - just format emphasis, no previous response
                messages: Iterable[ChatCompletionMessageParam] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_user_prompt},
                ]
                if config.debug:
                    print(
                        f"[order_tests] Chunk {chunk_idx}: Second retry - adding format emphasis only"
                    )

        else:
            # First attempt
            messages: Iterable[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        if llm is not None:
            result = call_llm_with_retry(llm, messages=messages)
        else:
            result = create_chat_completion(
                model=config.gatekeeper_model,
                messages=messages,
                debug=config.debug,
                **config.gatekeeper_model_args,
            )
        result_message = result.choices[0].message

        conversation_messages = [*messages, as_assistant_message(result_message)]
        gatekeeper_messages.append(conversation_messages)

        raw_response = result_message.content or ""
        raw_response = raw_response.strip()

        test_results, num_successful = _parse_xml_test_response(
            test_names, raw_response, config.debug and attempt == 0
        )  # Only debug on first attempt

        # Check if we got the expected number of test blocks
        expected_count = len(test_names)
        actual_count = num_successful

        if actual_count == expected_count:
            # Success - we got the right number of blocks
            if config.debug and attempt > 0:
                print(
                    f"[order_tests] Chunk {chunk_idx}: Retry {attempt} successful - got {actual_count} test blocks as expected"
                )
            return test_results
        elif attempt < 2:
            # Not the final attempt, log and retry
            if config.debug:
                print(
                    f"[order_tests] Chunk {chunk_idx}: Expected {expected_count} test blocks, got {actual_count}. Retrying with format emphasis... (attempt {attempt + 1}/3)"
                )
                print(f"Raw response: {raw_response}")
        else:
            # Final attempt also failed, log and continue with what we have
            if config.debug:
                print(
                    f"[order_tests] Chunk {chunk_idx}: All retries failed - still got {actual_count} test blocks instead of {expected_count}. Proceeding with available results."
                )
            return test_results

    raise RuntimeError("Unreachable code")  # For the type checker


@dataclass(frozen=True)
class TestRunnerResponse:
    """Result of a batch of ordered tests, including any limit notifications."""

    test_names: list[str]
    test_results: list[OrderTestResponse]
    limit_messages: list[str]

    def format_test_results(self) -> str:
        """Format test results including any limit messages.

        Returns:
            str: Formatted results text including any limit messages.
        """
        # Format the test results
        results_text = "Test Results:\n"
        for i, (test_name, test_result) in enumerate(zip(self.test_names, self.test_results)):
            results_text += f"{i + 1}. {test_name}: {test_result['results']}\n"

        if self.limit_messages:
            results_text += f"\nSYSTEM NOTICE: {'; '.join(self.limit_messages)}\n"

        return results_text


class TestRunner(Protocol):
    """TestRunner interface that is exposed to diagnostic agents."""

    def __call__(self, test_names: list[str]) -> TestRunnerResponse:
        """Process a batch of test requests and return their results.

        Args:
            test_names: List of test names to process. See module docstring for details.

        Returns:
            TestRunnerResponse containing test results and any limit messages
        """
        ...

    @property
    def max_turns(self) -> int:
        """Maximum number of turns allowed for this test runner."""
        ...

    @property
    def turns_left(self) -> int:
        """Get the number of turns remaining."""
        ...


class BatchTestRunner:
    """Batch test runner for processing multiple diagnostic tests.

    This class manages test execution, turn counting, and result tracking
    for diagnostic agents running on clinical cases.
    """

    def __init__(
        self,
        config: SDBenchConfig,
        case: CaseData,
        collector: dict[str, Any],
        llm: ModelCreateChatCompletion | None = None,
    ):
        """Args:
        config: Configuration for the test runner.
        case: The clinical case data.
        collector: Dictionary to store results of processing, including logs and metadata.
        """
        self._config = config
        self._case = case
        self._debug = config.debug
        self._llm = llm

        self._max_turns = self._config.max_turns
        self._turns_left = self._max_turns
        self._completed_tests: list[str] = []
        self._requested_tests: list[str] = []
        self._num_rounds = 0
        self._num_test_rounds = 0
        self._collector = collector
        self._collector.setdefault("test_logs", [])  # Initialize test_logs in data

    def __call__(self, test_names: list[str]) -> TestRunnerResponse:
        """Process a batch of test requests and return their results.

        Args:
            test_names: List of tests to process

        Returns:
            List of results, one for each test requested
        """
        if not isinstance(test_names, Iterable):
            raise ValueError("test_descriptions must be an Iterable")

        # Check turn limit before processing
        if self._turns_left <= 0:
            raise ValueError("No turns left")

        # Apply question and test limits if configured
        test_names, limit_messages = self._parse_and_limit_requests(test_names)

        if limit_messages and self._debug:
            print(f"[BatchTestRunner] Applied limits: {'; '.join(limit_messages)}")

        # Check if this round contains any tests (not just questions)
        has_tests = any(
            not test_name.startswith("Question for the patient:") for test_name in test_names
        )

        # Update state
        self._turns_left -= 1
        self._requested_tests.extend(test_names)
        self._num_rounds += 1

        if has_tests:
            self._num_test_rounds += 1

        if self._debug:
            print(f"[BatchTestRunner] Using batch mode for {len(test_names)} tests")

        batch_results = order_tests(
            config=self._config,
            case=self._case,
            test_names=test_names,
            previous_tests=self._completed_tests,
            collector=self._collector,
            llm=self._llm,
        )

        # Update completed tests
        for result in batch_results:
            if "test_name" in result and result["test_name"] not in self._completed_tests:
                self._completed_tests.append(result["test_name"])

        # Add batch results to test logs
        for result in batch_results:
            self._collector["test_logs"].append(
                {
                    "test_name": result.get("test_name", "Unknown"),
                    "reason": result.get("reason", ""),
                    "result": result.get("results", ""),
                    "is_synthetic": result.get("is_synthetic", False),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "batch_mode": True,
                }
            )

        return TestRunnerResponse(
            test_names=test_names,
            test_results=batch_results,
            limit_messages=limit_messages,
        )

    def _parse_and_limit_requests(self, test_names: list[str]) -> tuple[list[str], list[str]]:
        """Parse a list of test descriptions and separate questions from tests, applying limits.

        Args:
            test_names: List of test descriptions that may include questions and tests
            max_questions_per_turn: Maximum number of questions to allow per turn
            max_tests_per_turn: Maximum number of tests to allow per turn
            enforce_single_turn_type: If True, only allow questions OR tests per turn, not both

        Returns:
            tuple: (limited_requests, limit_messages)
                - limited_requests: List of allowed requests (questions + tests, up to limits)
                - limit_messages: List of messages about any limits that were applied
        """
        if (
            self._config.max_questions_per_turn is None
            and self._config.max_tests_per_turn is None
            and not self._config.test_runner_enforce_single_turn_type
        ):
            return test_names, []  # No limits to apply

        questions: list[str] = []
        tests: list[str] = []
        other_requests: list[str] = []
        limit_messages: list[str] = []

        # Separate questions from tests from other requests
        for request in test_names:
            if not isinstance(request, str):
                print(
                    f"[BatchTestRunner] Warning: Non-string test name: {type(request)}: {request}"
                )
                continue
            if request.startswith("Question for the patient:"):
                questions.append(request)
            elif request.startswith("Test to order:"):
                tests.append(request)
            else:
                # If it doesn't have our prefixes, treat it as a test
                other_requests.append(request)

        # Check for turn type mixing if enforcement is enabled
        all_tests = tests + other_requests
        if (
            self._config.test_runner_enforce_single_turn_type
            and len(questions) > 0
            and len(all_tests) > 0
        ):
            limit_messages.append(
                "Turn type violation: You cannot mix questions and tests in the same turn. Please choose either questions OR tests for this turn."
            )
            # Return empty list to force reformulation
            return [], limit_messages

        # Apply question limits
        original_question_count = len(questions)
        if (
            self._config.max_questions_per_turn is not None
            and len(questions) > self._config.max_questions_per_turn
        ):
            questions = questions[: self._config.max_questions_per_turn]
            limit_messages.append(
                f"Question limit applied: {original_question_count} questions requested, limited to {self._config.max_questions_per_turn}"
            )

        # Apply test limits (combine tests and other requests for limit purposes)
        original_test_count = len(all_tests)
        if (
            self._config.max_tests_per_turn is not None
            and len(all_tests) > self._config.max_tests_per_turn
        ):
            all_tests = all_tests[: self._config.max_tests_per_turn]
            limit_messages.append(
                f"Test limit applied: {original_test_count} tests requested, limited to {self._config.max_tests_per_turn}"
            )

        # Combine allowed requests
        limited_requests = questions + all_tests

        return limited_requests, limit_messages

    @property
    def max_turns(self) -> int:
        """Maximum number of turns allowed for this test runner."""
        return self._max_turns

    @property
    def requested_tests(self) -> list[str]:
        """Get all tests that were requested through this runner."""
        return self._requested_tests.copy()

    @property
    def num_rounds(self) -> int:
        """Get the number of rounds/visits to clinician."""
        return self._num_rounds

    @property
    def num_test_rounds(self) -> int:
        """Get the number of rounds where diagnostic tests were ordered (excluding question-only rounds)."""
        return self._num_test_rounds

    @property
    def requested_tests_only(self) -> list[str]:
        """Get only the diagnostic tests that were requested (excluding questions for the patient)."""
        tests_only = [
            req for req in self._requested_tests if not req.startswith("Question for the patient:")
        ]
        return tests_only

    @property
    def turns_left(self) -> int:
        """Get the number of turns left."""
        return self._turns_left

    @property
    def completed_tests(self) -> list[str]:
        """Get a copy of completed tests list."""
        return self._completed_tests.copy()
