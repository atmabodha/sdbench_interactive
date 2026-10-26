"""Interactive CLI entry point for SDBench human-in-the-loop diagnostic sessions.

Bypasses the automated Gatekeeper and Judge so a human user can interact directly
with a diagnostic agent via standard terminal I/O.
Run with: uv run python -m sdbench.cli.interactive --agent baseline --model <model>
"""

import argparse
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable

import dotenv
from openai import OpenAI

import agents
from agents import baseline as baselineAgent
from config import AgentConfig, SDBenchConfig, get_sdbench_config
from gatekeeper import OrderTestResponse, TestRunnerResponse
import models
from models import ChatCompletionMessageParam, ModelCreateChatCompletion

# Prefix constants matching the standard SDBench Gatekeeper conventions.
QUESTION_PREFIX = "Question for the patient: "
TEST_PREFIX = "Test to order: "

# Default baseline system prompt used to initialize the diagnostic conversation.
DEFAULT_SYSTEM_PROMPT = """\
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


@dataclass
class ParsedAgentAction:
  """Structured representation of an agent's parsed turn action."""

  actionType: str
  requests: list[str] = field(default_factory=list)
  diagnosis: str | None = None
  recommendedTests: list[str] = field(default_factory=list)
  treatments: list[str] = field(default_factory=list)
  errorMessage: str | None = None


def cleanRequestPrefix(rawRequest: str) -> str:
  """Strip standard question or test prefixes from a request string for display."""
  if not rawRequest:
    return ""
  trimmedRequest = rawRequest.strip()
  if trimmedRequest.lower().startswith(QUESTION_PREFIX.lower()):
    return trimmedRequest[len(QUESTION_PREFIX) :].strip()
  if trimmedRequest.lower().startswith(TEST_PREFIX.lower()):
    return trimmedRequest[len(TEST_PREFIX) :].strip()
  return trimmedRequest


def normalizeStringList(rawValue: Any) -> list[str]:
  """Convert a string or iterable of strings into a clean list of non-empty strings."""
  if rawValue is None:
    return []
  if isinstance(rawValue, str):
    strippedItem = rawValue.strip()
    return [strippedItem] if strippedItem else []
  if isinstance(rawValue, list):
    return [str(item).strip() for item in rawValue if str(item).strip()]
  return []


def extractJsonCandidate(responseContent: str) -> str | None:
  """Extract a JSON block if the response is formatted as JSON or fenced JSON."""
  trimmedContent = responseContent.strip()
  if not trimmedContent:
    return None

  # Check for markdown fenced JSON block first.
  fencedMatch = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmedContent, re.DOTALL | re.IGNORECASE)
  if fencedMatch:
    return fencedMatch.group(1).strip()

  # Check if the raw content starts with '{' indicating a JSON object attempt.
  if trimmedContent.startswith("{"):
    return trimmedContent

  return None


def parseJsonAction(jsonText: str) -> ParsedAgentAction:
  """Parse a JSON-formatted agent response into a ParsedAgentAction."""
  try:
    payload = json.loads(jsonText)
  except json.JSONDecodeError as decodeError:
    return ParsedAgentAction(
      actionType="invalid",
      errorMessage=f"Malformed JSON output: {decodeError.msg}",
    )

  if not isinstance(payload, dict):
    return ParsedAgentAction(
      actionType="invalid",
      errorMessage="JSON action payload must be a JSON object.",
    )

  rawAction = str(payload.get("action") or payload.get("action_type") or "").strip().lower()
  if rawAction in ("ask_question", "question", "ask_questions"):
    questionItems = normalizeStringList(
      payload.get("questions") or payload.get("question") or payload.get("content")
    )
    if not questionItems:
      return ParsedAgentAction(
        actionType="invalid",
        errorMessage="Action 'ask_question' requires a non-empty 'question' or 'questions' field.",
      )
    formattedQuestions = [
      item if item.startswith(QUESTION_PREFIX) else f"{QUESTION_PREFIX}{item}"
      for item in questionItems
    ]
    return ParsedAgentAction(actionType="ask_question", requests=formattedQuestions)

  if rawAction in ("order_test", "test", "order_tests"):
    testItems = normalizeStringList(
      payload.get("tests") or payload.get("test") or payload.get("content")
    )
    if not testItems:
      return ParsedAgentAction(
        actionType="invalid",
        errorMessage="Action 'order_test' requires a non-empty 'test' or 'tests' field.",
      )
    formattedTests = [
      item if item.startswith(TEST_PREFIX) else f"{TEST_PREFIX}{item}"
      for item in testItems
    ]
    return ParsedAgentAction(actionType="order_test", requests=formattedTests)

  if rawAction in ("commit_diagnosis", "diagnose", "diagnosis"):
    diagnosisText = str(payload.get("diagnosis") or payload.get("content") or "").strip()
    if not diagnosisText:
      return ParsedAgentAction(
        actionType="invalid",
        errorMessage="Action 'commit_diagnosis' requires a non-empty 'diagnosis' field.",
      )
    recommendedTests = normalizeStringList(
      payload.get("recommended_tests") or payload.get("recommendedTests") or payload.get("tests")
    )
    treatments = normalizeStringList(
      payload.get("treatments") or payload.get("treatment") or payload.get("recommended_treatments")
    )
    return ParsedAgentAction(
      actionType="commit_diagnosis",
      diagnosis=diagnosisText,
      recommendedTests=recommendedTests,
      treatments=treatments,
    )

  return ParsedAgentAction(
    actionType="invalid",
    errorMessage=(
      f"Invalid action '{rawAction}'. Expected 'ask_question', 'order_test', or 'commit_diagnosis'."
    ),
  )


def extractTagValues(responseContent: str, tagName: str) -> list[str]:
  """Extract all non-empty inner strings for a given XML tag name."""
  pattern = rf"<{tagName}>(.*?)</{tagName}>"
  matches = re.findall(pattern, responseContent, re.IGNORECASE | re.DOTALL)
  return [match.strip() for match in matches if match.strip()]


def parseAgentAction(responseContent: str) -> ParsedAgentAction:
  """Parse the agent's raw response into an action (ask_question, order_test, or commit_diagnosis).

  Supports both the standard SDBench XML tags (<question>, <test>, <diagnosis>)
  and structured JSON action payloads.
  """
  if not responseContent or not responseContent.strip():
    return ParsedAgentAction(
      actionType="invalid",
      errorMessage="Empty response received from the agent.",
    )

  # First check if the agent produced a <diagnosis> tag (standard SDBench format).
  diagnosisText = baselineAgent.extract_diagnosis_from_response(responseContent)
  if diagnosisText:
    treatments = extractTagValues(responseContent, "treatment")
    recommendedTests = extractTagValues(responseContent, "recommended_test")
    return ParsedAgentAction(
      actionType="commit_diagnosis",
      diagnosis=diagnosisText,
      recommendedTests=recommendedTests,
      treatments=treatments,
    )

  # Next check for <question> and <test> tags using the baseline extractor.
  extractedItems = baselineAgent.extract_test_list_from_response(responseContent)
  if extractedItems:
    hasQuestions = any(item.startswith(QUESTION_PREFIX) for item in extractedItems)
    hasTests = any(not item.startswith(QUESTION_PREFIX) for item in extractedItems)
    actionType = "ask_question" if hasQuestions and not hasTests else "order_test"
    return ParsedAgentAction(actionType=actionType, requests=extractedItems)

  # Check if the response is a JSON or malformed JSON action attempt.
  jsonCandidate = extractJsonCandidate(responseContent)
  if jsonCandidate is not None:
    return parseJsonAction(jsonCandidate)

  # Check for unclosed XML tags to provide a clear correction message.
  if re.search(r"<(question|test|diagnosis)>", responseContent, re.IGNORECASE):
    return ParsedAgentAction(
      actionType="invalid",
      errorMessage="Malformed XML tags detected. Ensure all <question>, <test>, or <diagnosis> tags are properly closed.",
    )

  return ParsedAgentAction(
    actionType="invalid",
    errorMessage=(
      "No valid action found. Please order tests using <test> tags, ask questions using "
      "<question> tags, or commit a final diagnosis using <diagnosis> tags."
    ),
  )


def formatObservation(batchResponse: TestRunnerResponse) -> str:
  """Format the human user's answers/results into the observation format expected by the agent."""
  resultsText = "Results:\n"
  for resultItem in batchResponse.test_results:
    resultsText += f"- {resultItem['test_name']}: {resultItem['results']}\n"

  if batchResponse.limit_messages:
    resultsText += f"\nSYSTEM NOTICE: {'; '.join(batchResponse.limit_messages)}\n"

  return resultsText


class InteractiveTestRunner:
  """Human-in-the-loop TestRunner replacement for Gatekeeper BatchTestRunner.

  Prompts the human user via standard terminal I/O for each question or diagnostic
  test ordered by the agent while preserving SDBench turn budgets and logs.
  """

  def __init__(
    self,
    config: SDBenchConfig,
    collector: dict[str, Any],
    inputFn: Callable[[str], str] = input,
    outputFn: Callable[[str], None] = print,
  ):
    """Initialize the interactive test runner with turn limits and I/O callbacks."""
    self._config = config
    self._maxTurns = config.max_turns
    self._turnsLeft = self._maxTurns
    self._completedTests: list[str] = []
    self._requestedTests: list[str] = []
    self._numRounds = 0
    self._numTestRounds = 0
    self._collector = collector
    self._collector.setdefault("test_logs", [])
    self._inputFn = inputFn
    self._outputFn = outputFn

  def __call__(self, testNames: list[str]) -> TestRunnerResponse:
    """Prompt the human user for each requested question or test and return structured results."""
    if not isinstance(testNames, Iterable) or isinstance(testNames, (str, bytes)):
      raise ValueError("testNames must be an Iterable of request strings")

    if self._turnsLeft <= 0:
      raise ValueError("No turns left")

    limitedNames, limitMessages = self._parseAndLimitRequests(list(testNames))
    if not limitedNames and limitMessages:
      return TestRunnerResponse(
        test_names=[],
        test_results=[],
        limit_messages=limitMessages,
      )

    hasDiagnosticTests = any(
      not requestName.startswith(QUESTION_PREFIX) for requestName in limitedNames
    )

    self._turnsLeft -= 1
    self._requestedTests.extend(limitedNames)
    self._numRounds += 1
    if hasDiagnosticTests:
      self._numTestRounds += 1

    batchResults: list[OrderTestResponse] = []
    for requestName in limitedNames:
      userAnswer = self._promptHumanForRequest(requestName)
      responseItem = OrderTestResponse(
        {
          "test_name": requestName,
          "results": userAnswer,
          "is_synthetic": False,
        }
      )
      batchResults.append(responseItem)

      if requestName not in self._completedTests:
        self._completedTests.append(requestName)

      self._collector["test_logs"].append(
        {
          "test_name": requestName,
          "reason": "",
          "result": userAnswer,
          "is_synthetic": False,
          "timestamp": datetime.datetime.now().isoformat(),
          "batch_mode": True,
        }
      )

    return TestRunnerResponse(
      test_names=limitedNames,
      test_results=batchResults,
      limit_messages=limitMessages,
    )

  def _promptHumanForRequest(self, requestName: str) -> str:
    """Display a single question or test order to the terminal and read the user's input."""
    cleanedDisplay = cleanRequestPrefix(requestName)
    if requestName.startswith(QUESTION_PREFIX):
      self._outputFn(f"\n[Agent Question] {cleanedDisplay}")
      userReply = self._inputFn("Patient response: ").strip()
      return userReply if userReply else "Patient provided no response."

    self._outputFn(f"\n[Agent Test Order] {cleanedDisplay}")
    userReply = self._inputFn("Enter test result (or press Enter if unavailable): ").strip()
    return userReply if userReply else f"Results not available for {cleanedDisplay}"

  def _parseAndLimitRequests(self, testNames: list[str]) -> tuple[list[str], list[str]]:
    """Enforce single-turn-type, question-per-turn, and test-per-turn limits."""
    if (
      self._config.max_questions_per_turn is None
      and self._config.max_tests_per_turn is None
      and not self._config.test_runner_enforce_single_turn_type
    ):
      return testNames, []

    questionsList: list[str] = []
    testsList: list[str] = []
    otherList: list[str] = []
    limitMessages: list[str] = []

    for requestItem in testNames:
      if not isinstance(requestItem, str):
        continue
      if requestItem.startswith(QUESTION_PREFIX):
        questionsList.append(requestItem)
      elif requestItem.startswith(TEST_PREFIX):
        testsList.append(requestItem)
      else:
        otherList.append(f"{TEST_PREFIX}{requestItem.strip()}")

    combinedTests = testsList + otherList
    if (
      self._config.test_runner_enforce_single_turn_type
      and len(questionsList) > 0
      and len(combinedTests) > 0
    ):
      limitMessages.append(
        "Turn type violation: You cannot mix questions and tests in the same turn. "
        "Please choose either questions OR tests for this turn."
      )
      return [], limitMessages

    if (
      self._config.max_questions_per_turn is not None
      and len(questionsList) > self._config.max_questions_per_turn
    ):
      originalCount = len(questionsList)
      questionsList = questionsList[: self._config.max_questions_per_turn]
      limitMessages.append(
        f"Question limit applied: {originalCount} questions requested, "
        f"limited to {self._config.max_questions_per_turn}"
      )

    if (
      self._config.max_tests_per_turn is not None
      and len(combinedTests) > self._config.max_tests_per_turn
    ):
      originalCount = len(combinedTests)
      combinedTests = combinedTests[: self._config.max_tests_per_turn]
      limitMessages.append(
        f"Test limit applied: {originalCount} tests requested, "
        f"limited to {self._config.max_tests_per_turn}"
      )

    return questionsList + combinedTests, limitMessages

  @property
  def maxTurns(self) -> int:
    """Maximum number of turns allowed."""
    return self._maxTurns

  @property
  def max_turns(self) -> int:
    """Protocol-compatible alias for maxTurns."""
    return self._maxTurns

  @property
  def turnsLeft(self) -> int:
    """Number of diagnostic turns remaining."""
    return self._turnsLeft

  @property
  def turns_left(self) -> int:
    """Protocol-compatible alias for turnsLeft."""
    return self._turnsLeft

  @property
  def requestedTests(self) -> list[str]:
    """Copy of all questions and tests requested so far."""
    return self._requestedTests.copy()

  @property
  def requested_tests(self) -> list[str]:
    """Protocol-compatible alias for requestedTests."""
    return self._requestedTests.copy()

  @property
  def requestedTestsOnly(self) -> list[str]:
    """Copy of only the diagnostic tests requested (excluding patient questions)."""
    return [req for req in self._requestedTests if not req.startswith(QUESTION_PREFIX)]

  @property
  def requested_tests_only(self) -> list[str]:
    """Protocol-compatible alias for requestedTestsOnly."""
    return self.requestedTestsOnly

  @property
  def numRounds(self) -> int:
    """Total number of interactive information-gathering rounds completed."""
    return self._numRounds

  @property
  def num_rounds(self) -> int:
    """Protocol-compatible alias for numRounds."""
    return self._numRounds

  @property
  def numTestRounds(self) -> int:
    """Number of rounds in which at least one diagnostic test was ordered."""
    return self._numTestRounds

  @property
  def num_test_rounds(self) -> int:
    """Protocol-compatible alias for numTestRounds."""
    return self._numTestRounds

  @property
  def completedTests(self) -> list[str]:
    """Copy of all completed test and question requests."""
    return self._completedTests.copy()

  @property
  def completed_tests(self) -> list[str]:
    """Protocol-compatible alias for completedTests."""
    return self._completedTests.copy()


class InteractiveAgent:
  """Drives an interactive diagnostic session with a human user via standard I/O."""

  def __init__(
    self,
    agentName: str,
    agentLlm: ModelCreateChatCompletion,
    agentConfig: AgentConfig,
    testRunner: InteractiveTestRunner,
    collector: dict[str, Any],
    outputFn: Callable[[str], None] = print,
  ):
    """Initialize the interactive agent state and message history."""
    self.agentName = agentName
    self.agentLlm = agentLlm
    self.agentConfig = agentConfig
    self.testRunner = testRunner
    self.collector = collector
    self.outputFn = outputFn
    self.messages: list[ChatCompletionMessageParam] = []

  def initializeState(self, caseSummary: str) -> None:
    """Initialize the agent's message history with the system prompt and initial presentation."""
    effectiveSummary = (
      caseSummary.strip()
      if caseSummary and caseSummary.strip()
      else "A new patient case is available. You need to ask questions and order tests to reach a diagnosis."
    )
    self.messages = [
      {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
      {
        "role": "user",
        "content": (
          f"Patient presentation:\n\n{effectiveSummary}\n\n"
          "What questions would you like to ask?\nOr what tests would you like to order?"
        ),
      },
    ]
    self.collector["agent_messages"] = self.messages

  def printFinalDiagnosis(self, action: ParsedAgentAction) -> None:
    """Display the final committed diagnosis and any recommended tests or treatments."""
    self.outputFn("\n" + "=" * 60)
    self.outputFn("FINAL DIAGNOSIS")
    self.outputFn("=" * 60)
    self.outputFn(f"Diagnosis: {action.diagnosis}")
    if action.recommendedTests:
      self.outputFn(f"Recommended Tests: {', '.join(action.recommendedTests)}")
    if action.treatments:
      self.outputFn(f"Recommended Treatments: {', '.join(action.treatments)}")
    self.outputFn("=" * 60)

  def run(self, caseSummary: str) -> str:
    """Execute the main interactive while loop until diagnosis commitment or turn exhaustion."""
    # If a custom non-baseline agent module is selected, verify it exists.
    agentModule = getattr(agents, self.agentName, None)
    if agentModule is None:
      availableAgents = [name for name in dir(agents) if not name.startswith("_")]
      raise ValueError(f"Unknown agent: {self.agentName}. Available: {availableAgents}")

    # Initialize the conversation state with the user's initial case summary.
    self.initializeState(caseSummary)

    while self.testRunner.turnsLeft > 0:
      response = self.agentLlm(messages=self.messages)
      assistantContent = response.choices[0].message.content or ""
      self.messages.append({"role": "assistant", "content": assistantContent})

      parsedAction = parseAgentAction(assistantContent)

      if parsedAction.actionType == "commit_diagnosis" and parsedAction.diagnosis:
        self.collector["agent_messages"] = self.messages
        self.collector["recommended_tests"] = parsedAction.recommendedTests
        self.collector["treatments"] = parsedAction.treatments
        self.printFinalDiagnosis(parsedAction)
        return parsedAction.diagnosis

      if parsedAction.actionType in ("ask_question", "order_test") and parsedAction.requests:
        try:
          batchResponse = self.testRunner(parsedAction.requests)
        except (EOFError, KeyboardInterrupt):
          self.outputFn("\nSession ended by user.")
          self.collector["agent_messages"] = self.messages
          return "Session interrupted by user"

        observationText = formatObservation(batchResponse)
        self.messages.append({"role": "user", "content": observationText})
        continue

      # Handle malformed JSON or invalid action by prompting the agent to self-correct.
      correctionPrompt = (
        "Please order tests using <test> tags, questions using <question> tags, "
        "or provide diagnosis using <diagnosis> tags."
      )
      if parsedAction.errorMessage:
        correctionPrompt = f"{parsedAction.errorMessage} {correctionPrompt}"
      self.messages.append({"role": "user", "content": correctionPrompt})

    # Out of turns: force the agent to commit a final diagnosis.
    self.messages.append(
      {
        "role": "user",
        "content": "No more tests available. Please provide your diagnosis using <diagnosis> tags.",
      }
    )
    finalResponse = self.agentLlm(messages=self.messages)
    finalContent = finalResponse.choices[0].message.content or ""
    self.messages.append({"role": "assistant", "content": finalContent})

    finalAction = parseAgentAction(finalContent)
    self.collector["agent_messages"] = self.messages
    if finalAction.actionType == "commit_diagnosis" and finalAction.diagnosis:
      self.printFinalDiagnosis(finalAction)
      return finalAction.diagnosis

    fallbackDiagnosis = "Unable to provide diagnosis"
    self.printFinalDiagnosis(
      ParsedAgentAction(actionType="commit_diagnosis", diagnosis=fallbackDiagnosis)
    )
    return fallbackDiagnosis


def resolveDefaultModel(fallbackModel: str = "o4-mini") -> str:
  """Resolve the default model name from environment variables loaded from .env."""
  envKeys = ("SDBENCH_MODEL", "AGENT_MODEL", "OPENROUTER_MODEL", "OPENAI_MODEL", "MODEL")
  for envKey in envKeys:
    envValue = os.getenv(envKey)
    if envValue and envValue.strip():
      return envValue.strip()
  return fallbackModel


def resolveDefaultAgent(fallbackAgent: str = "baseline") -> str:
  """Resolve the default diagnostic agent name from environment variables or default to baseline."""
  envKeys = ("SDBENCH_AGENT", "AGENT")
  for envKey in envKeys:
    envValue = os.getenv(envKey)
    if envValue and envValue.strip():
      return envValue.strip()
  return fallbackAgent


def buildArgumentParser() -> argparse.ArgumentParser:
  """Construct the CLI argument parser mirroring existing SDBench CLIs."""
  defaultAgent = resolveDefaultAgent()
  defaultModel = resolveDefaultModel()

  parser = argparse.ArgumentParser(
    description="Run SDBench in interactive human-in-the-loop diagnostic mode"
  )
  parser.add_argument(
    "--agent",
    type=str,
    default=defaultAgent,
    help=f"Diagnostic agent to use (default: {defaultAgent})",
  )
  parser.add_argument(
    "--model",
    type=str,
    default=defaultModel,
    help=f"Model for the agent to use, read from .env SDBENCH_MODEL/MODEL if set (default: {defaultModel})",
  )
  parser.add_argument(
    "--model-args",
    type=json.loads,
    default=None,
    help="Model arguments as JSON dict",
  )
  parser.add_argument(
    "--api-key",
    type=str,
    default=None,
    help="API key override for OpenRouter or local model endpoint",
  )
  parser.add_argument(
    "--base-url",
    type=str,
    default=None,
    help="Base URL override for local model endpoint",
  )
  parser.add_argument(
    "--dataset",
    type=str,
    default="synthetic",
    help="Dataset name for base SDBench config (default: synthetic)",
  )
  parser.add_argument(
    "--sdbench-config",
    type=str,
    default="default",
    help="SDBench config name (default: default)",
  )
  parser.add_argument(
    "--max-turns",
    type=int,
    default=None,
    help="Override maximum number of diagnostic turns",
  )
  parser.add_argument(
    "--case-summary",
    type=str,
    default=None,
    help="Optional initial case summary (if omitted, prompted via stdin)",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug flag on configuration",
  )
  parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Write session result JSON to file",
  )
  return parser


def resolveModelArgs(
  modelName: str,
  rawModelArgs: dict[str, Any] | None,
  configModelArgs: dict[str, Any],
  apiKey: str | None = None,
  baseUrl: str | None = None,
) -> dict[str, Any]:
  """Merge CLI model arguments, config model arguments, and optional API key/base URL overrides."""
  mergedArgs: dict[str, Any] = {**(rawModelArgs or {}), **configModelArgs}

  if apiKey:
    if modelName.startswith(models.LOCAL_MODEL_PREFIX):
      mergedArgs["local_model_api_key"] = apiKey
    elif apiKey.startswith("sk-or-"):
      os.environ["OPENROUTER_API_KEY"] = apiKey
    elif not os.getenv("OPENROUTER_API_KEY"):
      os.environ["OPENAI_API_KEY"] = apiKey
    else:
      os.environ["OPENROUTER_API_KEY"] = apiKey

  if baseUrl:
    if modelName.startswith(models.LOCAL_MODEL_PREFIX):
      mergedArgs["local_model_base_url"] = baseUrl
    else:
      os.environ["SDBENCH_LOCAL_BASE_URL"] = baseUrl

  return mergedArgs


def validateModelCredentials(modelName: str, modelArgs: dict[str, Any] | None = None) -> None:
  """Validate that required API credentials or endpoints are configured for the chosen model."""
  effectiveArgs = modelArgs or {}

  # Local models require a base URL via CLI/model args or SDBENCH_LOCAL_BASE_URL.
  if modelName.startswith(models.LOCAL_MODEL_PREFIX):
    localBaseUrl = effectiveArgs.get("local_model_base_url") or os.getenv("SDBENCH_LOCAL_BASE_URL")
    if not localBaseUrl:
      raise ValueError(
        f"Local model '{modelName}' requires '--base-url <url>' or SDBENCH_LOCAL_BASE_URL in .env."
      )
    return

  # Forced Azure OpenAI requires both AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.
  forceAzure = os.getenv("SDBENCH_FORCE_AZURE", "0").lower() in ("1", "true")
  if forceAzure:
    if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
      raise ValueError(
        "SDBENCH_FORCE_AZURE is enabled, but AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT is not set."
      )
    return

  # Standard models accept either OPENROUTER_API_KEY or OPENAI_API_KEY.
  apiKeyVar = str(effectiveArgs.get("openrouter_api_key_var", "OPENROUTER_API_KEY"))
  hasOpenRouterKey = bool(os.getenv(apiKeyVar) or os.getenv("OPENROUTER_API_KEY"))
  hasOpenAiKey = bool(os.getenv("OPENAI_API_KEY"))
  if not hasOpenRouterKey and not hasOpenAiKey:
    raise ValueError(
      "No API key found. Create a '.env' file in this root folder with "
      "OPENROUTER_API_KEY=<your-key> or OPENAI_API_KEY=<your-key>, or pass --api-key <your-key>."
    )


def loadEnvironmentFiles() -> None:
  """Load environment variables from CWD .env and package directory .env."""
  packageEnvPath = Path(__file__).resolve().parent.parent / ".env"
  if packageEnvPath.is_file():
    dotenv.load_dotenv(dotenv_path=packageEnvPath, override=False)
  dotenv.load_dotenv(override=False)


def buildAgentLlm(
  modelName: str,
  debug: bool = False,
  **modelArgs: Any,
) -> ModelCreateChatCompletion:
  """Create the chat completion callable, supporting OpenRouter, local, Azure, and direct OpenAI keys."""
  hasOpenRouterKey = bool(os.getenv("OPENROUTER_API_KEY"))
  forceAzure = os.getenv("SDBENCH_FORCE_AZURE", "0").lower() in ("1", "true")
  openaiApiKey = os.getenv("OPENAI_API_KEY")

  if (
    not modelName.startswith(models.LOCAL_MODEL_PREFIX)
    and not hasOpenRouterKey
    and not forceAzure
    and openaiApiKey
  ):
    directClient = OpenAI(api_key=openaiApiKey)
    resolvedModel = modelName.split("/", 1)[1] if modelName.startswith("openai/") else modelName

    def callOpenAiDirect(
      messages: Iterable[ChatCompletionMessageParam], **callKwargs: Any
    ) -> Any:
      combinedKwargs = {**modelArgs, **callKwargs}
      return directClient.chat.completions.create(
        model=resolvedModel,
        messages=messages,
        stream=False,
        **combinedKwargs,
      )

    return callOpenAiDirect

  return models.model_create_chat_completion_fn(modelName, debug=debug, **modelArgs)


def runInteractiveSession(
  config: SDBenchConfig,
  caseSummary: str,
  agentName: str = "baseline",
  model: str | ModelCreateChatCompletion = "o4-mini",
  modelArgs: dict[str, Any] | None = None,
  apiKey: str | None = None,
  baseUrl: str | None = None,
  inputFn: Callable[[str], str] = input,
  outputFn: Callable[[str], None] = print,
) -> dict[str, Any]:
  """Run a complete human-in-the-loop interactive diagnostic session and return session results."""
  agentModule = getattr(agents, agentName, None)
  if agentModule is None:
    availableAgents = [name for name in dir(agents) if not name.startswith("_")]
    raise ValueError(f"Unknown agent: {agentName}. Available: {availableAgents}")

  if callable(model):
    agentLlm = model
  else:
    allModelArgs = resolveModelArgs(
      modelName=model,
      rawModelArgs=modelArgs,
      configModelArgs=config.agent_model_args,
      apiKey=apiKey,
      baseUrl=baseUrl,
    )
    agentLlm = buildAgentLlm(
      model, debug=config.debug, **allModelArgs
    )

  sessionResults: dict[str, Any] = {}
  testRunner = InteractiveTestRunner(
    config=config,
    collector=sessionResults,
    inputFn=inputFn,
    outputFn=outputFn,
  )
  agentConfig = AgentConfig.from_sdbench_config(config)

  interactiveAgent = InteractiveAgent(
    agentName=agentName,
    agentLlm=agentLlm,
    agentConfig=agentConfig,
    testRunner=testRunner,
    collector=sessionResults,
    outputFn=outputFn,
  )
  finalDiagnosis = interactiveAgent.run(caseSummary)

  sessionResults["model_diagnosis"] = finalDiagnosis
  sessionResults["requested_tests"] = testRunner.requestedTests
  sessionResults["requested_tests_only"] = testRunner.requestedTestsOnly
  sessionResults["num_rounds"] = testRunner.numRounds
  sessionResults["num_test_rounds"] = testRunner.numTestRounds

  return sessionResults


# Provide snake_case aliases for external callers expecting standard Python naming.
clean_request_prefix = cleanRequestPrefix
parse_agent_action = parseAgentAction
format_observation = formatObservation
resolve_default_model = resolveDefaultModel
resolve_default_agent = resolveDefaultAgent
build_argument_parser = buildArgumentParser
resolve_model_args = resolveModelArgs
run_interactive_session = runInteractiveSession


def main(argv: list[str] | None = None) -> None:
  """Main CLI entry point for interactive SDBench mode."""
  loadEnvironmentFiles()

  parser = buildArgumentParser()
  args = parser.parse_args(argv)

  config = get_sdbench_config(args.dataset, config_name=args.sdbench_config)
  config.debug = args.debug
  if args.max_turns is not None:
    config.max_turns = args.max_turns

  if getattr(agents, args.agent, None) is None:
    availableAgents = [name for name in dir(agents) if not name.startswith("_")]
    print(f"Error: Unknown agent '{args.agent}'. Available: {availableAgents}")
    sys.exit(1)

  # Resolve model args and validate credentials before prompting the user for symptoms.
  mergedModelArgs = resolveModelArgs(
    modelName=args.model,
    rawModelArgs=args.model_args,
    configModelArgs=config.agent_model_args,
    apiKey=args.api_key,
    baseUrl=args.base_url,
  )
  try:
    validateModelCredentials(args.model, mergedModelArgs)
  except ValueError as credentialError:
    print(f"Configuration error: {credentialError}", file=sys.stderr)
    sys.exit(1)

  print(f"Starting SDBench Interactive Mode (agent={args.agent}, model={args.model})")

  try:
    caseSummary = (
      args.case_summary
      if args.case_summary is not None
      else input("Enter your initial symptoms: ").strip()
    )
  except (EOFError, KeyboardInterrupt):
    print("\nInteractive session cancelled.")
    return

  result = runInteractiveSession(
    config=config,
    caseSummary=caseSummary,
    agentName=args.agent,
    model=args.model,
    modelArgs=args.model_args,
    apiKey=args.api_key,
    baseUrl=args.base_url,
  )

  print(f"\nTotal rounds completed: {result['num_rounds']}")
  print(f"Diagnostic test rounds: {result['num_test_rounds']}")

  if args.output:
    with open(args.output, "w") as outputFile:
      json.dump(result, outputFile, indent=2)
    print(f"\nSession result written to {args.output}")


if __name__ == "__main__":
  main()
