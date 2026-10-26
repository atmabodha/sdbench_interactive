"""Unit tests for the SDBench interactive CLI utility functions and runner."""

from types import SimpleNamespace
from typing import Any, Iterable
import os
import unittest
from unittest.mock import patch

from sdbench.cli.interactive import (
  InteractiveAgent,
  InteractiveTestRunner,
  ParsedAgentAction,
  QUESTION_PREFIX,
  TEST_PREFIX,
  buildAgentLlm,
  buildArgumentParser,
  cleanRequestPrefix,
  extractJsonCandidate,
  extractTagValues,
  formatObservation,
  loadEnvironmentFiles,
  normalizeStringList,
  parseAgentAction,
  parseJsonAction,
  resolveDefaultAgent,
  resolveDefaultModel,
  resolveModelArgs,
  runInteractiveSession,
)
from sdbench.config import SDBenchConfig
from sdbench.gatekeeper import OrderTestResponse, TestRunnerResponse
from sdbench.models import ChatCompletionMessageParam


class MockChatCompletionFactory:
  """Helper factory to build mock OpenAI ChatCompletion responses for testing."""

  @staticmethod
  def createResponse(contentText: str) -> Any:
    """Create a minimal mock ChatCompletion object with a single message choice."""
    messageObj = SimpleNamespace(role="assistant", content=contentText)
    choiceObj = SimpleNamespace(message=messageObj)
    return SimpleNamespace(choices=[choiceObj])


class TestInteractiveUtilities(unittest.TestCase):
  """Unit tests for parsing, formatting, and interactive session utilities."""

  def _makeTestConfig(self, maxTurns: int = 5) -> SDBenchConfig:
    """Create a minimal SDBenchConfig instance for unit testing."""
    return SDBenchConfig(
      max_turns=maxTurns,
      gatekeeper_prompt_version="v6_fixed",
      gatekeeper_model="gpt-5",
      judge_prompt_version="v4_reasoning_with_tests",
      judge_model="gpt-5",
    )

  def testCleanRequestPrefix(self) -> None:
    """Verify question and test prefixes are stripped cleanly."""
    self.assertEqual(cleanRequestPrefix(""), "")
    self.assertEqual(
      cleanRequestPrefix("Question for the patient: Do you have a fever?"),
      "Do you have a fever?",
    )
    self.assertEqual(
      cleanRequestPrefix("Test to order: Complete Blood Count"),
      "Complete Blood Count",
    )
    self.assertEqual(cleanRequestPrefix("Chest X-ray"), "Chest X-ray")

  def testNormalizeStringList(self) -> None:
    """Verify string and list values are normalized into clean string lists."""
    self.assertEqual(normalizeStringList(None), [])
    self.assertEqual(normalizeStringList("   "), [])
    self.assertEqual(normalizeStringList("CBC"), ["CBC"])
    self.assertEqual(normalizeStringList([" CBC ", "", "BMP "]), ["CBC", "BMP"])

  def testExtractJsonCandidate(self) -> None:
    """Verify raw JSON and fenced markdown JSON blocks are extracted properly."""
    self.assertIsNone(extractJsonCandidate(""))
    self.assertIsNone(extractJsonCandidate(" Plain text without JSON "))
    rawJson = '{"action": "ask_question", "question": "Any cough?"}'
    self.assertEqual(extractJsonCandidate(rawJson), rawJson)
    fencedJson = f"```json\n{rawJson}\n```"
    self.assertEqual(extractJsonCandidate(fencedJson), rawJson)

  def testExtractTagValues(self) -> None:
    """Verify XML tag values are extracted in order."""
    sampleText = "<treatment>IV fluids</treatment>\n<treatment>Antibiotics</treatment>"
    self.assertEqual(extractTagValues(sampleText, "treatment"), ["IV fluids", "Antibiotics"])

  def testParseJsonActionValidAndInvalid(self) -> None:
    """Verify JSON actions for ask_question, order_test, commit_diagnosis, and malformed input."""
    questionAction = parseJsonAction(
      '{"action": "ask_question", "questions": ["How long have you had pain?"]}'
    )
    self.assertEqual(questionAction.actionType, "ask_question")
    self.assertEqual(
      questionAction.requests,
      [f"{QUESTION_PREFIX}How long have you had pain?"],
    )

    testAction = parseJsonAction('{"action": "order_test", "test": "Serum lactate"}')
    self.assertEqual(testAction.actionType, "order_test")
    self.assertEqual(testAction.requests, [f"{TEST_PREFIX}Serum lactate"])

    diagnosisAction = parseJsonAction(
      '{"action": "commit_diagnosis", "diagnosis": "Acute appendicitis", '
      '"treatments": ["Appendectomy"], "recommended_tests": ["Abdominal CT"]}'
    )
    self.assertEqual(diagnosisAction.actionType, "commit_diagnosis")
    self.assertEqual(diagnosisAction.diagnosis, "Acute appendicitis")
    self.assertEqual(diagnosisAction.treatments, ["Appendectomy"])
    self.assertEqual(diagnosisAction.recommendedTests, ["Abdominal CT"])

    malformedAction = parseJsonAction('{"action": "ask_question", broken}')
    self.assertEqual(malformedAction.actionType, "invalid")
    self.assertIn("Malformed JSON", malformedAction.errorMessage or "")

  def testParseAgentActionXmlAndFallback(self) -> None:
    """Verify XML tag extraction for questions, tests, diagnosis, and invalid responses."""
    xmlQuestion = parseAgentAction("<question>Any recent travel?</question>")
    self.assertEqual(xmlQuestion.actionType, "ask_question")
    self.assertEqual(xmlQuestion.requests, [f"{QUESTION_PREFIX}Any recent travel?"])

    xmlTest = parseAgentAction("<test>Blood cultures</test>")
    self.assertEqual(xmlTest.actionType, "order_test")
    self.assertEqual(xmlTest.requests, [f"{TEST_PREFIX}Blood cultures"])

    xmlDiagnosis = parseAgentAction(
      "<diagnosis>Community-acquired pneumonia</diagnosis>"
      "<treatment>Azithromycin</treatment>"
    )
    self.assertEqual(xmlDiagnosis.actionType, "commit_diagnosis")
    self.assertEqual(xmlDiagnosis.diagnosis, "Community-acquired pneumonia")
    self.assertEqual(xmlDiagnosis.treatments, ["Azithromycin"])

    unclosedTag = parseAgentAction("<question>Unclosed question")
    self.assertEqual(unclosedTag.actionType, "invalid")
    self.assertIn("Malformed XML", unclosedTag.errorMessage or "")

  def testFormatObservation(self) -> None:
    """Verify Gatekeeper-style observation formatting from TestRunnerResponse."""
    testResponse = TestRunnerResponse(
      test_names=[f"{TEST_PREFIX}CBC"],
      test_results=[
        OrderTestResponse(
          {
            "test_name": f"{TEST_PREFIX}CBC",
            "results": "WBC 14.2, Hb 13.5",
            "is_synthetic": False,
          }
        )
      ],
      limit_messages=["Test limit applied: 2 tests requested, limited to 1"],
    )
    formattedText = formatObservation(testResponse)
    self.assertIn("Results:\n- Test to order: CBC: WBC 14.2, Hb 13.5", formattedText)
    self.assertIn("SYSTEM NOTICE: Test limit applied", formattedText)

  def testInteractiveTestRunnerAndSessionLoop(self) -> None:
    """Verify multi-turn interactive loop handles question, error recovery, test, and diagnosis."""
    config = self._makeTestConfig(maxTurns=5)
    userInputs = iter(["Yes, high fever for 3 days.", "WBC elevated at 15,000/uL."])
    capturedOutputs: list[str] = []

    turnResponses = iter(
      [
        MockChatCompletionFactory.createResponse("<question>Do you have a fever?</question>"),
        MockChatCompletionFactory.createResponse("{malformed json response"),
        MockChatCompletionFactory.createResponse("<test>CBC</test>"),
        MockChatCompletionFactory.createResponse("<diagnosis>Acute bacterial infection</diagnosis>"),
      ]
    )

    def fakeLlm(messages: Iterable[ChatCompletionMessageParam], **kwargs: Any) -> Any:
      return next(turnResponses)

    sessionResult = runInteractiveSession(
      config=config,
      caseSummary="Patient presents with chills and fatigue.",
      agentName="baseline",
      model=fakeLlm,
      inputFn=lambda promptText: next(userInputs),
      outputFn=capturedOutputs.append,
    )

    self.assertEqual(sessionResult["model_diagnosis"], "Acute bacterial infection")
    self.assertEqual(sessionResult["num_rounds"], 2)
    self.assertEqual(sessionResult["num_test_rounds"], 1)
    self.assertEqual(len(sessionResult["test_logs"]), 2)
    self.assertTrue(any("FINAL DIAGNOSIS" in line for line in capturedOutputs))

  def testBuildArgumentParserAndResolveModelArgs(self) -> None:
    """Verify CLI parser defaults from .env and model argument resolution."""
    with patch.dict(os.environ, {"SDBENCH_MODEL": "claude-4-sonnet"}, clear=False):
      self.assertEqual(resolveDefaultModel(), "claude-4-sonnet")
      self.assertEqual(resolveDefaultAgent(), "baseline")
      defaultParser = buildArgumentParser()
      zeroFlagArgs = defaultParser.parse_args([])
      self.assertEqual(zeroFlagArgs.agent, "baseline")
      self.assertEqual(zeroFlagArgs.model, "claude-4-sonnet")

    parser = buildArgumentParser()
    parsedArgs = parser.parse_args(["--agent", "baseline", "--model", "local-llama"])
    self.assertEqual(parsedArgs.agent, "baseline")
    self.assertEqual(parsedArgs.model, "local-llama")

    resolvedLocal = resolveModelArgs(
      modelName="local-llama",
      rawModelArgs={"temperature": 0.2},
      configModelArgs={},
      apiKey="test-key",
      baseUrl="http://localhost:8000/v1",
    )
    self.assertEqual(resolvedLocal["temperature"], 0.2)
    self.assertEqual(resolvedLocal["local_model_api_key"], "test-key")
    self.assertEqual(resolvedLocal["local_model_base_url"], "http://localhost:8000/v1")

    loadEnvironmentFiles()
    with patch.dict(
      os.environ,
      {"OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": ""},
      clear=False,
    ):
      llmFn = buildAgentLlm("o4-mini")
      self.assertTrue(callable(llmFn))


if __name__ == "__main__":
  unittest.main()
