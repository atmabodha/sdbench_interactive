from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from sdbench.costing import cpt_matcher  # pyright: ignore[reportAttributeAccessIssue]
    from sdbench.models import ChatCompletionMessageParam


class CostMethodEstimate(TypedDict, total=False):
    """Either 'cost' or 'error' will be present."""

    cost: float
    error: str
    time: float  # Time taken in seconds

    # llm_estimator or llm_batch_estimator values
    messages: list["ChatCompletionMessageParam"]
    batch_test_name: str  # Only for llm_batch_estimator

    # cpt_matcher values
    cpt_codes: list[str]
    cpt_selection: "cpt_matcher.CPTSelectionResult"


class CostEstimate(TypedDict):
    """Full cost estimate for a single test."""

    test_name: str
    cost: float | None  # Set, unless method is "failure"
    method: str  # One of "cpt_lookup", "llm_fallback", "llm_only", "batch_llm_only", "failure"
    cpt_estimate: CostMethodEstimate | None
    llm: CostMethodEstimate | None
    explanation: str


class CostsEstimate(TypedDict):
    """Aggregate cost estimates for all tests in a run."""

    config: dict[str, Any]
    raw_analysis: str
    total_unique_tests: int
    total_requests: int
    num_rounds: int
    cost_details: list[CostEstimate]
    clinician_cost_per_visit: int
    method: str  # One of "unified_cpt_llm", "llm_only", "batch_llm_only", "no_tests"
    totals: "CostsEstimateTotals"


class CostsEstimateTotals(TypedDict):
    """Totals section of a CostsEstimate."""

    tests_only: float
    clinician_visits: float
    total_with_visits: float
    average_per_test: float
