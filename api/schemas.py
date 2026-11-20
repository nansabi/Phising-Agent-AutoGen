"""Pydantic request and response models for the local web API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed"]
JobStep = Literal[
    "Extract",
    "Check",
    "Classify",
    "Debate",
    "Policy",
    "Chart",
    "Awaiting approval",
]


class HealthResponse(BaseModel):
    """Expose only safe service readiness booleans."""

    status: Literal["ok"]
    groq_key_configured: bool
    docker_available: bool


class AnalyzeResponse(BaseModel):
    """Identify a newly queued analysis."""

    job_id: str
    case_id: str


class JobResponse(BaseModel):
    """Report one background task's progress without internal tracebacks."""

    job_id: str
    case_id: str
    status: JobStatus
    current_step: JobStep | None = None
    error_text: str | None = None


class CaseSummary(BaseModel):
    """One case row for the web application's case list."""

    id: str
    status: str
    verdict: str
    confidence: str | None = None
    date: str


class LinkFinding(BaseModel):
    """Keep the original URL a plain string and provide a safe display form."""

    url: str
    display_url: str
    risk_score: int | None = None
    reasons: list[str] = Field(default_factory=list)


class DebateMessage(BaseModel):
    """One normalized turn in a debate attempt."""

    attempt: int
    round: int
    speaker: str
    message: str


class CaseDetail(BaseModel):
    """Evidence and workflow state needed by the case detail page."""

    id: str
    source_type: str
    email_text: str
    current_verdict: str
    case_plan: list[str]
    extracted_facts: dict[str, Any]
    sender_domain_findings: str
    link_findings: list[LinkFinding]
    auth_results: list[str]
    debate_history: list[DebateMessage]
    policy_matches: list[dict[str, Any]]
    similar_past_cases: list[dict[str, Any]]
    analyst_notes: list[str]
    status: str
    cost: dict[str, Any]
    chart_url: str | None = None
    report_url: str | None = None
    created_at: str = ""
    updated_at: str = ""


class DecisionResponse(BaseModel):
    """Return the updated approval state and current verdict."""

    case_id: str
    status: str
    current_verdict: str


class NoteRequest(BaseModel):
    """Validate an analyst note before saving it."""

    note: str = Field(min_length=1, max_length=2000)


class RejectRequest(BaseModel):
    """Validate the concrete human challenge for a debate rerun."""

    reason: str = Field(min_length=1, max_length=1000)


class CostResponse(BaseModel):
    """Summarize saved model costs in USD."""

    total_cost: float
    cost_today: float


class LogsResponse(BaseModel):
    """Return safe LLM call metadata without prompt or response contents."""

    calls: list[dict[str, Any]]
