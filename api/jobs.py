"""Thread-safe in-memory status tracking for background API tasks."""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Callable, Literal
from uuid import uuid4

from src.debate import normalize_text
from src.intake import mask_pii_text

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

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="phishing-desk")
_lock = RLock()
_jobs: dict[str, "JobRecord"] = {}
_SECRET_PATTERN = re.compile(r"\b(?:gsk|sk)[-_][A-Za-z0-9_-]{12,}\b")
_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(?:api[_ -]?key|authorization)\s*[:=]\s*\S+"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+\S+")
_URL_PATTERN = re.compile(r"(?i)\bhttps?://[^\s<>\"']+")


@dataclass
class JobRecord:
    """Store the minimal progress state returned by the polling endpoint."""

    job_id: str
    case_id: str
    status: JobStatus = "queued"
    current_step: JobStep | None = "Extract"
    error_text: str | None = None


def submit_job(
    case_id: str,
    task: Callable[[str, str], None],
) -> JobRecord:
    """Queue a task and return its initial state immediately."""
    record = JobRecord(job_id=uuid4().hex, case_id=case_id)
    with _lock:
        _jobs[record.job_id] = record
    _executor.submit(_run_task, record.job_id, task)
    return get_job(record.job_id)


def get_job(job_id: str) -> JobRecord | None:
    """Return a copy of one job record for a race-free API response."""
    with _lock:
        record = _jobs.get(job_id)
        return JobRecord(**asdict(record)) if record else None


def set_job_step(job_id: str, step: JobStep) -> None:
    """Update a background task's current phase."""
    with _lock:
        record = _jobs.get(job_id)
        if record is not None:
            record.current_step = step


def active_job_for_case(case_id: str) -> JobRecord | None:
    """Find an unfinished job that already owns this case."""
    with _lock:
        for record in _jobs.values():
            if record.case_id == case_id and record.status in {"queued", "running"}:
                return JobRecord(**asdict(record))
    return None


def _run_task(job_id: str, task: Callable[[str, str], None]) -> None:
    """Run one task while translating errors to concise safe text."""
    with _lock:
        record = _jobs[job_id]
        record.status = "running"
    try:
        task(job_id, record.case_id)
    except Exception as error:
        message = normalize_text(str(error)).strip() or "The background task failed."
        message = _SECRET_PATTERN.sub("[REDACTED]", message)
        message = _KEY_VALUE_PATTERN.sub("credential=[REDACTED]", message)
        message = _BEARER_PATTERN.sub("Bearer [REDACTED]", message)
        message = mask_pii_text(message)
        message = _URL_PATTERN.sub(
            lambda match: match.group(0)
            .replace("https://", "hxxps://")
            .replace("http://", "hxxp://")
            .replace(".", "[.]"),
            message,
        )
        with _lock:
            record.status = "failed"
            record.error_text = message[:500]
            record.current_step = None
    else:
        with _lock:
            record.status = "done"
