"""Expose the local phishing-desk workflow through a loopback-only API."""

import json
import re
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from main_phase5 import build_case_record, rerun_saved_case_debate
from main_phase6 import _retrieve_policy
from src.analyst import docker_is_available, run_analyst_chart
from src.approval import approval_loop
from src.case_store import (
    CASE_ID_PATTERN,
    DEFAULT_CASES_DIR,
    PROJECT_ROOT,
    create_case_id,
    list_cases,
    load_case,
    save_case,
)
from src.config import GROQ_API_KEY
from src.debate import normalize_text
from src.desk_chief import CASE_PLAN, run_desk_chief
from src.intake import (
    SUPPORTED_IMAGE_TYPES,
    load_email,
    trim_email_text,
)
from src.observability import (
    DEFAULT_LOG_DB_PATH,
    read_recent_calls,
    start_logging,
    stop_logging,
)
from src.rag import PolicyChecker
from src.report import write_report
from api.jobs import (
    JobRecord,
    active_job_for_case,
    get_job,
    set_job_step,
    submit_job,
)
from api.schemas import (
    AnalyzeResponse,
    CaseDetail,
    CaseSummary,
    CostResponse,
    DebateMessage,
    DecisionResponse,
    HealthResponse,
    JobResponse,
    LinkFinding,
    LogsResponse,
    NoteRequest,
    RejectRequest,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
CASES_DIR = DEFAULT_CASES_DIR
INPUTS_DIR = PROJECT_ROOT / "inputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOG_DB_PATH = DEFAULT_LOG_DB_PATH

_runtime_logging_lock = Lock()
_case_locks_guard = Lock()
_case_locks: dict[str, RLock] = {}
_URL_PATTERN = re.compile(
    r"""(?i)(?:https?://|www\.)[^\s<>"']+"""
)
_FIELD_PATTERN = re.compile(r"(?im)^\s*([A-Za-z][A-Za-z _-]*):\s*(.*?)\s*$")
_CONFIDENCE_PATTERN = re.compile(r"(?im)^\s*Confidence:\s*(.+?)\s*$")
_VERDICT_PATTERN = re.compile(r"(?i)\b(PHISHING|SUSPICIOUS|LEGIT)\b")
_LINK_LINE_PATTERN = re.compile(r"(?i)^\s*[-*]?\s*Link:\s*(\S.*?)\s*$")
_SCORE_PATTERN = re.compile(r"(?i)^\s*(?:[-*]\s*)?Risk Score:\s*(\d+)")
_REASONS_PATTERN = re.compile(r"(?i)^\s*(?:[-*]\s*)?Reasons:\s*(.*?)\s*$")
_AUTH_PATTERN = re.compile(
    r"(?i)^\s*(?:[-*]\s*)?(SPF|DKIM|DMARC):\s*(.*?)\s*$"
)

app = FastAPI(
    title="phishing-desk local API",
    description="Loopback-only wrapper for the defensive email-analysis workflow.",
    version="7.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return readiness indicators without disclosing any key material."""
    return HealthResponse(
        status="ok",
        groq_key_configured=bool(GROQ_API_KEY),
        docker_available=docker_is_available(),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse, status_code=202)
async def analyze(
    text: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    offline: bool = Form(default=False),
    no_docker: bool = Form(default=False),
    mask_pii: bool = Form(default=True),
) -> AnalyzeResponse:
    """Validate one input, remove temporary uploads, and queue analysis."""
    if (text is None) == (file is None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of pasted text or an uploaded file.",
        )

    source_type = "text"
    input_path: Path | None = None
    if text is not None:
        if not text.strip():
            raise HTTPException(status_code=422, detail="Email text is empty.")
        if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Email text exceeds 5 MiB.")
        email_text = text
    else:
        if file is None:
            raise HTTPException(status_code=422, detail="No email input was provided.")
        email_text, source_type, input_path = await _read_upload(file)

    if email_text is not None:
        email_text, _ = trim_email_text(email_text)
    case_id = create_case_id()

    def task(job_id: str, assigned_case_id: str) -> None:
        _run_analysis_job(
            job_id,
            assigned_case_id,
            email_text,
            input_path,
            source_type,
            offline,
            no_docker,
            mask_pii,
        )

    record = submit_job(case_id, task)
    return AnalyzeResponse(job_id=record.job_id, case_id=record.case_id)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    """Return the current state of an analysis or rejection job."""
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(record)


@app.get("/api/cases", response_model=list[CaseSummary])
def cases() -> list[CaseSummary]:
    """List saved cases with their verdict confidence and creation date."""
    result: list[CaseSummary] = []
    for row in list_cases(CASES_DIR):
        try:
            case = load_case(row["case_id"], CASES_DIR)
        except FileNotFoundError:
            continue
        verdict_text = normalize_text(str(case.get("current_verdict", "")))
        confidence_match = _CONFIDENCE_PATTERN.search(verdict_text)
        verdict_match = _VERDICT_PATTERN.search(verdict_text)
        result.append(
            CaseSummary(
                id=row["case_id"],
                status=str(case.get("status", row["status"])),
                verdict=verdict_match.group(1).upper() if verdict_match else "UNKNOWN",
                confidence=confidence_match.group(1) if confidence_match else None,
                date=str(case.get("created_at", row["date"])),
            )
        )
    return result


@app.get("/api/cases/{case_id}", response_model=CaseDetail)
def case_detail(case_id: str) -> CaseDetail:
    """Return a PII-filtered case, evidence, debate, and safe local URLs."""
    case = _load_case(case_id)
    summaries = case.get("pipeline_summaries", [])
    if not isinstance(summaries, list):
        summaries = []
    facts = _parse_extracted_facts(
        str(summaries[0]) if summaries else ""
    )
    link_summary = str(summaries[1]) if len(summaries) > 1 else ""
    findings, sender_findings, auth_results = _parse_link_findings(link_summary)
    if not findings:
        extracted_links = facts.get("Links", [])
        if isinstance(extracted_links, list):
            findings = [
                _safe_link_finding({"url": value, "reasons": []})
                for value in extracted_links
                if isinstance(value, str)
            ]
    facts["Links"] = [
        {"url": item["url"], "display_url": item["display_url"]}
        for item in findings
    ]

    chart_url = (
        f"/api/cases/{case_id}/chart"
        if (_report_file(case_id, chart=True)).is_file()
        else None
    )
    report_url = (
        f"/api/cases/{case_id}/report"
        if (
            str(case.get("status")) == "APPROVED"
            and _report_file(case_id, chart=False).is_file()
        )
        else None
    )
    return CaseDetail(
        id=case_id,
        source_type=str(case.get("source_type", "text")),
        email_text=defang_text(str(case.get("email_text", ""))),
        current_verdict=defang_text(
            normalize_text(str(case.get("current_verdict", "")))
        ),
        case_plan=[normalize_text(step) for step in CASE_PLAN],
        extracted_facts=_safe_value(facts),
        sender_domain_findings=defang_text(sender_findings),
        link_findings=[LinkFinding(**finding) for finding in findings],
        auth_results=[defang_text(item) for item in auth_results],
        debate_history=_debate_messages(case.get("debate_history", [])),
        policy_matches=_safe_list(case.get("policy_matches")),
        similar_past_cases=_safe_list(case.get("similar_past_cases")),
        analyst_notes=_safe_strings(case.get("analyst_notes")),
        status=str(case.get("status", "UNKNOWN")),
        cost=_safe_value(_cost_record(case)),
        chart_url=chart_url,
        report_url=report_url,
        created_at=str(case.get("created_at", "")),
        updated_at=str(case.get("updated_at", "")),
    )


@app.post("/api/cases/{case_id}/approve", response_model=DecisionResponse)
def approve_case(case_id: str) -> DecisionResponse:
    """Approve an open case through the existing injectable approval loop."""
    with _case_lock(case_id):
        _ensure_no_active_job(case_id)
        case = _load_case(case_id)
        if case.get("status") == "NEEDS_REVIEW":
            raise HTTPException(
                status_code=409,
                detail="This case needs human review and cannot be auto-approved.",
            )
        if case.get("status") != "APPROVED":
            updated = approval_loop(
                case,
                input_fn=lambda _prompt: "a",
                save_fn=_persist_case,
            )
            _write_final_report(updated)
        elif not _report_file(case_id, chart=False).is_file():
            _write_final_report(case)
        return _decision_response(case)


@app.post(
    "/api/cases/{case_id}/reject",
    response_model=AnalyzeResponse,
    status_code=202,
)
def reject_case(case_id: str, request: RejectRequest) -> AnalyzeResponse:
    """Queue one saved-debate continuation with the analyst's challenge."""
    reason = normalize_text(request.reason.strip())
    if not reason:
        raise HTTPException(status_code=422, detail="Rejection reason is empty.")
    with _case_lock(case_id):
        _ensure_no_active_job(case_id)
        case = _load_case(case_id)
        reasons = case.get("rejection_reasons", [])
        if not isinstance(reasons, list):
            raise HTTPException(status_code=409, detail="Case rejection history is invalid.")
        if case.get("status") == "NEEDS_REVIEW" or len(reasons) >= 2:
            raise HTTPException(
                status_code=409,
                detail="The case has reached the two-rejection review limit.",
            )
        if case.get("status") == "APPROVED":
            raise HTTPException(status_code=409, detail="An approved case cannot be rejected.")

        def task(job_id: str, assigned_case_id: str) -> None:
            _run_rejection_job(job_id, assigned_case_id, reason)

        record = submit_job(case_id, task)
    return AnalyzeResponse(job_id=record.job_id, case_id=case_id)


@app.post("/api/cases/{case_id}/note", response_model=DecisionResponse)
def add_note(case_id: str, request: NoteRequest) -> DecisionResponse:
    """Append and persist an analyst note without rerunning analysis."""
    note = normalize_text(request.note.strip())
    if not note:
        raise HTTPException(status_code=422, detail="Analyst note is empty.")
    with _case_lock(case_id):
        _ensure_no_active_job(case_id)
        case = _load_case(case_id)
        notes = case.setdefault("analyst_notes", [])
        if not isinstance(notes, list):
            raise HTTPException(status_code=409, detail="Case notes are invalid.")
        notes.append(note)
        _persist_case(case)
        if case.get("status") == "APPROVED":
            _write_final_report(case)
    return _decision_response(case)


@app.post("/api/cases/{case_id}/save", response_model=DecisionResponse)
def save_open_case(case_id: str) -> DecisionResponse:
    """Save an unfinished case as OPEN without rerunning its pipeline."""
    with _case_lock(case_id):
        _ensure_no_active_job(case_id)
        case = _load_case(case_id)
        case["status"] = "OPEN"
        case.pop("report_path", None)
        case.pop("report_generated_at", None)
        _persist_case(case)
    return _decision_response(case)


@app.get(
    "/api/cases/{case_id}/report",
    response_class=PlainTextResponse,
)
def case_report(case_id: str) -> PlainTextResponse:
    """Return the generated Markdown report for an approved case."""
    case = _load_case(case_id)
    path = _report_file(case_id, chart=False)
    if case.get("status") != "APPROVED" or not path.is_file():
        raise HTTPException(status_code=404, detail="Approved case report not found.")
    return PlainTextResponse(
        defang_text(path.read_text(encoding="utf-8")),
        media_type="text/markdown",
    )


@app.get("/api/cases/{case_id}/chart")
def case_chart(case_id: str) -> FileResponse:
    """Return a generated case chart as a PNG file."""
    _load_case(case_id)
    path = _report_file(case_id, chart=True)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Case chart not found.")
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.get("/api/logs", response_model=LogsResponse)
def logs(limit: int = Query(default=20, ge=1, le=200)) -> LogsResponse:
    """Return recent LLM metadata without request or response content."""
    return LogsResponse(calls=read_recent_calls(limit, LOG_DB_PATH))


@app.get("/api/costs", response_model=CostResponse)
def costs() -> CostResponse:
    """Sum saved AutoGen usage and the cases created today (UTC)."""
    today = datetime.now(timezone.utc).date()
    total_cost = 0.0
    today_cost = 0.0
    for row in list_cases(CASES_DIR):
        try:
            case = load_case(row["case_id"], CASES_DIR)
        except FileNotFoundError:
            continue
        case_cost = _case_total_cost(case)
        total_cost += case_cost
        created_at = _parse_datetime(str(case.get("created_at", "")))
        if created_at is not None and created_at.date() == today:
            today_cost += case_cost
    return CostResponse(
        total_cost=round(total_cost, 8),
        cost_today=round(today_cost, 8),
    )


async def _read_upload(
    upload: UploadFile,
) -> tuple[str | None, str, Path | None]:
    """Validate and briefly stage one accepted upload inside ignored inputs/."""
    filename = upload.filename or ""
    suffix = Path(filename.replace("\\", "/")).suffix.casefold()
    accepted_text = {".txt", ".eml"}
    accepted_images = set(SUPPORTED_IMAGE_TYPES)
    if suffix not in accepted_text | accepted_images:
        await upload.close()
        raise HTTPException(
            status_code=415,
            detail="Upload must be .txt, .eml, .png, .jpg, .jpeg, or .webp.",
        )

    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds 5 MiB.")
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    _validate_upload_content(suffix, payload)

    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = INPUTS_DIR / f"{uuid4().hex}{suffix}"
    keep_for_background_job = False
    try:
        temporary_path.write_bytes(payload)
        if suffix in accepted_text:
            return load_email(file=temporary_path), "file", None
        keep_for_background_job = True
        return None, "image", temporary_path
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=422,
            detail=normalize_text(str(error)),
        ) from error
    finally:
        if not keep_for_background_job:
            temporary_path.unlink(missing_ok=True)


def _validate_upload_content(suffix: str, payload: bytes) -> None:
    """Check image signatures as well as file extensions."""
    image_signatures = {
        ".png": payload.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": payload.startswith(b"\xff\xd8\xff"),
        ".jpeg": payload.startswith(b"\xff\xd8\xff"),
        ".webp": (
            len(payload) >= 12
            and payload.startswith(b"RIFF")
            and payload[8:12] == b"WEBP"
        ),
    }
    if suffix in image_signatures and not image_signatures[suffix]:
        raise HTTPException(
            status_code=415,
            detail="Uploaded image bytes do not match the file extension.",
        )
    if suffix in {".txt", ".eml"}:
        try:
            payload.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=415,
                detail="Uploaded email text must be UTF-8.",
            ) from error


def _run_analysis_job(
    job_id: str,
    case_id: str,
    email_text: str | None,
    input_path: Path | None,
    source_type: str,
    offline: bool,
    no_docker: bool,
    mask_pii: bool,
) -> None:
    """Run the existing Desk_Chief, policy, and Analyst functions in order."""
    try:
        if input_path is not None:
            email_text = load_email(image=input_path)
        if email_text is None or not email_text.strip():
            raise ValueError("Email input is empty.")
        email_text, _ = trim_email_text(email_text)

        with _runtime_logging_lock:
            logging_started = False
            try:
                start_logging(LOG_DB_PATH)
                logging_started = True
                result = run_desk_chief(
                    email_text,
                    offline=offline,
                    progress_callback=lambda step: set_job_step(job_id, step),
                )
                case = build_case_record(
                    email_text,
                    source_type,
                    result,
                    case_id=case_id,
                )
                case["mask_pii"] = mask_pii
                case["offline"] = offline
                case["no_docker"] = no_docker

                set_job_step(job_id, "Policy")
                _retrieve_policy(PolicyChecker(), case)
                set_job_step(job_id, "Chart")
                chart_path = run_analyst_chart(
                    case_id,
                    email_text,
                    no_docker=no_docker,
                    reports_dir=REPORTS_DIR,
                )
                if chart_path is not None:
                    case["chart_path"] = _relative_path(chart_path)
                set_job_step(job_id, "Awaiting approval")
                _persist_case(case)
            finally:
                if logging_started:
                    stop_logging()
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)


def _run_rejection_job(job_id: str, case_id: str, reason: str) -> None:
    """Resume one saved debate, refresh evidence, and apply the 2-rejection cap."""
    with _case_lock(case_id), _runtime_logging_lock:
        case = _load_case(case_id)
        rejection_reasons = case.get("rejection_reasons", [])
        if not isinstance(rejection_reasons, list):
            raise RuntimeError("The saved rejection history is invalid.")
        if len(rejection_reasons) >= 2:
            case["status"] = "NEEDS_REVIEW"
            set_job_step(job_id, "Awaiting approval")
            _persist_case(case)
            return

        logging_started = False
        try:
            start_logging(LOG_DB_PATH)
            logging_started = True

            def rerun(
                current_case: dict[str, Any],
                challenge: str,
            ) -> dict[str, Any]:
                set_job_step(job_id, "Debate")
                updated = rerun_saved_case_debate(
                    current_case,
                    challenge,
                    verbose=False,
                    offline=bool(current_case.get("offline", False)),
                )
                set_job_step(job_id, "Policy")
                _retrieve_policy(PolicyChecker(), updated)
                set_job_step(job_id, "Chart")
                chart_path = run_analyst_chart(
                    case_id,
                    str(updated.get("email_text", "")),
                    no_docker=bool(updated.get("no_docker", False)),
                    reports_dir=REPORTS_DIR,
                )
                if chart_path is None:
                    updated.pop("chart_path", None)
                else:
                    updated["chart_path"] = _relative_path(chart_path)
                updated.pop("report_path", None)
                updated.pop("report_generated_at", None)
                set_job_step(job_id, "Awaiting approval")
                return updated

            responses = iter(("r", reason, "s"))
            with redirect_stdout(StringIO()):
                approval_loop(
                    case,
                    input_fn=lambda _prompt: next(responses),
                    save_fn=_persist_case,
                    rerun_debate_fn=rerun,
                    max_rejections=2,
                )
        finally:
            if logging_started:
                stop_logging()


def _write_final_report(case: dict[str, Any]) -> None:
    """Use the existing report writer, then persist its path and final state."""
    case["report_generated_at"] = datetime.now(timezone.utc).isoformat()
    report_path = write_report(case, REPORTS_DIR)
    case["report_path"] = _relative_path(report_path)
    _persist_case(case)


def _persist_case(case: dict[str, Any]) -> None:
    """Save with the PII preference recorded when the case was created."""
    save_case(
        case,
        CASES_DIR,
        mask_pii=bool(case.get("mask_pii", True)),
    )


def _load_case(case_id: str) -> dict[str, Any]:
    """Translate validated case-store errors to safe HTTP responses."""
    try:
        return load_case(case_id, CASES_DIR)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Case not found.") from error


def _case_lock(case_id: str) -> RLock:
    """Return a per-case lock to keep approval changes serial."""
    with _case_locks_guard:
        return _case_locks.setdefault(case_id, RLock())


def _ensure_no_active_job(case_id: str) -> None:
    active = active_job_for_case(case_id)
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Case already has active job {active.job_id}.",
        )


def _decision_response(case: dict[str, Any]) -> DecisionResponse:
    return DecisionResponse(
        case_id=str(case["case_id"]),
        status=str(case.get("status", "UNKNOWN")),
        current_verdict=defang_text(
            normalize_text(str(case.get("current_verdict", "")))
        ),
    )


def _job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        job_id=record.job_id,
        case_id=record.case_id,
        status=record.status,
        current_step=record.current_step,
        error_text=record.error_text,
    )


def _report_file(case_id: str, chart: bool) -> Path:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise HTTPException(status_code=400, detail="Invalid case ID format.")
    suffix = "_chart.png" if chart else ".md"
    return REPORTS_DIR / f"{case_id}{suffix}"


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _parse_extracted_facts(summary: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for match in _FIELD_PATTERN.finditer(normalize_text(summary)):
        label = match.group(1).strip().casefold()
        value = match.group(2).strip()
        if label == "headers":
            try:
                facts["Headers"] = json.loads(value)
            except json.JSONDecodeError:
                facts["Headers"] = value
        elif label == "links":
            try:
                parsed_links = json.loads(value)
                facts["Links"] = (
                    [item for item in parsed_links if isinstance(item, str)]
                    if isinstance(parsed_links, list)
                    else []
                )
            except json.JSONDecodeError:
                facts["Links"] = []
        elif label in {"sender", "keywords"}:
            facts[label.title()] = value
    return facts


def _parse_link_findings(
    summary: str,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Extract sender, link risk, and authentication facts from Phase 2."""
    normalized = normalize_text(summary)
    lines = normalized.splitlines()
    links: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    sender_lines: list[str] = []
    auth_results: list[str] = []
    in_link_section = False
    for line in lines:
        if re.match(r"(?i)^\s*Links\s*:\s*$", line):
            in_link_section = True
            continue
        link_match = _LINK_LINE_PATTERN.match(line)
        if link_match:
            if current is not None:
                links.append(_safe_link_finding(current))
            raw_url = link_match.group(1).strip("`\"'")
            current = {
                "url": raw_url,
                "risk_score": None,
                "reasons": [],
            }
            continue
        score_match = _SCORE_PATTERN.match(line)
        if score_match and current is not None:
            current["risk_score"] = int(score_match.group(1))
            continue
        reason_match = _REASONS_PATTERN.match(line)
        if reason_match and current is not None:
            current["reasons"] = [
                item.strip()
                for item in reason_match.group(1).split(";")
                if item.strip()
            ]
            continue
        auth_match = _AUTH_PATTERN.match(line)
        if auth_match:
            auth_results.append(
                f"{auth_match.group(1).upper()}: {auth_match.group(2).strip()}"
            )
        if not in_link_section and line.strip():
            sender_lines.append(line.strip())
    if current is not None:
        links.append(_safe_link_finding(current))
    if not auth_results:
        auth_results = [
            "SPF: not available",
            "DKIM: not available",
            "DMARC: not available",
        ]
    return links, "\n".join(sender_lines), auth_results


def _safe_link_finding(finding: dict[str, Any]) -> dict[str, Any]:
    url = str(finding.get("url", ""))
    display_url = defang_url(url)
    return {
        "url": display_url,
        "display_url": display_url,
        "risk_score": finding.get("risk_score"),
        "reasons": [
            defang_text(str(reason))
            for reason in finding.get("reasons", [])
        ],
    }


def _debate_messages(history: Any) -> list[DebateMessage]:
    if not isinstance(history, list):
        return []
    result: list[DebateMessage] = []
    for attempt_record in history:
        if not isinstance(attempt_record, dict):
            continue
        attempt = int(attempt_record.get("attempt", 1) or 1)
        messages = attempt_record.get("messages", [])
        if not isinstance(messages, list):
            continue
        round_number = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            speaker = str(message.get("name", ""))
            if speaker not in {"Classifier", "Skeptic"}:
                continue
            if speaker == "Classifier":
                round_number += 1
            if round_number == 0:
                round_number = 1
            content = message.get("content")
            if isinstance(content, str):
                result.append(
                    DebateMessage(
                        attempt=attempt,
                        round=round_number,
                        speaker=speaker,
                        message=defang_text(normalize_text(content)),
                    )
                )
    return result


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return defang_text(normalize_text(value))
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    return value


def _safe_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        _safe_value(item)
        for item in value
        if isinstance(item, dict)
    ]


def _safe_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        defang_text(normalize_text(item))
        for item in value
        if isinstance(item, str)
    ]


def defang_url(url: str) -> str:
    """Defang a URL's scheme and host without changing its path or query."""
    value = normalize_text(url.strip())
    if not value:
        return value
    scheme_prefix = ""
    remainder = value
    if "://" in value:
        scheme, remainder = value.split("://", 1)
        normalized_scheme = scheme.casefold()
        scheme_prefix = {
            "http": "hxxp://",
            "https": "hxxps://",
        }.get(normalized_scheme, f"{scheme}://")
    authority_match = re.match(r"([^/?#]*)(.*)", remainder)
    if authority_match is None:
        return scheme_prefix + value.replace(".", "[.]")
    authority, suffix = authority_match.groups()
    userinfo = ""
    if "@" in authority:
        userinfo, authority = authority.rsplit("@", 1)
        userinfo += "@"
    if authority.startswith("["):
        safe_authority = authority
    else:
        host, separator, port = authority.partition(":")
        safe_authority = host.replace(".", "[.]")
        if separator:
            safe_authority += separator + port
    return scheme_prefix + userinfo + safe_authority + suffix


def defang_text(text: str) -> str:
    """Replace embedded HTTP links with safe, non-clickable display text."""
    normalized = normalize_text(text)

    def replace_match(match: re.Match[str]) -> str:
        raw_url = match.group(0)
        cleaned_url = raw_url.rstrip(".,;:!?)]}")
        punctuation = raw_url[len(cleaned_url) :]
        return defang_url(cleaned_url) + punctuation

    return _URL_PATTERN.sub(replace_match, normalized)


def _case_total_cost(case: dict[str, Any]) -> float:
    total = case.get("total_cost", {})
    if not isinstance(total, dict):
        return 0.0
    usage = total.get("usage_including_cached_inference", {})
    if not isinstance(usage, dict):
        return 0.0
    try:
        return float(usage.get("total_cost", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _cost_record(case: dict[str, Any]) -> dict[str, Any]:
    value = case.get("total_cost", {})
    return value if isinstance(value, dict) else {}


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)
