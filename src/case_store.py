"""Save, load, and list privacy-filtered Phase 5 case files."""

import copy
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.intake import mask_pii_text, sender_domain_from_email

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_DIR = PROJECT_ROOT / "cases"
CASE_ID_PATTERN = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")
CASE_STATUSES = {"OPEN", "APPROVED", "REJECTED", "NEEDS_REVIEW"}
SOURCE_TYPES = {"text", "file", "image"}


def create_case_id() -> str:
    """Create an ID from the UTC timestamp and a short random hash."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    short_hash = hashlib.sha256(
        f"{timestamp}:{uuid.uuid4().hex}".encode("ascii")
    ).hexdigest()[:8]
    return f"{timestamp}-{short_hash}"


def save_case(
    case: dict[str, Any],
    cases_dir: str | Path = DEFAULT_CASES_DIR,
    mask_pii: bool = True,
) -> Path:
    """Write one case as JSON, masking personal data unless opted out."""
    case_id = str(case.get("case_id", ""))
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("The case has an invalid case_id.")
    status = case.get("status")
    if status not in CASE_STATUSES:
        raise ValueError(f"Invalid case status: {status!r}.")
    source_type = case.get("source_type")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Invalid input source type: {source_type!r}.")

    now = datetime.now(timezone.utc).isoformat()
    case.setdefault("created_at", now)
    case["updated_at"] = now
    record = copy.deepcopy(case)
    if mask_pii:
        sender_domain = sender_domain_from_email(
            str(record.get("email_text", ""))
        )
        record = _mask_case_values(record, sender_domain)

    directory = Path(cases_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case_id}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def load_case(
    case_id: str,
    cases_dir: str | Path = DEFAULT_CASES_DIR,
) -> dict[str, Any]:
    """Load one validated case file by ID."""
    path = _case_path(case_id, cases_dir)
    try:
        case = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Case '{case_id}' was not found.") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read case '{case_id}'.") from error
    if not isinstance(case, dict) or case.get("case_id") != case_id:
        raise RuntimeError(f"Case file '{case_id}' has an invalid structure.")
    return case


def list_cases(cases_dir: str | Path = DEFAULT_CASES_DIR) -> list[dict[str, str]]:
    """Return a concise, newest-first case listing."""
    directory = Path(cases_dir)
    if not directory.exists():
        return []

    cases: list[dict[str, str]] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read case file '{path.name}'.") from error
        if not isinstance(data, dict):
            raise RuntimeError(f"Case file '{path.name}' has an invalid structure.")
        cases.append(
            {
                "case_id": str(data.get("case_id", path.stem)),
                "status": str(data.get("status", "UNKNOWN")),
                "verdict": _verdict_label(str(data.get("current_verdict", ""))),
                "date": str(data.get("created_at", "")),
            }
        )
    return sorted(cases, key=lambda case: case["date"], reverse=True)


def format_case_list(cases: list[dict[str, str]]) -> str:
    """Format the case listing as a small terminal table."""
    if not cases:
        return "No saved cases."
    heading = f"{'CASE ID':<34} {'STATUS':<14} {'VERDICT':<12} DATE"
    rows = [
        f"{case['case_id']:<34} {case['status']:<14} "
        f"{case['verdict']:<12} {case['date']}"
        for case in cases
    ]
    return "\n".join([heading, *rows])


def _case_path(case_id: str, cases_dir: str | Path) -> Path:
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("Invalid case ID format.")
    return Path(cases_dir) / f"{case_id}.json"


def _mask_case_values(value: Any, sender_domain: str | None) -> Any:
    if isinstance(value, str):
        return mask_pii_text(value, sender_domain)
    if isinstance(value, list):
        return [_mask_case_values(item, sender_domain) for item in value]
    if isinstance(value, dict):
        return {
            key: _mask_case_values(item, sender_domain)
            for key, item in value.items()
        }
    return value


def _verdict_label(verdict: str) -> str:
    match = re.search(r"(?i)\b(PHISHING|SUSPICIOUS|LEGIT)\b", verdict)
    return match.group(1).upper() if match else "UNKNOWN"
