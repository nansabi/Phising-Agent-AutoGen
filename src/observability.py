"""Start AutoGen SQLite logging and read compact call/message summaries."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from autogen import runtime_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DB_PATH = PROJECT_ROOT / "logs" / "phishing_desk.db"


def start_logging(db_path: str | Path = DEFAULT_LOG_DB_PATH) -> str:
    """Start AutoGen's SQLite runtime logger and verify its connection."""
    if runtime_logging.logging_enabled():
        raise RuntimeError("AutoGen runtime logging is already active.")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        session_id = runtime_logging.start(
            logger_type="sqlite",
            config={"dbname": str(path)},
        )
        connection = runtime_logging.get_connection()
        if not runtime_logging.logging_enabled() or connection is None:
            raise RuntimeError("AutoGen runtime logging did not start.")
        connection.execute("SELECT 1").fetchone()
    except (OSError, sqlite3.Error, RuntimeError) as error:
        if runtime_logging.logging_enabled():
            runtime_logging.stop()
        raise RuntimeError(f"Could not start SQLite observability at '{path}'.") from error
    return session_id


def stop_logging() -> None:
    """Stop the AutoGen runtime logger if it is active."""
    if runtime_logging.logging_enabled():
        runtime_logging.stop()


def log_agent_message(agent: str, message: str) -> None:
    """Write an explicit local-agent message to AutoGen's events table."""
    if not runtime_logging.logging_enabled():
        raise RuntimeError("Cannot log an agent message before logging is started.")
    runtime_logging.log_event(
        agent,
        "agent_message",
        message=message,
    )


def read_recent_calls(
    limit: int = 20,
    db_path: str | Path = DEFAULT_LOG_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent completion metadata without exposing request/response text."""
    if limit < 1:
        raise ValueError("The log limit must be at least one.")
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        with closing(sqlite3.connect(str(path))) as connection:
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(
                """
                SELECT source_name, request, response, cost, start_time, end_time
                FROM chat_completions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not read observability database '{path}'.") from error

    return [
        _format_call_row(source, request, response, cost, start, end)
        for source, request, response, cost, start, end in rows
    ]


def read_recent_agent_messages(
    limit: int = 20,
    db_path: str | Path = DEFAULT_LOG_DB_PATH,
) -> list[dict[str, str]]:
    """Read recent AutoGen agent-message events for tests and local inspection."""
    if limit < 1:
        raise ValueError("The log limit must be at least one.")
    path = Path(db_path)
    if not path.is_file():
        return []
    try:
        with closing(sqlite3.connect(str(path))) as connection:
            connection.execute("PRAGMA query_only = ON")
            rows = connection.execute(
                """
                SELECT source_name, json_state, timestamp
                FROM events
                WHERE event_name IN ('received_message', 'agent_message')
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not read observability database '{path}'.") from error

    messages: list[dict[str, str]] = []
    for source, state_json, timestamp in rows:
        try:
            state = json.loads(state_json or "{}")
        except json.JSONDecodeError as error:
            raise RuntimeError("An agent-message log entry is malformed.") from error
        message = state.get("message", "")
        if isinstance(message, dict):
            message = message.get("content", "")
        messages.append(
            {
                "agent": str(source or "Unknown"),
                "message": str(message or ""),
                "time": str(timestamp or ""),
            }
        )
    return messages


def format_recent_calls(calls: list[dict[str, Any]]) -> str:
    """Render recent calls as a compact table for the --show-log option."""
    if not calls:
        return "No LLM calls have been recorded."
    header = (
        f"{'AGENT':<20} {'MODEL':<28} {'PROMPT':>8} {'COMPLETION':>10} "
        f"{'COST':>12} TIME"
    )
    rows = [
        f"{str(call['agent'])[:20]:<20} {str(call['model'])[:28]:<28} "
        f"{call['prompt_tokens']:>8} {call['completion_tokens']:>10} "
        f"{call['cost']:>12.6f} {call['time']}"
        for call in calls
    ]
    return "\n".join([header, *rows])


def _format_call_row(
    source: Any,
    request_json: Any,
    response_json: Any,
    cost: Any,
    start_time: Any,
    end_time: Any,
) -> dict[str, Any]:
    request = _parse_json(request_json)
    response = _parse_json(response_json)
    if isinstance(response, dict) and isinstance(response.get("response"), str):
        nested_response = _parse_json(response["response"])
        if isinstance(nested_response, dict):
            response = nested_response
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    model = ""
    if isinstance(request, dict):
        model = str(request.get("model") or "")
    if not model and isinstance(response, dict):
        model = str(response.get("model") or "")
    return {
        "agent": str(source or "Unknown"),
        "model": model or "unknown",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost": float(cost or 0),
        "time": str(end_time or start_time or ""),
    }


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}
