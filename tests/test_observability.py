"""API-free tests for SQLite AutoGen logging and safe log summaries."""

import sqlite3
import unittest
from contextlib import closing
from tempfile import TemporaryDirectory
from pathlib import Path

from autogen import ConversableAgent, runtime_logging

from src.observability import (
    format_recent_calls,
    log_agent_message,
    read_recent_agent_messages,
    read_recent_calls,
    start_logging,
    stop_logging,
)


class ObservabilityTests(unittest.TestCase):
    """Check that local messages are persisted and call views stay compact."""

    def test_agent_message_round_trip_and_empty_call_list(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "phishing_desk.db"
            start_logging(database_path)
            try:
                log_agent_message(
                    "Policy_Checker",
                    "Policy matches: raw-IP rule applies.",
                )
            finally:
                stop_logging()

            messages = read_recent_agent_messages(db_path=database_path)
            calls = read_recent_calls(db_path=database_path)

        self.assertEqual(messages[0]["agent"], "Policy_Checker")
        self.assertIn("raw-IP rule applies", messages[0]["message"])
        self.assertEqual(calls, [])
        self.assertEqual(
            format_recent_calls(calls),
            "No LLM calls have been recorded.",
        )

    def test_auto_gen_client_api_key_is_not_persisted(self) -> None:
        sentinel = "sk-" + ("T" * 48)
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "safe.db"
            start_logging(database_path)
            try:
                ConversableAgent(
                    name="SafeTestAgent",
                    llm_config={
                        "config_list": [
                            {"model": "offline-test-model", "api_key": sentinel}
                        ]
                    },
                )
            finally:
                stop_logging()

            with closing(sqlite3.connect(database_path)) as connection:
                stored_values = connection.execute(
                    """
                    SELECT init_args FROM agents
                    UNION ALL
                    SELECT init_args FROM oai_wrappers
                    UNION ALL
                    SELECT init_args FROM oai_clients
                    """
                ).fetchall()

        self.assertNotIn(sentinel, repr(stored_values))

    def test_call_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            read_recent_calls(limit=0)


if __name__ == "__main__":
    unittest.main()
