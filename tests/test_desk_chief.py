"""API-free tests for the Desk_Chief formatter and nested-chat wrapper."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from autogen.agentchat.chat import ChatResult

from src.debate import DebateResult
from src.pipeline import STEP_NAMES
from src.desk_chief import (
    create_desk_chief,
    format_final_reply,
    print_nested_steps,
    run_desk_chief,
)


def make_debate_result() -> DebateResult:
    """Build a representative debate result without an API call."""
    classifier_message = "\n".join(
        (
            "Verdict: PHISHING",
            "Confidence: 92%",
            "Sender domain findings: sender domain has risk indicators.",
            "Reasons:",
            "- The link uses a raw IP.",
            "- The message requests account verification.",
            "- The link does not match the sender domain.",
        )
    )
    return DebateResult(
        phase2_results=[],
        chat_result=ChatResult(summary="AGREE", chat_history=[], cost={}),
        messages=[],
        rounds_used=2,
        skeptic_agreed=True,
        round_limit_hit=False,
        final_classifier_message=classifier_message,
        skeptic_last_message="AGREE",
        cost={},
    )


class DeskChiefTests(unittest.TestCase):
    """Exercise formatting, verbosity, registration, and nested carryover."""

    def test_final_reply_has_the_required_case_fields(self) -> None:
        result = format_final_reply(make_debate_result())

        self.assertEqual(
            result,
            "\n".join(
                (
                    "Verdict: PHISHING",
                    "Confidence: 92%",
                    "Sender domain findings: sender domain has risk indicators.",
                    "Reasons:",
                    "1. The link uses a raw IP.",
                    "2. The message requests account verification.",
                    "3. The link does not match the sender domain.",
                    "Debate rounds used: 2",
                    "Skeptic status: AGREED",
                )
            ),
        )

    def test_nested_steps_are_quiet_unless_verbose(self) -> None:
        phase2_results = [
            ChatResult(summary=f"{name} summary", cost={})
            for name in STEP_NAMES
        ]
        debate_result = make_debate_result()

        quiet_output = StringIO()
        with redirect_stdout(quiet_output):
            print_nested_steps(phase2_results, debate_result, verbose=False)
        self.assertEqual(quiet_output.getvalue(), "")

        verbose_output = StringIO()
        with redirect_stdout(verbose_output):
            print_nested_steps(phase2_results, debate_result, verbose=True)
        output = verbose_output.getvalue()
        self.assertIn("Nested step: Extractor", output)
        self.assertIn("Extractor summary", output)
        self.assertIn("Nested step: Debate", output)
        self.assertIn("Skeptic status: AGREED", output)

    def test_nested_chats_use_user_proxy_as_the_trigger(self) -> None:
        session = create_desk_chief()

        triggers = [
            entry["trigger"] for entry in session.desk_chief._reply_func_list
        ]

        self.assertIn(session.user_proxy, triggers)
        self.assertEqual(session.desk_chief.name, "Desk_Chief")
        self.assertEqual(session.user_proxy.name, "User_Proxy")

    def test_nested_debate_receives_pipeline_summary_as_carryover(self) -> None:
        phase2_results = [
            ChatResult(summary=f"{name} findings", cost={})
            for name in STEP_NAMES
        ]
        debate_result = make_debate_result()
        with (
            patch("src.desk_chief.run_pipeline", return_value=phase2_results),
            patch("src.desk_chief.run_debate", return_value=debate_result) as run_debate,
        ):
            result = run_desk_chief("From: sender@example.test\n\nExample email")

        self.assertIn("Verdict: PHISHING", result.final_reply)
        run_debate.assert_called_once()
        self.assertIn(
            "Extractor findings",
            run_debate.call_args.kwargs["pipeline_carryover"],
        )


if __name__ == "__main__":
    unittest.main()
