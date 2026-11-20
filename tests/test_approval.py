"""API-free tests for the human approval loop and resume behavior."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.approval import approval_loop
from src.case_store import create_case_id, save_case
from main_phase5 import resume_case


def make_case() -> dict[str, object]:
    """Create a small in-memory case for approval tests."""
    return {
        "case_id": create_case_id(),
        "source_type": "text",
        "email_text": "From: sender@example.test",
        "pipeline_summaries": [],
        "debate_history": [],
        "current_verdict": "Verdict: PHISHING\nConfidence: 90%",
        "status": "OPEN",
        "analyst_notes": [],
        "rejection_reasons": [],
    }


class ApprovalTests(unittest.TestCase):
    """Exercise decisions with injected input and persistence callbacks."""

    def test_approve_flow(self) -> None:
        case = make_case()
        saved: list[str] = []
        inputs = iter(["a"])

        updated = approval_loop(
            case,
            input_fn=lambda _prompt: next(inputs),
            save_fn=lambda value: saved.append(str(value["status"])),
        )

        self.assertEqual(updated["status"], "APPROVED")
        self.assertEqual(saved, ["APPROVED"])

    def test_reject_flow_stops_after_two_challenges(self) -> None:
        case = make_case()
        inputs = iter(["r", "First concern", "r", "Second concern"])
        challenges: list[str] = []

        def rerun(
            current_case: dict[str, object],
            reason: str,
        ) -> dict[str, object]:
            challenges.append(reason)
            current_case["current_verdict"] = f"Updated after: {reason}"
            return current_case

        updated = approval_loop(
            case,
            input_fn=lambda _prompt: next(inputs),
            save_fn=lambda _case: None,
            rerun_debate_fn=rerun,
        )

        self.assertEqual(updated["status"], "NEEDS_REVIEW")
        self.assertEqual(
            updated["rejection_reasons"],
            ["First concern", "Second concern"],
        )
        self.assertEqual(challenges, ["First concern", "Second concern"])

    def test_reject_then_approve_uses_the_revised_verdict(self) -> None:
        case = make_case()
        inputs = iter(["r", "The link was missed", "a"])

        def rerun(
            current_case: dict[str, object],
            reason: str,
        ) -> dict[str, object]:
            current_case["current_verdict"] = f"Revised verdict for {reason}"
            return current_case

        with patch("builtins.print") as printed:
            updated = approval_loop(
                case,
                input_fn=lambda _prompt: next(inputs),
                save_fn=lambda _case: None,
                rerun_debate_fn=rerun,
            )

        self.assertEqual(updated["status"], "APPROVED")
        self.assertTrue(
            any(
                "Revised verdict for The link was missed" in str(call)
                for call in printed.call_args_list
            )
        )

    def test_resume_approval_does_not_run_pipeline(self) -> None:
        case = make_case()
        with TemporaryDirectory() as temporary_directory:
            save_case(case, temporary_directory)
            with patch("main_phase5.run_desk_chief") as run_analysis:
                updated = resume_case(
                    case["case_id"],
                    input_fn=lambda _prompt: "a",
                    cases_dir=temporary_directory,
                )

        run_analysis.assert_not_called()
        self.assertEqual(updated["status"], "APPROVED")


if __name__ == "__main__":
    unittest.main()
