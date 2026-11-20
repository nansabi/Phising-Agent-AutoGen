"""API-free tests for deterministic Markdown report generation."""

import unittest

from src.report import build_report


class ReportTests(unittest.TestCase):
    """Ensure every required case section appears in the written report."""

    def test_report_contains_all_required_sections_and_policy_matches(self) -> None:
        case = {
            "case_id": "20261120T045724208818Z-82539024",
            "current_verdict": (
                "Verdict: PHISHING\n"
                "Confidence: 90%\n"
                "Sender domain findings: example.test lacks authentication."
            ),
            "pipeline_summaries": [
                "Sender: sender@example.test",
                "SPF: fail\nDKIM: not available\n"
                "Link: http://192.0.2.44/login uses a raw IP.",
            ],
            "debate_rounds_used": 2,
            "skeptic_status": "AGREED",
            "debate_history": [
                {
                    "messages": [
                        {
                            "name": "Classifier",
                            "content": "PHISHING with raw-IP and verification evidence.",
                        },
                        {"name": "Skeptic", "content": "AGREE"},
                    ]
                }
            ],
            "policy_matches": [
                {
                    "rule_id": "R1",
                    "rule_text": "Never click links with raw IP addresses.",
                    "why_matches": "A numeric IP address appears in a link.",
                }
            ],
            "similar_past_cases": [
                {
                    "case_id": "CASE-01",
                    "summary": "Raw-IP account-verification lure.",
                    "why_similar": "Matched evidence terms: account, ip.",
                }
            ],
            "analyst_notes": ["Reviewed the raw-IP finding."],
            "chart_path": "reports/20261120T045724208818Z-82539024_chart.png",
            "rejection_reasons": [],
            "status": "APPROVED",
            "created_at": "2026-11-20T04:55:00+00:00",
            "updated_at": "2026-11-20T04:58:00+00:00",
            "report_generated_at": "2026-11-20T04:59:00+00:00",
        }

        report = build_report(case)

        for heading in (
            "## Verdict",
            "## Sender domain findings",
            "## Link findings",
            "## Authentication results",
            "## Analyst chart",
            "## Debate summary",
            "## Policy matches",
            "## Similar past cases",
            "## Analyst notes",
            "## Approval status",
            "## Timestamps",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, report)
        self.assertIn("Never click links with raw IP addresses.", report)
        self.assertIn("SPF: fail", report)
        self.assertIn("APPROVED", report)
        self.assertIn("Latest Classifier point", report)
        self.assertIn(
            "![Risk score chart](./20261120T045724208818Z-82539024_chart.png)",
            report,
        )
        self.assertNotIn("https://", report)

    def test_report_masks_personal_data_from_summaries_and_notes(self) -> None:
        case = {
            "case_id": "20261120T045724208818Z-82539024",
            "email_text": "From: Alice <alice@example.test>",
            "current_verdict": (
                "Verdict: SUSPICIOUS\n"
                "Confidence: 70%\n"
                "Sender domain findings: example.test."
            ),
            "pipeline_summaries": [
                "sender",
                "Contact bob@private.example at 415-555-0123.",
            ],
            "analyst_notes": ["Call bob@private.example at 415-555-0123."],
            "status": "OPEN",
        }

        report = build_report(case)

        self.assertIn("example.test", report)
        self.assertIn("[EMAIL_REDACTED]", report)
        self.assertIn("[PHONE_REDACTED]", report)
        self.assertNotIn("alice@", report)
        self.assertNotIn("bob@", report)
        self.assertNotIn("415-555-0123", report)


if __name__ == "__main__":
    unittest.main()
