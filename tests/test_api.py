"""API tests using local fakes so no provider key or Docker is required."""

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import api.main as api
from src.case_store import create_case_id, save_case


def _case_record(
    email_text: str,
    source_type: str,
    _result: object,
    case_id: str | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id or create_case_id(),
        "source_type": source_type,
        "email_text": email_text,
        "pipeline_summaries": [
            (
                "Sender: sender@example.test\n"
                'Headers: {"From": "sender@example.test"}\n'
                'Links: ["http://192.0.2.44/path"]\n'
                "Keywords: verify"
            ),
            (
                "Sender domain: example.test\n"
                "Sender domain risk score: 0\n"
                "Sender domain reasons: No configured risks\n"
                "Links:\n"
                "- Link: http://192.0.2.44/path\n"
                "  Risk Score: 60\n"
                "  Reasons: raw IP; no HTTPS\n"
                "Authentication results:\n"
                "- SPF: fail\n"
                "- DKIM: pass\n"
                "- DMARC: not available\n"
            ),
            (
                "Verdict: PHISHING\n"
                "Confidence: 90%\n"
                "Sender domain findings: example.test\n"
            ),
        ],
        "pipeline_costs": [],
        "debate_history": [
            {
                "attempt": 1,
                "human_challenge": "",
                "messages": [
                    {
                        "name": "Debate_Admin",
                        "content": "Opening analysis",
                    },
                    {
                        "name": "Classifier",
                        "content": "PHISHING because http://192.0.2.44/path is raw IP evidence.",
                    },
                    {"name": "Skeptic", "content": "AGREE"},
                ],
            }
        ],
        "debate_costs": [],
        "current_verdict": (
            "Verdict: PHISHING\n"
            "Confidence: 90%\n"
            "Sender domain findings: example.test"
        ),
        "debate_rounds_used": 1,
        "skeptic_status": "AGREED",
        "total_cost": {
            "usage_including_cached_inference": {"total_cost": 0.0},
            "usage_excluding_cached_inference": {"total_cost": 0.0},
        },
        "status": "OPEN",
        "analyst_notes": [],
        "rejection_reasons": [],
    }


class ApiTests(unittest.TestCase):
    """Cover the public routes and background-job behavior without an LLM."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.patches = [
            patch.object(api, "CASES_DIR", root / "cases"),
            patch.object(api, "INPUTS_DIR", root / "inputs"),
            patch.object(api, "REPORTS_DIR", root / "reports"),
            patch.object(api, "LOG_DB_PATH", root / "logs" / "api.db"),
        ]
        for item in self.patches:
            item.start()
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        self.client.close()
        for item in reversed(self.patches):
            item.stop()
        self.temporary_directory.cleanup()

    def test_health_returns_only_safe_configuration_booleans(self) -> None:
        with (
            patch.object(api, "GROQ_API_KEY", "sensitive-test-sentinel"),
            patch.object(api, "docker_is_available", return_value=False),
        ):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "groq_key_configured": True,
                "docker_available": False,
            },
        )
        self.assertNotIn("sensitive-test-sentinel", response.text)

    def test_analyze_poll_list_defang_and_approve_without_keyboard(self) -> None:
        def fake_analysis(
            email_text: str,
            offline: bool,
            progress_callback: object,
        ) -> object:
            self.assertTrue(offline)
            for step in ("Extract", "Check", "Classify", "Debate"):
                progress_callback(step)  # type: ignore[operator]
            return object()

        email_text = (
            "From: sender@example.test\n"
            "Subject: Verify account\n\n"
            "Visit http://192.0.2.44/path"
        )
        with (
            patch(
                "api.main.run_desk_chief",
                side_effect=fake_analysis,
            ) as run_analysis,
            patch("api.main.build_case_record", side_effect=_case_record),
            patch(
                "api.main._retrieve_policy",
                side_effect=lambda _checker, case: case.update(
                    policy_matches=[{"rule_id": "R1", "rule_text": "Raw IP rule"}],
                    similar_past_cases=[],
                ),
            ),
            patch("api.main.run_analyst_chart", return_value=None),
        ):
            response = self.client.post(
                "/api/analyze",
                data={
                    "text": email_text,
                    "offline": "true",
                    "no_docker": "true",
                },
            )
            self.assertEqual(response.status_code, 202)
            accepted = response.json()
            job = self._wait_for_job(accepted["job_id"])
            self.assertEqual(job["status"], "done")
            self.assertEqual(job["current_step"], "Awaiting approval")

            detail = self.client.get(
                f"/api/cases/{accepted['case_id']}"
            ).json()
            self.assertEqual(detail["status"], "OPEN")
            self.assertIn("[SENDER]@example.test", detail["email_text"])
            self.assertIn("hxxp://192[.]0[.]2[.]44/path", detail["email_text"])
            self.assertEqual(
                detail["link_findings"][0]["display_url"],
                "hxxp://192[.]0[.]2[.]44/path",
            )
            self.assertEqual(detail["link_findings"][0]["risk_score"], 60)
            self.assertEqual(detail["auth_results"][0], "SPF: fail")
            self.assertEqual(detail["debate_history"][0]["speaker"], "Classifier")
            self.assertEqual(detail["debate_history"][1]["speaker"], "Skeptic")

            listed = self.client.get("/api/cases").json()
            self.assertEqual(listed[0]["id"], accepted["case_id"])
            self.assertEqual(listed[0]["verdict"], "PHISHING")
            self.assertEqual(listed[0]["confidence"], "90%")

            approval = self.client.post(
                f"/api/cases/{accepted['case_id']}/approve"
            )
            self.assertEqual(approval.status_code, 200)
            self.assertEqual(approval.json()["status"], "APPROVED")
            self.assertEqual(run_analysis.call_count, 1)
            report = self.client.get(
                f"/api/cases/{accepted['case_id']}/report"
            )
            self.assertEqual(report.status_code, 200)
            self.assertIn("## Verdict", report.text)

    def test_upload_rejects_unknown_file_types(self) -> None:
        response = self.client.post(
            "/api/analyze",
            files={"file": ("message.exe", b"not an email")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertIn("Upload must be", response.json()["detail"])

    def test_rejection_limit_runs_debate_once_then_requires_review(self) -> None:
        case_id = create_case_id()
        case = _case_record(
            "From: sender@example.test\n\nhttp://192.0.2.44/path",
            "text",
            object(),
        )
        case["case_id"] = case_id
        case["offline"] = True
        case["no_docker"] = True
        case["rejection_reasons"] = ["First challenge"]
        save_case(case, api.CASES_DIR)

        def fake_rerun(
            current_case: dict[str, object],
            reason: str,
            **_kwargs: object,
        ) -> dict[str, object]:
            current_case["current_verdict"] = (
                f"Verdict: SUSPICIOUS\nConfidence: 75%\nChallenge: {reason}"
            )
            return current_case

        with (
            patch("api.main.rerun_saved_case_debate", side_effect=fake_rerun) as rerun,
            patch(
                "api.main._retrieve_policy",
                side_effect=lambda _checker, current_case: current_case,
            ),
            patch("api.main.run_analyst_chart", return_value=None),
        ):
            accepted = self.client.post(
                f"/api/cases/{case_id}/reject",
                json={"reason": "The authentication result was overlooked."},
            )
            self.assertEqual(accepted.status_code, 202)
            job = self._wait_for_job(accepted.json()["job_id"])
            self.assertEqual(job["status"], "done")
            saved_case = self.client.get(f"/api/cases/{case_id}").json()
            self.assertEqual(saved_case["status"], "NEEDS_REVIEW")
            rerun.assert_called_once()

            too_many = self.client.post(
                f"/api/cases/{case_id}/reject",
                json={"reason": "A third challenge."},
            )
            self.assertEqual(too_many.status_code, 409)

    def _wait_for_job(self, job_id: str) -> dict[str, object]:
        for _ in range(500):
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            job = response.json()
            if job["status"] in {"done", "failed"}:
                self.assertEqual(job["status"], "done", job.get("error_text"))
                return job
            time.sleep(0.01)
        self.fail(f"Job {job_id} did not finish in time.")


if __name__ == "__main__":
    unittest.main()
