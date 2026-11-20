"""API-free tests for case persistence, trimming, and PII masking."""

import unittest
from tempfile import TemporaryDirectory

from src.case_store import (
    create_case_id,
    list_cases,
    load_case,
    save_case,
)
from src.intake import trim_email_text


def make_case(verdict: str = "PHISHING") -> dict[str, object]:
    """Create a complete case record for the storage tests."""
    return {
        "case_id": create_case_id(),
        "source_type": "text",
        "email_text": "From: sender@example.test\n\nBody",
        "pipeline_summaries": ["Verdict: " + verdict],
        "debate_history": [],
        "current_verdict": f"Verdict: {verdict}\nConfidence: 90%",
        "status": "OPEN",
        "analyst_notes": [],
        "rejection_reasons": [],
    }


class CaseStoreTests(unittest.TestCase):
    """Verify persisted case files and defensive data transforms."""

    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            case = make_case()
            save_case(case, temporary_directory, mask_pii=False)

            loaded = load_case(case["case_id"], temporary_directory)

        self.assertEqual(loaded["case_id"], case["case_id"])
        self.assertEqual(loaded["current_verdict"], case["current_verdict"])
        self.assertIn("created_at", loaded)
        self.assertIn("updated_at", loaded)

    def test_list_cases_returns_id_status_verdict_and_date(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            first = make_case("PHISHING")
            second = make_case("LEGIT")
            second["status"] = "APPROVED"
            save_case(first, temporary_directory, mask_pii=False)
            save_case(second, temporary_directory, mask_pii=False)

            listed = list_cases(temporary_directory)

        self.assertEqual({case["case_id"] for case in listed}, {
            first["case_id"],
            second["case_id"],
        })
        approved = next(case for case in listed if case["case_id"] == second["case_id"])
        self.assertEqual(approved["status"], "APPROVED")
        self.assertEqual(approved["verdict"], "LEGIT")
        self.assertTrue(approved["date"])

    def test_saved_case_masks_non_sender_emails_and_phone_numbers(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            case = make_case()
            case["email_text"] = (
                "From: Alice <alice@example.test>\n\n"
                "Contact bob@private.example or 415-555-0123."
            )
            case["analyst_notes"] = ["Call Alice at 415-555-0123."]
            save_case(case, temporary_directory)

            loaded = load_case(case["case_id"], temporary_directory)

        self.assertIn("[SENDER]@example.test", loaded["email_text"])
        self.assertIn("[EMAIL_REDACTED]", loaded["email_text"])
        self.assertIn("[PHONE_REDACTED]", loaded["email_text"])
        self.assertIn("[PHONE_REDACTED]", loaded["analyst_notes"][0])
        self.assertNotIn("alice@", loaded["email_text"])
        self.assertNotIn("bob@", loaded["email_text"])

    def test_trim_preserves_headers_body_edges_and_all_links(self) -> None:
        email_text = (
            "From: sender@example.test\nSubject: Keep this\n\n"
            + "FIRSTPART " * 120
            + " https://one.example/path "
            + "MIDDLEPART " * 400
            + " https://two.example/path "
            + "LASTPART " * 120
        )

        trimmed, did_trim = trim_email_text(email_text, limit=500)

        self.assertTrue(did_trim)
        self.assertIn("From: sender@example.test", trimmed)
        self.assertIn("Subject: Keep this", trimmed)
        self.assertIn("FIRSTPART", trimmed)
        self.assertIn("LASTPART", trimmed)
        self.assertIn("https://one.example/path", trimmed)
        self.assertIn("https://two.example/path", trimmed)
        self.assertNotIn("MIDDLEPART " * 100, trimmed)

    def test_short_email_is_not_trimmed(self) -> None:
        email_text = "From: sender@example.test\n\nHello"

        trimmed, did_trim = trim_email_text(email_text)

        self.assertEqual(trimmed, email_text)
        self.assertFalse(did_trim)


if __name__ == "__main__":
    unittest.main()
