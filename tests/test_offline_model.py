"""API-free tests for the rule-based AutoGen model client."""

import unittest

from src.offline_model import RuleBasedModelClient, offline_llm_config


class OfflineModelTests(unittest.TestCase):
    """Check that local model responses match the agent prompt format."""

    def test_classifier_returns_the_required_verdict_fields(self) -> None:
        client = RuleBasedModelClient(offline_llm_config()["config_list"][0])
        response = client.create(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Classifier. Give one verdict.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "From: Security <security@example.test>\n"
                            "Subject: Verify account\n\n"
                            "Please verify your account immediately: "
                            "http://192.0.2.44/verify"
                        ),
                    },
                ]
            }
        )

        text = client.message_retrieval(response)[0]

        self.assertIn("Verdict: PHISHING", text)
        self.assertRegex(text, r"(?m)^Confidence: \d+%$")
        self.assertIn("Sender domain findings:", text)
        self.assertEqual(len([line for line in text.splitlines() if line.startswith("- ")]), 3)
        self.assertTrue(text.endswith("TERMINATE"))

    def test_clean_email_with_unavailable_authentication_is_legit(self) -> None:
        text = self._classify(
            "From: community@community.example\n"
            "Subject: Community newsletter\n\n"
            "This harmless sample shares a monthly community update. "
            "Read the schedule at "
            "https://events.community.example/newsletter\n"
            "SPF: not available\nDKIM: not available\nDMARC: not available"
        )

        self.assertIn("Verdict: LEGIT", text)
        self.assertIn("informational only", text.casefold())

    def test_phish2_stays_suspicious_or_phishing_offline(self) -> None:
        text = self._classify(
            "From: account-team@example.test\n"
            "Subject: Synthetic link review\n\n"
            "This sample contains the word urgent and a lookalike hostname: "
            "https://micros0ft.example/urgent"
        )

        self.assertRegex(text, r"Verdict: (SUSPICIOUS|PHISHING)")

    def test_skeptic_always_returns_exact_agreement_marker(self) -> None:
        client = RuleBasedModelClient(offline_llm_config()["config_list"][0])
        response = client.create(
            {"messages": [{"role": "system", "content": "You are Skeptic."}]}
        )

        self.assertEqual(client.message_retrieval(response), ["AGREE"])
        self.assertEqual(client.cost(response), 0.0)

    def _classify(self, email_text: str) -> str:
        client = RuleBasedModelClient(offline_llm_config()["config_list"][0])
        response = client.create(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Classifier. Give one verdict.",
                    },
                    {"role": "user", "content": email_text},
                ]
            }
        )
        return client.message_retrieval(response)[0]


if __name__ == "__main__":
    unittest.main()
