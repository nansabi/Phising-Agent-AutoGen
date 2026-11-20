"""API-free tests for local security-policy and past-case retrieval."""

import unittest

from src.rag import LocalPolicyRetriever, PolicyChecker


class RagTests(unittest.TestCase):
    """Verify retrieval ranks relevant policy evidence without an LLM."""

    def test_raw_ip_email_retrieves_the_raw_ip_policy(self) -> None:
        result = LocalPolicyRetriever().retrieve(
            "From: sender@example.test\n\n"
            "Please verify your account: http://192.0.2.44/login"
        )

        self.assertTrue(result.policy_matches)
        self.assertEqual(result.policy_matches[0].rule_id, "R1")
        self.assertIn("numeric IP address", result.policy_matches[0].why_matches)
        rule_ids = {match.rule_id for match in result.policy_matches}
        self.assertIn("R3", rule_ids)
        self.assertNotIn("R4", rule_ids)
        self.assertNotIn("R6", rule_ids)
        self.assertNotIn("R8", rule_ids)
        self.assertEqual(
            [case.case_id for case in result.similar_past_cases],
            ["CASE-01"],
        )

    def test_policy_checker_uses_a_local_auto_gen_reply(self) -> None:
        result = PolicyChecker().check(
            "Please verify your account at http://192.0.2.44/login"
        )

        self.assertTrue(result.policy_matches)
        self.assertEqual(result.policy_matches[0].rule_id, "R1")

    def test_r7_is_informational_when_authentication_is_unavailable(self) -> None:
        result = LocalPolicyRetriever().retrieve(
            "From: community@community.example\n"
            "Subject: Community newsletter\n"
            "Read the schedule at "
            "https://events.community.example/newsletter",
            pipeline_summaries=[
                "Authentication results:\n"
                "SPF: not available\n"
                "DKIM: not available\n"
                "DMARC: not available"
            ],
        )

        r7 = next(
            match for match in result.policy_matches
            if match.rule_id == "R7"
        )
        self.assertIn("Informational only", r7.why_matches)
        self.assertIn("not a risk indicator", r7.why_matches)


if __name__ == "__main__":
    unittest.main()
