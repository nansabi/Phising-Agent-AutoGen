"""API-free tests for the Phase 3 custom speaker selection rules."""

import unittest
from contextlib import redirect_stdout
from io import StringIO

from autogen.agentchat.chat import ChatResult

from src.agents import (
    CLASSIFIER_SYSTEM_MESSAGE,
    DEBATE_CLASSIFIER_SYSTEM_MESSAGE,
    MISSING_AUTHENTICATION_RULE,
    PHASE2_CLASSIFIER_SYSTEM_MESSAGE,
    RESERVED_DOMAIN_BEHAVIOR_RULE,
    SKEPTIC_SYSTEM_MESSAGE,
)
from src.debate import (
    build_starting_message,
    normalize_text,
    select_debate_speaker,
)


class FakeAgent:
    """Represent a named participant without constructing an LLM agent."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeGroupChat:
    """Provide only the GroupChat attributes used by the selector."""

    def __init__(self, messages: list[dict[str, str]]) -> None:
        self.messages = messages
        self.max_round = 8
        self.agents = [
            FakeAgent("Debate_Admin"),
            FakeAgent("Classifier"),
            FakeAgent("Skeptic"),
        ]

    def agent_by_name(self, name: str) -> FakeAgent | None:
        return next((agent for agent in self.agents if agent.name == name), None)


class DebateSpeakerSelectionTests(unittest.TestCase):
    """Check all speaker transitions without calling an API."""

    def test_normalize_text_replaces_unicode_punctuation(self) -> None:
        text = (
            "a\u2011b c\u2013d e\u2014f "
            "\u2018g\u2019 \u201ch\u201d\u202f!"
        )

        self.assertEqual(normalize_text(text), "a-b c-d e-f 'g' \"h\" !")

    def test_agree_detection_works_after_normalization(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Skeptic", "content": "AGREE\u202f"}]
        )

        selected = self.select_silently(FakeAgent("Skeptic"), groupchat)

        self.assertIsNone(selected)

    def test_debate_prompts_include_reserved_domain_rule(self) -> None:
        for system_message in (
            CLASSIFIER_SYSTEM_MESSAGE,
            PHASE2_CLASSIFIER_SYSTEM_MESSAGE,
            DEBATE_CLASSIFIER_SYSTEM_MESSAGE,
            SKEPTIC_SYSTEM_MESSAGE,
        ):
            self.assertIn(RESERVED_DOMAIN_BEHAVIOR_RULE, system_message)

    def test_classifier_and_skeptic_prompts_reject_missing_auth_as_evidence(self) -> None:
        for system_message in (
            CLASSIFIER_SYSTEM_MESSAGE,
            PHASE2_CLASSIFIER_SYSTEM_MESSAGE,
            DEBATE_CLASSIFIER_SYSTEM_MESSAGE,
            SKEPTIC_SYSTEM_MESSAGE,
        ):
            self.assertIn("NOT evidence of phishing", system_message)
            self.assertIn(MISSING_AUTHENTICATION_RULE, system_message)

    def test_starting_message_contains_all_phase2_summaries(self) -> None:
        results = [
            ChatResult(summary="Sender: sender@example.test"),
            ChatResult(summary="Domain risk: 0/100"),
            ChatResult(summary="SUSPICIOUS, 75%"),
        ]

        message = build_starting_message(results)

        self.assertIn("Sender: sender@example.test", message)
        self.assertIn("Domain risk: 0/100", message)
        self.assertIn("SUSPICIOUS, 75%", message)
        self.assertIn(
            "Reserved or example domains in this sample are placeholders for "
            "real ones. Judge behavior only.",
            message,
        )

    def select_silently(
        self,
        last_speaker: FakeAgent,
        groupchat: FakeGroupChat,
    ) -> FakeAgent | None:
        with redirect_stdout(StringIO()):
            return select_debate_speaker(last_speaker, groupchat)

    def test_admin_selects_classifier(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Debate_Admin", "content": "Starting evidence"}]
        )

        selected = self.select_silently(FakeAgent("Debate_Admin"), groupchat)

        self.assertEqual(selected.name, "Classifier")

    def test_classifier_selects_skeptic(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Classifier", "content": "PHISHING, 80%"}]
        )

        selected = self.select_silently(FakeAgent("Classifier"), groupchat)

        self.assertEqual(selected.name, "Skeptic")

    def test_skeptic_challenge_selects_classifier(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Skeptic", "content": "CHALLENGE: evidence is limited"}]
        )

        selected = self.select_silently(FakeAgent("Skeptic"), groupchat)

        self.assertEqual(selected.name, "Classifier")

    def test_skeptic_agreement_ends_chat(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Skeptic", "content": "The evidence is sufficient.\nAGREE"}]
        )

        selected = self.select_silently(FakeAgent("Skeptic"), groupchat)

        self.assertIsNone(selected)

    def test_round_limit_ends_chat(self) -> None:
        groupchat = FakeGroupChat(
            [{"name": "Classifier", "content": "Still reviewing"}] * 8
        )

        selected = self.select_silently(FakeAgent("Classifier"), groupchat)

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
