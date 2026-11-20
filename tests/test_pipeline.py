"""API-free tests for Phase 2 queue ordering, carryover, and empty links."""

import unittest
from unittest.mock import patch

from autogen.agentchat.chat import ChatResult

from src.pipeline import build_chat_queue, run_pipeline


class StubAgent:
    """Provide the attributes AutoGen needs when validating a chat queue."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.client = object()


class StubExecutor(StubAgent):
    """Return fixed chat results while recording AutoGen's carryovers."""

    def __init__(self, results: list[ChatResult]) -> None:
        super().__init__("ToolExecutor")
        self.results = iter(results)
        self.calls: list[dict[str, object]] = []

    def initiate_chat(self, **kwargs: object) -> ChatResult:
        self.calls.append(kwargs)
        return next(self.results)


class PipelineTests(unittest.TestCase):
    """Exercise Phase 2 sequencing without contacting an LLM provider."""

    def setUp(self) -> None:
        self.extractor = StubAgent("Extractor")
        self.link_checker = StubAgent("Link_Checker")
        self.classifier = StubAgent("Classifier")

    def test_empty_links_skip_link_check_tool_instructions(self) -> None:
        executor = StubExecutor([])
        queue = build_chat_queue(
            "From: sender@example.test",
            self.extractor,
            self.link_checker,
            self.classifier,
            executor,
        )
        message = queue[1]["message"]
        instructions = message(
            executor,
            self.link_checker,
            {"carryover": ["Sender: sender@example.test\nLinks: []"]},
        )

        self.assertIn("Do not call check_link", instructions)
        self.assertIn("compare_link_domains with an empty link list", instructions)
        self.assertIn("Links: []", instructions)

    def test_chat_messages_report_each_analysis_step(self) -> None:
        executor = StubExecutor([])
        steps: list[str] = []
        queue = build_chat_queue(
            "From: sender@example.test",
            self.extractor,
            self.link_checker,
            self.classifier,
            executor,
            progress_callback=steps.append,
        )

        queue[0]["message"](executor, self.extractor, {})
        queue[1]["message"](
            executor,
            self.link_checker,
            {"carryover": ["Sender: sender@example.test\nLinks: []"]},
        )
        queue[2]["message"](executor, self.classifier, {})

        self.assertEqual(steps, ["Extract", "Check", "Classify"])

    def test_initiate_chats_passes_each_prior_summary_forward(self) -> None:
        summaries = [
            "Sender: sender@example.test\nLinks: [\"https://example.test/a\"]",
            "Link risk: 0/100",
            "LEGIT, 90%",
        ]
        fake_results = [
            ChatResult(chat_id=index, chat_history=[], summary=summary, cost={})
            for index, summary in enumerate(summaries)
        ]
        executor = StubExecutor(fake_results)
        with patch(
            "src.pipeline.build_phase2_agents",
            return_value=(
                self.extractor,
                self.link_checker,
                self.classifier,
                executor,
            ),
        ):
            results = run_pipeline("From: sender@example.test\n\nHello")

        self.assertEqual(len(results), 3)
        self.assertEqual(executor.calls[0]["carryover"], [])
        self.assertEqual(executor.calls[1]["carryover"], [summaries[0]])
        self.assertEqual(executor.calls[2]["carryover"], summaries[:2])

        link_message = executor.calls[1]["message"]
        instructions = link_message(
            executor,
            self.link_checker,
            {"carryover": executor.calls[1]["carryover"]},
        )
        self.assertIn("check_link once for every listed link", instructions)
        self.assertIn("https://example.test/a", instructions)


if __name__ == "__main__":
    unittest.main()
