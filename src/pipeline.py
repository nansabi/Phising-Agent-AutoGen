"""Run extraction, risk checks, and classification as three sequential chats."""

import json
import re
from typing import Any, Callable

from autogen import ConversableAgent, UserProxyAgent, initiate_chats
from autogen.agentchat.chat import ChatResult

from src.agents import build_phase2_agents
from src.tools import extract_links

STEP_NAMES = ("Extractor", "Link_Checker", "Classifier")

FACTS_SUMMARY_PROMPT = """Create a clean list of facts from the Extractor's
tool results only. Do not assess risk or give a verdict. Return exactly these
fields on separate lines: Sender, Headers, Links, and Keywords. Links must be a
valid JSON array of strings, or [] when there are no links. Do not add any
other fields."""

LINKS_SUMMARY_PROMPT = """Summarize only the Link_Checker's rule-based results.
Include the sender domain and its risk score, each link with its risk score
and reasons, SPF/DKIM/DMARC results, and link-domain comparison findings. If
there are no links, say so. Do not give an overall verdict."""

CLASSIFIER_SUMMARY_PROMPT = """Return the Classifier's final response with its
verdict, confidence percentage, the Sender domain findings line, and exactly
three short reasons. Preserve its verdict and evidence; omit only the final
TERMINATE marker."""


def build_chat_queue(
    email_text: str,
    extractor: ConversableAgent,
    link_checker: ConversableAgent,
    classifier: ConversableAgent,
    tool_executor: UserProxyAgent,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Build the three ordered chats; AutoGen carries summaries forward."""
    def extractor_message(
        sender: ConversableAgent,
        recipient: ConversableAgent,
        context: dict[str, Any],
    ) -> str:
        del sender, recipient, context
        if progress_callback is not None:
            progress_callback("Extract")
        return (
            "Extract facts only. Use the extraction tools on this email:\n\n"
            f"{email_text}"
        )

    def checker_message(
        sender: ConversableAgent,
        recipient: ConversableAgent,
        context: dict[str, Any],
    ) -> str:
        if progress_callback is not None:
            progress_callback("Check")
        return link_checker_message(sender, recipient, context)

    def classifier_message(
        sender: ConversableAgent,
        recipient: ConversableAgent,
        context: dict[str, Any],
    ) -> str:
        del sender, recipient, context
        if progress_callback is not None:
            progress_callback("Classify")
        return (
            "Use both automatically carried-over summaries to classify "
            "the email. Give a verdict, confidence, a Sender domain "
            "findings line, and exactly three short reasons."
        )

    return [
        {
            "sender": tool_executor,
            "recipient": extractor,
            "message": extractor_message,
            "max_turns": 8,
            "summary_method": "reflection_with_llm",
            "summary_args": {"summary_prompt": FACTS_SUMMARY_PROMPT},
            "silent": True,
        },
        {
            "sender": tool_executor,
            "recipient": link_checker,
            "message": checker_message,
            "max_turns": 8,
            "summary_method": "last_msg",
            "summary_args": {"summary_prompt": LINKS_SUMMARY_PROMPT},
            "silent": True,
        },
        {
            "sender": tool_executor,
            "recipient": classifier,
            "message": classifier_message,
            "max_turns": 2,
            "summary_method": "last_msg",
            "summary_args": {"summary_prompt": CLASSIFIER_SUMMARY_PROMPT},
            "silent": True,
        },
    ]


def link_checker_message(
    sender: ConversableAgent,
    recipient: ConversableAgent,
    context: dict[str, Any],
) -> str:
    """Tell Link_Checker whether the Extractor found links to score."""
    del sender, recipient
    carryover = context.get("carryover", [])
    summaries = (
        "\n".join(str(item) for item in carryover)
        if isinstance(carryover, list)
        else str(carryover)
    )
    if _links_from_summary(summaries):
        instructions = (
            "Use the Extractor facts in the carried-over summary. Call "
            "check_link once for every listed link, and call "
            "check_sender_domain, check_auth_results, and "
            "compare_link_domains as directed by your system message."
        )
    else:
        instructions = (
            "The Extractor facts contain no links. Do not call check_link. Still "
            "call check_sender_domain and check_auth_results from the extracted "
            "headers, and call compare_link_domains with an empty link list."
        )
    return f"{instructions}\n\nExtractor facts:\n{summaries}"


def run_pipeline(
    email_text: str,
    offline: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> list[ChatResult]:
    """Run the three chats in order and return their AutoGen results."""
    extractor, link_checker, classifier, tool_executor = build_phase2_agents(
        offline=offline
    )
    queue = build_chat_queue(
        email_text,
        extractor,
        link_checker,
        classifier,
        tool_executor,
        progress_callback=progress_callback,
    )
    return initiate_chats(queue)


def _links_from_summary(summary: str) -> list[str]:
    """Read the Extractor's JSON Links field, with text extraction as fallback."""
    match = re.search(r"(?im)^\s*Links\s*:\s*(.*?)\s*$", summary)
    if not match:
        return extract_links(summary)

    raw_links = match.group(1).strip().strip("`")
    try:
        parsed_links = json.loads(raw_links)
    except json.JSONDecodeError:
        if raw_links.casefold() in {"", "none", "no links", "[]"}:
            return []
        return extract_links(raw_links)

    if isinstance(parsed_links, list):
        return [link for link in parsed_links if isinstance(link, str)]
    return extract_links(raw_links)
