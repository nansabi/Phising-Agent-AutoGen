"""Run a constrained Classifier-Skeptic debate after Phase 2 analysis."""

import unicodedata
from dataclasses import dataclass
from typing import Any

from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.agentchat.chat import ChatResult
from autogen.agentchat.utils import gather_usage_summary

from src.agents import (
    DEBATE_CLASSIFIER_SYSTEM_MESSAGE,
    SKEPTIC_SYSTEM_MESSAGE,
    build_phase2_classifier,
)
from src.config import llm_config
from src.offline_model import RuleBasedModelClient, offline_llm_config
from src.pipeline import run_pipeline

GROUP_CHAT_MAX_ROUND = 8
_TEXT_TRANSLATION = str.maketrans(
    {
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u202f": " ",
        "\u2022": "-",
    }
)


@dataclass
class DebateResult:
    """Collect Phase 2 and debate results for display by the CLI."""

    phase2_results: list[ChatResult]
    chat_result: ChatResult
    messages: list[dict[str, Any]]
    rounds_used: int
    skeptic_agreed: bool
    round_limit_hit: bool
    final_classifier_message: str
    skeptic_last_message: str
    cost: dict[str, Any]


def build_starting_message(
    phase2_results: list[ChatResult],
    pipeline_carryover: str | None = None,
    human_challenge: str | None = None,
) -> str:
    """Combine Phase 2 summaries or their nested-chat carryover."""
    if len(phase2_results) != 3:
        raise RuntimeError(
            f"Expected 3 Phase 2 results, received {len(phase2_results)}."
        )

    evidence = pipeline_carryover or "\n\n".join(
        (
            f"Phase 2 Extractor facts:\n{phase2_results[0].summary or ''}",
            f"Phase 2 link and sender findings:\n{phase2_results[1].summary or ''}",
            f"Phase 2 first verdict:\n{phase2_results[2].summary or ''}",
        )
    )
    message = (
        "Review the Phase 2 analysis below. The Classifier should reassess the "
        "initial verdict, then the Skeptic should challenge or agree.\n\n"
        "Reserved or example domains in this sample are placeholders for real "
        "ones. Judge behavior only.\n\n"
        f"Phase 2 tool results and verdict:\n{evidence}"
    )
    if human_challenge:
        message += (
            "\n\nHuman reviewer challenge (assess against the Phase 2 tool "
            "results):\n"
            f"{normalize_text(human_challenge)}"
        )
    return message


def select_debate_speaker(
    last_speaker: Agent,
    groupchat: GroupChat,
) -> Agent | None:
    """Choose the next constrained speaker or end after agreement/round limit."""
    if len(groupchat.messages) >= groupchat.max_round:
        return None

    if last_speaker.name == "Debate_Admin":
        next_name = "Classifier"
    elif last_speaker.name == "Classifier":
        next_name = "Skeptic"
    elif last_speaker.name == "Skeptic":
        last_message = groupchat.messages[-1] if groupchat.messages else {}
        status = _skeptic_response_status(str(last_message.get("content") or ""))
        if status == "AGREE":
            return None
        if status == "CHALLENGE":
            next_name = "Classifier"
        else:
            return None
    else:
        return None

    next_speaker = groupchat.agent_by_name(next_name)
    if next_speaker is None:
        raise RuntimeError(f"Required debate speaker '{next_name}' was not found.")

    round_number = max(1, (len(groupchat.messages) - 1) // 2 + 1)
    print(f"Round {round_number}: {next_speaker.name}", flush=True)
    return next_speaker


def run_phase3(email_text: str, offline: bool = False) -> DebateResult:
    """Run Phase 2, then debate its first verdict in an AutoGen GroupChat."""
    return run_debate(run_pipeline(email_text, offline=offline), offline=offline)


def run_debate(
    phase2_results: list[ChatResult],
    pipeline_carryover: str | None = None,
    human_challenge: str | None = None,
    resume_messages: list[dict[str, Any]] | None = None,
    offline: bool = False,
) -> DebateResult:
    """Run the Phase 3 debate from existing Phase 2 results."""
    if len(phase2_results) != 3:
        raise RuntimeError(
            f"Expected 3 Phase 2 results, received {len(phase2_results)}."
        )

    classifier = build_phase2_classifier(offline=offline)
    classifier.update_system_message(DEBATE_CLASSIFIER_SYSTEM_MESSAGE)
    classifier.update_max_consecutive_auto_reply(GROUP_CHAT_MAX_ROUND)
    skeptic = ConversableAgent(
        name="Skeptic",
        system_message=SKEPTIC_SYSTEM_MESSAGE,
        llm_config=offline_llm_config() if offline else dict(llm_config),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=GROUP_CHAT_MAX_ROUND,
    )
    if offline:
        skeptic.register_model_client(RuleBasedModelClient)
    debate_admin = UserProxyAgent(
        name="Debate_Admin",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    groupchat = GroupChat(
        agents=[debate_admin, classifier, skeptic],
        messages=[],
        max_round=(
            len(resume_messages) + GROUP_CHAT_MAX_ROUND
            if resume_messages
            else GROUP_CHAT_MAX_ROUND
        ),
        speaker_selection_method=select_debate_speaker,
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        name="Debate_Manager",
        llm_config=False,
        silent=True,
    )
    if resume_messages:
        last_speaker, last_message = manager.resume(
            resume_messages,
            silent=True,
        )
        if last_speaker not in groupchat.agents:
            raise RuntimeError(
                "The saved debate history does not end with a group-chat speaker."
            )
        groupchat.append(last_message, last_speaker)
        for agent in groupchat.agents:
            if agent != last_speaker:
                last_speaker.send(
                    last_message,
                    agent,
                    request_reply=False,
                    silent=True,
                )
        challenge_message = (
            "Human reviewer challenge (assess against the saved Phase 2 tool "
            "results):\n"
            f"{normalize_text(human_challenge or 'Reassess the current verdict.')}"
        )
        chat_result = debate_admin.initiate_chat(
            manager,
            message=challenge_message,
            clear_history=False,
            max_turns=1,
            summary_method="last_msg",
            silent=True,
        )
    else:
        starting_message = build_starting_message(
            phase2_results,
            pipeline_carryover,
            human_challenge,
        )
        chat_result = debate_admin.initiate_chat(
            manager,
            message=starting_message,
            summary_method="last_msg",
            silent=True,
        )

    messages = [
        _normalize_message(message) for message in groupchat.messages
    ]
    skeptic_last_message = _last_message_from(messages, "Skeptic")
    final_classifier_message = _last_message_from(messages, "Classifier")
    if not final_classifier_message:
        raise RuntimeError("The debate ended without a Classifier response.")

    skeptic_agreed = _skeptic_response_status(skeptic_last_message) == "AGREE"
    round_limit_hit = (
        not skeptic_agreed and len(messages) >= groupchat.max_round
    )
    speaker_messages = [
        message
        for message in messages
        if message.get("name") in {"Classifier", "Skeptic"}
    ]
    rounds_used = (len(speaker_messages) + 1) // 2

    return DebateResult(
        phase2_results=phase2_results,
        chat_result=chat_result,
        messages=messages,
        rounds_used=rounds_used,
        skeptic_agreed=skeptic_agreed,
        round_limit_hit=round_limit_hit,
        final_classifier_message=final_classifier_message,
        skeptic_last_message=skeptic_last_message,
        cost=gather_usage_summary([classifier, skeptic]),
    )


def _last_message_from(messages: list[dict[str, Any]], speaker_name: str) -> str:
    """Return the most recent content from one named participant."""
    for message in reversed(messages):
        if message.get("name") == speaker_name:
            return normalize_text(str(message.get("content") or "")).strip()
    return ""


def normalize_text(text: str) -> str:
    """Replace typographic punctuation and spacing with plain ASCII."""
    translated = text.translate(_TEXT_TRANSLATION)
    return unicodedata.normalize("NFKD", translated).encode(
        "ascii", "ignore"
    ).decode("ascii")


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a chat message with normalized text content."""
    normalized = dict(message)
    content = normalized.get("content")
    if isinstance(content, str):
        normalized["content"] = normalize_text(content)
    return normalized


def _skeptic_response_status(message: str) -> str | None:
    """Recognize only the Skeptic's required normalized one-line responses."""
    normalized = normalize_text(message).strip()
    if normalized == "AGREE":
        return "AGREE"
    if normalized.startswith("CHALLENGE: ") and "\n" not in normalized:
        reason = normalized[len("CHALLENGE: ") :].strip()
        if reason:
            return "CHALLENGE"
    return None
