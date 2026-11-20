"""Wrap the Phase 2 pipeline and Phase 3 debate as hidden nested chats."""

import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pprint import pprint
from typing import Any, Callable

from autogen import ConversableAgent, UserProxyAgent
from autogen.agentchat.chat import ChatResult

from src.debate import DebateResult, normalize_text, run_debate
from src.pipeline import STEP_NAMES, run_pipeline

CASE_PLAN = (
    "Extract email facts",
    "Check sender, links, and authentication",
    "Classify the email",
    "Debate the verdict",
    "Report the case summary",
)

USAGE_TYPES = (
    "usage_including_cached_inference",
    "usage_excluding_cached_inference",
)


@dataclass
class DeskChiefState:
    """Keep nested results while Desk_Chief handles one email."""

    verbose: bool
    offline: bool = False
    progress_callback: Callable[[str], None] | None = None
    phase2_results: list[ChatResult] = field(default_factory=list)
    pipeline_summary: str = ""
    debate_result: DebateResult | None = None
    final_reply: str = ""


@dataclass
class DeskChiefSession:
    """Hold the user-facing agents and their nested workflow state."""

    desk_chief: ConversableAgent
    user_proxy: UserProxyAgent
    state: DeskChiefState


@dataclass
class DeskChiefResult:
    """Return the single final reply and total nested-chat cost."""

    final_reply: str
    total_cost: dict[str, Any]
    phase2_results: list[ChatResult]
    debate_result: DebateResult


def create_desk_chief(
    verbose: bool = False,
    offline: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> DeskChiefSession:
    """Create Desk_Chief and register its pipeline and debate nested chats."""
    state = DeskChiefState(
        verbose=verbose,
        offline=offline,
        progress_callback=progress_callback,
    )
    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    desk_chief = ConversableAgent(
        name="Desk_Chief",
        system_message=(
            "You are Desk_Chief. Return the final case summary provided by the "
            "nested workflow without adding extra analysis."
        ),
        llm_config=False,
        human_input_mode="NEVER",
    )
    pipeline_runner = ConversableAgent(
        name="Pipeline_Runner",
        llm_config=False,
        human_input_mode="NEVER",
    )
    debate_runner = ConversableAgent(
        name="Debate_Runner",
        llm_config=False,
        human_input_mode="NEVER",
    )

    def run_pipeline_chat(
        recipient: ConversableAgent,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        config: Any = None,
    ) -> tuple[bool, str]:
        """Run Phase 2 when the first nested chat receives the email."""
        del recipient, sender, config
        email_text = _last_message_content(messages)
        if not email_text.strip():
            raise ValueError("The email input is empty.")
        results = _run_internal(
            state,
            run_pipeline,
            email_text,
            offline=state.offline,
            progress_callback=state.progress_callback,
        )
        if len(results) != len(STEP_NAMES):
            raise RuntimeError(
                f"Expected {len(STEP_NAMES)} Phase 2 results, received "
                f"{len(results)}."
            )
        state.phase2_results = results
        state.pipeline_summary = format_pipeline_carryover(results)
        return True, state.pipeline_summary

    def run_debate_chat(
        recipient: ConversableAgent,
        messages: list[dict[str, Any]] | None = None,
        sender: Any = None,
        config: Any = None,
    ) -> tuple[bool, str]:
        """Run Phase 3 using the Phase 2 summary carried into this chat."""
        del recipient, sender, config
        if not state.phase2_results or not state.pipeline_summary:
            raise RuntimeError("The Phase 2 pipeline chat did not produce carryover.")
        nested_message = _last_message_content(messages)
        if state.pipeline_summary not in nested_message:
            raise RuntimeError("The Phase 2 summary was not carried into the debate.")
        if state.progress_callback is not None:
            state.progress_callback("Debate")
        result = _run_internal(
            state,
            run_debate,
            state.phase2_results,
            pipeline_carryover=state.pipeline_summary,
            offline=state.offline,
        )
        state.debate_result = result
        state.final_reply = format_final_reply(result)
        return True, state.final_reply

    pipeline_runner.register_reply(user_proxy, run_pipeline_chat)
    debate_runner.register_reply(user_proxy, run_debate_chat)

    chat_queue = [
        {
            "sender": user_proxy,
            "recipient": pipeline_runner,
            "message": _email_from_outer_chat,
            "max_turns": 1,
            "summary_method": "last_msg",
            "silent": True,
        },
        {
            "sender": user_proxy,
            "recipient": debate_runner,
            "message": (
                "Run the Phase 3 debate using the Phase 2 pipeline summary "
                "carried from the prior nested chat."
            ),
            "max_turns": 1,
            "summary_method": "last_msg",
            "silent": True,
        },
    ]
    desk_chief.register_nested_chats(
        trigger=user_proxy,
        chat_queue=chat_queue,
        reply_func_from_nested_chats="summary_from_nested_chats",
    )

    return DeskChiefSession(desk_chief, user_proxy, state)


def run_desk_chief(
    email_text: str,
    verbose: bool = False,
    offline: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> DeskChiefResult:
    """Send one email to Desk_Chief and return its final reply and costs."""
    session = create_desk_chief(
        verbose,
        offline=offline,
        progress_callback=progress_callback,
    )
    chat_result = session.user_proxy.initiate_chat(
        session.desk_chief,
        message=email_text,
        max_turns=1,
        summary_method="last_msg",
        silent=True,
    )
    state = session.state
    if state.debate_result is None:
        raise RuntimeError("Desk_Chief did not produce a debate result.")

    final_reply = normalize_text(chat_result.summary or state.final_reply)
    if not final_reply:
        raise RuntimeError("Desk_Chief returned an empty case summary.")

    return DeskChiefResult(
        final_reply=final_reply,
        total_cost=aggregate_costs(
            [result.cost for result in state.phase2_results]
            + [state.debate_result.cost]
        ),
        phase2_results=state.phase2_results,
        debate_result=state.debate_result,
    )


def format_final_reply(result: DebateResult) -> str:
    """Format the classifier verdict and debate status as one clean message."""
    classifier_text = normalize_text(result.final_classifier_message).strip()
    verdict_match = re.search(
        r"(?im)^\s*(?:verdict\s*:\s*)?(PHISHING|SUSPICIOUS|LEGIT)\b[^\r\n]*",
        classifier_text,
    )
    if verdict_match is None:
        raise ValueError("The Classifier response does not contain a valid verdict.")

    confidence_match = re.search(
        r"(?im)^\s*confidence\s*:\s*(\d{1,3}\s*%)",
        classifier_text,
    )
    if confidence_match is None:
        confidence_match = re.search(r"\b(\d{1,3}\s*%)", verdict_match.group(0))
    if confidence_match is None:
        raise ValueError("The Classifier response does not contain confidence.")

    sender_match = re.search(
        r"(?im)^\s*Sender domain findings:\s*.*$",
        classifier_text,
    )
    if sender_match is None:
        raise ValueError("The Classifier response lacks Sender domain findings.")

    reasons = re.findall(
        r"(?m)^\s*(?:[-*]\s+|\d+[.)]\s+)(.+?)\s*$",
        classifier_text,
    )
    if len(reasons) < 3:
        raise ValueError("The Classifier response must include three reasons.")

    skeptic_status = "AGREED" if result.skeptic_agreed else "UNRESOLVED"
    reply = "\n".join(
        (
            f"Verdict: {verdict_match.group(1).upper()}",
            f"Confidence: {confidence_match.group(1).replace(' ', '')}",
            sender_match.group(0).strip(),
            "Reasons:",
            f"1. {reasons[0].strip()}",
            f"2. {reasons[1].strip()}",
            f"3. {reasons[2].strip()}",
            f"Debate rounds used: {result.rounds_used}",
            f"Skeptic status: {skeptic_status}",
        )
    )
    return normalize_text(reply)


def format_pipeline_carryover(results: list[ChatResult]) -> str:
    """Make the Phase 2 step summaries explicit for the nested debate chat."""
    if len(results) != len(STEP_NAMES):
        raise RuntimeError(
            f"Expected {len(STEP_NAMES)} Phase 2 results, received {len(results)}."
        )
    return "\n\n".join(
        f"{step_name}:\n{normalize_text(result.summary or '')}"
        for step_name, result in zip(STEP_NAMES, results)
    )


def print_nested_steps(
    phase2_results: list[ChatResult],
    debate_result: DebateResult,
    verbose: bool,
) -> None:
    """Print nested step summaries and costs only in verbose mode."""
    if not verbose:
        return

    for step_name, result in zip(STEP_NAMES, phase2_results):
        print(f"\n=== Nested step: {step_name} ===")
        print(f"Summary:\n{normalize_text(result.summary or '')}")
        print("Cost:")
        pprint(result.cost, sort_dicts=False)

    print("\n=== Nested step: Debate ===")
    print(f"Summary:\n{format_final_reply(debate_result)}")
    print("Cost:")
    pprint(debate_result.cost, sort_dicts=False)


def aggregate_costs(costs: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Sum AutoGen usage and cost values across all nested steps."""
    total: dict[str, Any] = {
        usage_type: {"total_cost": 0.0} for usage_type in USAGE_TYPES
    }
    for cost in costs:
        if not isinstance(cost, dict):
            continue
        for usage_type in USAGE_TYPES:
            usage = cost.get(usage_type, {})
            if not isinstance(usage, dict):
                continue
            target = total[usage_type]
            target["total_cost"] += float(usage.get("total_cost", 0) or 0)
            for model, metrics in usage.items():
                if model == "total_cost" or not isinstance(metrics, dict):
                    continue
                model_total = target.setdefault(model, {})
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        model_total[metric] = model_total.get(metric, 0) + value
    return total


def _run_internal(
    state: DeskChiefState,
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Hide internal chat logs unless verbose output was requested."""
    if state.verbose:
        return function(*args, **kwargs)
    with redirect_stdout(StringIO()):
        return function(*args, **kwargs)


def _last_message_content(messages: list[dict[str, Any]] | None) -> str:
    """Read the content of the latest nested-chat message."""
    if not messages:
        return ""
    return str(messages[-1].get("content") or "")


def _email_from_outer_chat(
    recipient: ConversableAgent,
    messages: list[dict[str, Any]],
    sender: UserProxyAgent,
    config: Any,
) -> str:
    """Pass the original user email to the first nested chat."""
    del recipient, sender, config
    return _last_message_content(messages)
