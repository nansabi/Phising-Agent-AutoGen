"""Provide a small rule-based AutoGen model client with no API dependency."""

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from autogen.oai.client import ModelClient

from src.tools import (
    LinkInput,
    check_auth_results,
    check_link,
    check_sender_domain,
    compare_link_domains,
    extract_headers,
    extract_keywords,
    extract_links,
    extract_sender,
)

OFFLINE_MODEL_NAME = "phishing-desk-rule-based"


@dataclass
class OfflineMessage:
    """A minimal assistant message compatible with AutoGen's client protocol."""

    content: str
    role: str = "assistant"


@dataclass
class OfflineChoice:
    """One response choice returned from the local model client."""

    message: OfflineMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class OfflineResponse:
    """The subset of a chat completion used by AutoGen 0.2."""

    choices: list[OfflineChoice]
    model: str = OFFLINE_MODEL_NAME
    id: str = ""
    created: int = 0
    object: str = "chat.completion"


class RuleBasedModelClient:
    """Answer Phase 2 and debate prompts using the project's offline tools."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        del kwargs
        self.model = str(config.get("model", OFFLINE_MODEL_NAME))

    def create(self, params: dict[str, Any]) -> ModelClient.ModelClientResponseProtocol:
        """Return a rule-based response for an AutoGen chat request."""
        messages = params.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        system_message = _system_message(messages)
        all_text = "\n\n".join(_message_content(item) for item in messages)
        system_lower = system_message.casefold()

        if "you are skeptic" in system_lower:
            answer = "AGREE"
        elif "create a clean list of facts" in all_text.casefold():
            answer = _reflection_facts(messages)
        elif "you are extractor" in system_lower:
            answer = _extractor_facts(messages)
        elif "you are link_checker" in system_lower:
            answer = _link_checker_findings(messages)
        elif "you are classifier" in system_lower:
            answer = _classifier_verdict(
                messages,
                debate="defensive review debate" in system_lower,
            )
        else:
            answer = "No additional tool-derived facts were available."

        return OfflineResponse(
            id=f"offline-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.model,
            choices=[OfflineChoice(message=OfflineMessage(content=answer))],
        )

    def message_retrieval(
        self,
        response: ModelClient.ModelClientResponseProtocol,
    ) -> list[str]:
        """Return assistant text from the local response."""
        return [
            str(choice.message.content or "")
            for choice in response.choices
        ]

    def cost(self, response: ModelClient.ModelClientResponseProtocol) -> float:
        """Offline inference has no provider cost."""
        del response
        return 0.0

    @staticmethod
    def get_usage(
        response: ModelClient.ModelClientResponseProtocol,
    ) -> dict[str, Any]:
        """Report zero token usage and cost for rule-based inference."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "model": response.model,
        }


def offline_llm_config() -> dict[str, Any]:
    """Return AutoGen configuration that selects only the local client."""
    return {
        "config_list": [
            {
                "model": OFFLINE_MODEL_NAME,
                "model_client_cls": RuleBasedModelClient.__name__,
            }
        ],
        "temperature": 0,
        "cache_seed": None,
    }


def _system_message(messages: list[Any]) -> str:
    for message in messages:
        if _message_role(message) == "system":
            return _message_content(message)
    return ""


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    return ""


def _extractor_facts(messages: list[Any]) -> str:
    """Extract facts with the same local tools used by the online agents."""
    email_text = _email_from_messages(messages)
    return _format_extractor_facts(email_text, include_marker=True)


def _reflection_facts(messages: list[Any]) -> str:
    """Reuse the Extractor's tool-derived fields for the summary prompt."""
    for message in reversed(messages):
        if _message_role(message) != "assistant":
            continue
        content = _message_content(message)
        if all(label in content for label in ("Sender:", "Headers:", "Links:", "Keywords:")):
            return content.replace("\nEXTRACTED", "").strip()
    return _format_extractor_facts(
        _email_from_messages(messages),
        include_marker=False,
    )


def _format_extractor_facts(email_text: str, include_marker: bool) -> str:
    headers = extract_headers(email_text)
    fields = [
        f"Sender: {extract_sender(email_text)}",
        "Headers: " + json.dumps(headers, ensure_ascii=True),
        "Links: " + json.dumps(extract_links(email_text), ensure_ascii=True),
        "Keywords: " + json.dumps(extract_keywords(email_text), ensure_ascii=True),
    ]
    if include_marker:
        fields.append("EXTRACTED")
    return "\n".join(fields)


def _link_checker_findings(messages: list[Any]) -> str:
    """Score the extracted links and sender facts without an API call."""
    context = "\n\n".join(_message_content(item) for item in messages)
    headers = _json_field(context, "Headers")
    if not isinstance(headers, dict):
        headers = {}
    links_value = _json_field(context, "Links")
    links = (
        [item for item in links_value if isinstance(item, str)]
        if isinstance(links_value, list)
        else extract_links(context)
    )
    from_header = str(headers.get("From") or _field(context, "Sender") or "")
    reply_value = headers.get("Reply-To")
    reply_to = str(reply_value) if reply_value else None
    domain_result = check_sender_domain(from_header, reply_to)
    auth_results = check_auth_results(headers)
    link_results = [check_link(LinkInput(url=url)) for url in links]
    domain_findings = compare_link_domains(domain_result.domain, links)

    lines = [
        f"Sender domain: {domain_result.domain}",
        f"Sender domain risk score: {domain_result.risk_score}",
        "Sender domain reasons: " + "; ".join(domain_result.reasons),
        "Links:",
    ]
    if link_results:
        for result in link_results:
            lines.extend(
                (
                    f"- Link: {result.url}",
                    f"  Risk Score: {result.risk_score}",
                    "  Reasons: " + "; ".join(result.reasons),
                )
            )
    else:
        lines.append("- No links found.")
    lines.append("Authentication results:")
    lines.extend(f"- {result}" for result in auth_results)
    lines.append("Link-domain findings:")
    lines.extend(f"- {finding}" for finding in domain_findings)
    lines.append("LINKS_CHECKED")
    return "\n".join(lines)


def _classifier_verdict(messages: list[Any], debate: bool) -> str:
    """Classify tool results with simple evidence thresholds."""
    context = "\n\n".join(_message_content(item) for item in messages)
    email_text = _email_from_messages(messages)
    headers = _json_field(context, "Headers")
    if not isinstance(headers, dict):
        headers = extract_headers(email_text)
    links_value = _json_field(context, "Links")
    links = (
        [item for item in links_value if isinstance(item, str)]
        if isinstance(links_value, list)
        else extract_links(email_text or context)
    )
    keywords_value = _json_field(context, "Keywords")
    keywords = (
        {str(item).casefold() for item in keywords_value}
        if isinstance(keywords_value, list)
        else set(extract_keywords(email_text or context))
    )
    from_header = str(
        headers.get("From")
        or _field(context, "Sender")
        or extract_sender(email_text)
    )
    reply_value = headers.get("Reply-To")
    domain_result = check_sender_domain(
        from_header,
        str(reply_value) if reply_value else None,
    )
    link_results = [check_link(LinkInput(url=url)) for url in links]
    link_domain_findings = compare_link_domains(domain_result.domain, links)
    has_credential_request = bool(
        keywords & {"verify", "password", "login", "confirm"}
    )
    has_urgency = bool(keywords & {"urgent", "immediately", "suspended"})
    raw_ip_link = any(
        "URL uses an IP address" in reason
        for result in link_results
        for reason in result.reasons
    )
    lookalike_spelling = any(
        "lookalike spelling" in reason.casefold()
        for result in link_results
        for reason in result.reasons
    ) or any(
        "lookalike spelling" in reason.casefold()
        for reason in domain_result.reasons
    )
    sender_mismatch = any(
        "unrelated to sender domain" in finding.casefold()
        for finding in link_domain_findings
    ) or any(
        "reply-to domain" in reason.casefold()
        or "display name claims" in reason.casefold()
        for reason in domain_result.reasons
    )
    auth_statuses = check_auth_results(headers)
    explicit_auth_fail = any(
        re.search(r":\s*fail$", status, re.IGNORECASE)
        for status in auth_statuses
    )
    behavioral_reasons: list[str] = []
    if has_urgency:
        behavioral_reasons.append(
            "The extracted keywords include urgency language."
        )
    if has_credential_request:
        behavioral_reasons.append(
            "The extracted keywords include credential or verification language."
        )
    if raw_ip_link:
        behavioral_reasons.append(
            "A checked link uses a raw IP address instead of a domain."
        )
    if lookalike_spelling:
        behavioral_reasons.append(
            "A checked sender or link contains a lookalike spelling."
        )
    if sender_mismatch:
        behavioral_reasons.append(
            "Tool results show a link/sender or Reply-To/From domain mismatch."
        )
    if explicit_auth_fail:
        behavioral_reasons.append(
            "Authentication tool results report an explicit SPF, DKIM, or DMARC FAIL."
        )

    if raw_ip_link or (has_credential_request and bool(links)) or (
        lookalike_spelling and has_urgency
    ):
        verdict = "PHISHING"
        confidence = 90
    elif behavioral_reasons:
        verdict = "SUSPICIOUS"
        confidence = 75
    else:
        verdict = "LEGIT"
        confidence = 85

    sender_reasons = "; ".join(domain_result.reasons)
    authentication_unavailable = all(
        status.endswith(": not available")
        for status in auth_statuses
    )
    if not behavioral_reasons:
        reasons = [
            "No concrete behavioral risk indicator was found in the tool results.",
            "Sender and link checks found no raw-IP, lookalike, or domain mismatch.",
        ]
        reasons.append(
            "SPF/DKIM/DMARC are not available; this is informational only."
            if authentication_unavailable
            else "No explicit SPF, DKIM, or DMARC FAIL was found."
        )
    else:
        reasons = behavioral_reasons[:3]
        if len(reasons) < 3 and authentication_unavailable:
            reasons.append(
                "SPF/DKIM/DMARC are not available; this is informational only."
            )
        while len(reasons) < 3:
            reasons.append(
                "No additional concrete behavioral indicator was found in the tool results."
            )
    lines = [
        f"Verdict: {verdict}",
        f"Confidence: {confidence}%",
        f"Sender domain findings: {domain_result.domain} risk "
        f"{domain_result.risk_score}/100; {sender_reasons}",
        "Reasons:",
        *(f"- {reason}" for reason in reasons),
    ]
    if not debate:
        lines.append("TERMINATE")
    return "\n".join(lines)


def _email_from_messages(messages: list[Any]) -> str:
    """Find the original email inside an intake or summary prompt."""
    user_texts = [
        _message_content(message)
        for message in messages
        if _message_role(message) == "user"
    ]
    combined = "\n\n".join(user_texts)
    marker = "Use the extraction tools on this email:"
    marker_index = combined.casefold().find(marker.casefold())
    if marker_index >= 0:
        combined = combined[marker_index + len(marker) :]
    from_match = re.search(r"(?im)^From:\s", combined)
    if from_match:
        combined = combined[from_match.start() :]
    else:
        combined = combined.strip()
    return combined


def _field(text: str, label: str) -> str:
    matches = list(
        re.finditer(
            rf"(?im)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
            text,
        )
    )
    return matches[-1].group(1).strip() if matches else ""


def _json_field(text: str, label: str) -> Any:
    value = _field(text, label)
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
