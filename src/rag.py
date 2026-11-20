"""Retrieve relevant local security policy and synthetic past cases."""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autogen import ConversableAgent, UserProxyAgent

from src.debate import normalize_text
from src.tools import check_auth_results, extract_headers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_ROOT / "policy"
POLICY_FILE = POLICY_DIR / "security_policy.md"
PAST_CASES_FILE = POLICY_DIR / "past_cases.md"
RAW_IP_PATTERN = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
)
ITEM_PATTERN = re.compile(r"^\s*-\s+\*\*([A-Z]+-?\d+)\*\*\s+(.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TOKEN_ALIASES = {
    "verifications": "verify",
    "verification": "verify",
    "verified": "verify",
    "verifies": "verify",
}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "approved",
    "case",
    "cases",
    "by",
    "checked",
    "domain",
    "domains",
    "email",
    "emails",
    "evidence",
    "for",
    "from",
    "in",
    "is",
    "it",
    "indicator",
    "indicators",
    "legit",
    "link",
    "links",
    "no",
    "notice",
    "not",
    "of",
    "on",
    "or",
    "request",
    "requests",
    "results",
    "risk",
    "sender",
    "should",
    "suspicious",
    "the",
    "to",
    "their",
    "use",
    "used",
    "uses",
    "through",
    "tools",
    "phishing",
    "with",
}


@dataclass(frozen=True)
class PolicyMatch:
    """One local policy rule and its matching evidence."""

    rule_id: str
    rule_text: str
    why_matches: str
    score: float


@dataclass(frozen=True)
class PastCaseMatch:
    """One synthetic past case similar to the current evidence."""

    case_id: str
    summary: str
    why_similar: str
    score: float


@dataclass(frozen=True)
class PolicyLookupResult:
    """The policy and past-case evidence returned for an email case."""

    policy_matches: list[PolicyMatch]
    similar_past_cases: list[PastCaseMatch]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible results for the local AutoGen agent."""
        return asdict(self)


class LocalPolicyRetriever:
    """Rank short local Markdown rules and cases using keyword relevance."""

    def __init__(
        self,
        policy_file: str | Path = POLICY_FILE,
        past_cases_file: str | Path = PAST_CASES_FILE,
        max_policy_matches: int = 4,
        max_past_cases: int = 2,
    ) -> None:
        if max_policy_matches < 1 or max_past_cases < 1:
            raise ValueError("Retriever result limits must be positive.")
        self.policy_items = _read_items(policy_file)
        self.past_case_items = _read_items(past_cases_file)
        self.max_policy_matches = max_policy_matches
        self.max_past_cases = max_past_cases

    def retrieve(
        self,
        email_text: str,
        pipeline_summaries: list[str] | None = None,
        verdict: str = "",
    ) -> PolicyLookupResult:
        """Return the most relevant local rules and synthetic case examples."""
        context = "\n".join(
            [email_text, *(pipeline_summaries or []), verdict]
        )
        query_tokens = _tokens(context)
        if RAW_IP_PATTERN.search(context) and re.search(
            r"https?://[^\s]*\d{1,3}(?:\.\d{1,3}){3}", context, re.IGNORECASE
        ):
            query_tokens.update({"raw", "ip", "address", "numeric", "host"})
        if {"verify", "account"} <= query_tokens:
            query_tokens.add("credential")
        authentication_unavailable = _authentication_unavailable(
            email_text,
            pipeline_summaries or [],
        )

        policy_matches = _rank_items(
            self.policy_items,
            query_tokens,
            self.max_policy_matches,
            self.past_case_items + self.policy_items,
        )
        past_cases = _rank_items(
            self.past_case_items,
            query_tokens,
            self.max_past_cases,
            self.past_case_items + self.policy_items,
        )

        return PolicyLookupResult(
            policy_matches=[
                PolicyMatch(
                    rule_id=item_id,
                    rule_text=text,
                    why_matches=_why_matches(
                        item_id,
                        text,
                        query_tokens,
                        authentication_unavailable,
                    ),
                    score=score,
                )
                for item_id, text, score in policy_matches
            ],
            similar_past_cases=[
                PastCaseMatch(
                    case_id=item_id,
                    summary=text,
                    why_similar=_matched_terms(text, query_tokens),
                    score=score,
                )
                for item_id, text, score in past_cases
            ],
        )


class PolicyChecker:
    """Expose the deterministic local retriever as an AutoGen agent."""

    def __init__(self, retriever: LocalPolicyRetriever | None = None) -> None:
        self.retriever = retriever or LocalPolicyRetriever()
        self.agent = ConversableAgent(
            name="Policy_Checker",
            system_message=(
                "You are the local Policy_Checker. Return the exact JSON lookup "
                "result supplied by the registered deterministic reply."
            ),
            llm_config=False,
            human_input_mode="NEVER",
        )
        self.requester = UserProxyAgent(
            name="Policy_Requester",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.agent.register_reply(
            trigger=UserProxyAgent,
            reply_func=self._reply,
        )

    def check(
        self,
        email_text: str,
        pipeline_summaries: list[str] | None = None,
        verdict: str = "",
    ) -> PolicyLookupResult:
        """Ask the local agent to retrieve evidence without an LLM call."""
        request = json.dumps(
            {
                "email_text": normalize_text(email_text),
                "pipeline_summaries": [
                    normalize_text(item) for item in (pipeline_summaries or [])
                ],
                "verdict": normalize_text(verdict),
            },
            ensure_ascii=True,
        )
        chat_result = self.requester.initiate_chat(
            self.agent,
            message=request,
            max_turns=1,
            summary_method="last_msg",
            silent=True,
        )
        summary = chat_result.summary
        if not isinstance(summary, str) or not summary.strip():
            raise RuntimeError("Policy_Checker returned an empty lookup result.")
        try:
            value = json.loads(summary)
            result = _result_from_dict(value)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Policy_Checker returned invalid lookup data.") from error
        return result

    def _reply(
        self,
        recipient: ConversableAgent,
        messages: list[dict[str, Any]],
        sender: Any,
        config: Any,
    ) -> tuple[bool, str]:
        """Handle one local lookup request and return its structured result."""
        del recipient, sender, config
        if not messages:
            raise ValueError("Policy_Checker received an empty request.")
        try:
            request = json.loads(str(messages[-1].get("content") or ""))
        except json.JSONDecodeError as error:
            raise ValueError("Policy_Checker received invalid request data.") from error
        if not isinstance(request, dict):
            raise ValueError("Policy_Checker request must be a JSON object.")

        email_text = request.get("email_text")
        summaries = request.get("pipeline_summaries", [])
        verdict = request.get("verdict", "")
        if not isinstance(email_text, str) or not isinstance(summaries, list):
            raise ValueError("Policy_Checker request fields have invalid types.")
        result = self.retriever.retrieve(
            email_text,
            [item for item in summaries if isinstance(item, str)],
            verdict if isinstance(verdict, str) else "",
        )
        return True, json.dumps(result.to_dict(), ensure_ascii=True)


def format_policy_appendix(result: PolicyLookupResult) -> str:
    """Format retrieved evidence for the approval prompt and final case summary."""
    lines = ["Policy matches:"]
    if result.policy_matches:
        for match in result.policy_matches:
            lines.append(
                f"- [{match.rule_id}] {match.rule_text} "
                f"Why it matches: {match.why_matches}"
            )
    else:
        lines.append("- No relevant policy rules matched the case evidence.")

    lines.append("Similar past cases:")
    if result.similar_past_cases:
        for case in result.similar_past_cases:
            lines.append(
                f"- [{case.case_id}] {case.summary} "
                f"Why similar: {case.why_similar}"
            )
    else:
        lines.append("- No similar sample cases were found.")
    return normalize_text("\n".join(lines))


def policy_result_from_case(case: dict[str, Any]) -> PolicyLookupResult:
    """Read policy evidence from a saved case, validating its basic shape."""
    raw_matches = case.get("policy_matches", [])
    raw_cases = case.get("similar_past_cases", [])
    if not isinstance(raw_matches, list) or not isinstance(raw_cases, list):
        raise ValueError("Saved policy results must be lists.")
    value = {
        "policy_matches": raw_matches,
        "similar_past_cases": raw_cases,
    }
    return _result_from_dict(value)


def _read_items(path: str | Path) -> list[tuple[str, str]]:
    """Read the ID and text from each simple Markdown bullet item."""
    item_path = Path(path)
    try:
        lines = item_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Could not read local retrieval file '{item_path}'.") from error
    items: list[tuple[str, str]] = []
    for line in lines:
        match = ITEM_PATTERN.match(line)
        if match:
            items.append((match.group(1), normalize_text(match.group(2))))
    if not items:
        raise RuntimeError(f"No retrieval items were found in '{item_path}'.")
    return items


def _tokens(text: str) -> set[str]:
    return {
        TOKEN_ALIASES.get(token, token)
        for token in TOKEN_PATTERN.findall(normalize_text(text).casefold())
        if token not in STOP_WORDS and len(token) > 1
    }


def _rank_items(
    items: list[tuple[str, str]],
    query_tokens: set[str],
    limit: int,
    corpus: list[tuple[str, str]],
) -> list[tuple[str, str, float]]:
    """Score each item with a small inverse-document-frequency weighting."""
    if not query_tokens:
        return []
    document_tokens = [_tokens(text) for _, text in corpus]
    document_frequency: dict[str, int] = {}
    for tokens in document_tokens:
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    ranked: list[tuple[str, str, float, int]] = []
    for position, (item_id, text) in enumerate(items):
        tokens = _tokens(text)
        matched = tokens & query_tokens
        if len(matched) < 2:
            continue
        score = sum(
            math.log((len(corpus) + 1) / (document_frequency[token] + 1)) + 1
            for token in matched
        ) / math.sqrt(max(1, len(tokens)))
        ranked.append((item_id, text, score, position))
    ranked.sort(key=lambda entry: (-entry[2], entry[3]))
    return [(item_id, text, score) for item_id, text, score, _ in ranked[:limit]]


def _matched_terms(text: str, query_tokens: set[str]) -> str:
    matched = sorted(_tokens(text) & query_tokens)
    return (
        "Matched evidence terms: " + ", ".join(matched[:6]) + "."
        if matched
        else "Relevant case evidence was found."
    )


def _why_matches(
    rule_id: str,
    text: str,
    query_tokens: set[str],
    authentication_unavailable: bool = False,
) -> str:
    if rule_id == "R7" and authentication_unavailable:
        return (
            "Informational only: SPF, DKIM, and DMARC are not available in "
            "the supplied headers; this is not a risk indicator."
        )
    if rule_id == "R1" and "ip" in query_tokens and "address" in query_tokens:
        return "A numeric IP address appears in a link in the email evidence."
    return _matched_terms(text, query_tokens)


def _authentication_unavailable(
    email_text: str,
    pipeline_summaries: list[str],
) -> bool:
    """Treat absent authentication results as informational, not adverse."""
    statuses: dict[str, str] = {}
    status_pattern = re.compile(
        r"\b(SPF|DKIM|DMARC)\s*:\s*"
        r"(not available|pass|fail|softfail|neutral|none|temperror|permerror)\b",
        re.IGNORECASE,
    )
    for summary in pipeline_summaries:
        for protocol, status in status_pattern.findall(summary):
            statuses[protocol.upper()] = status.casefold()

    if len(statuses) < 3:
        for result in check_auth_results(extract_headers(email_text)):
            protocol, _, status = result.partition(":")
            statuses.setdefault(protocol.strip().upper(), status.strip().casefold())
    return all(
        statuses.get(protocol) == "not available"
        for protocol in ("SPF", "DKIM", "DMARC")
    )


def _result_from_dict(value: Any) -> PolicyLookupResult:
    if not isinstance(value, dict):
        raise ValueError("Policy lookup result must be a JSON object.")
    policy_matches = value.get("policy_matches")
    similar_cases = value.get("similar_past_cases")
    if not isinstance(policy_matches, list) or not isinstance(similar_cases, list):
        raise ValueError("Policy lookup lists are missing.")
    matches = [
        PolicyMatch(
            rule_id=str(item["rule_id"]),
            rule_text=str(item["rule_text"]),
            why_matches=str(item["why_matches"]),
            score=float(item["score"]),
        )
        for item in policy_matches
        if isinstance(item, dict)
    ]
    cases = [
        PastCaseMatch(
            case_id=str(item["case_id"]),
            summary=str(item["summary"]),
            why_similar=str(item["why_similar"]),
            score=float(item["score"]),
        )
        for item in similar_cases
        if isinstance(item, dict)
    ]
    if len(matches) != len(policy_matches) or len(cases) != len(similar_cases):
        raise ValueError("Policy lookup items must be JSON objects.")
    return PolicyLookupResult(matches, cases)
