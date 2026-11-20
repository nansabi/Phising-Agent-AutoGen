"""Write a deterministic Markdown case report from saved case evidence."""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.case_store import CASE_ID_PATTERN, PROJECT_ROOT
from src.debate import normalize_text
from src.intake import mask_pii_text, sender_domain_from_email

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
VERDICT_PATTERN = re.compile(r"(?i)\b(PHISHING|SUSPICIOUS|LEGIT)\b")
CONFIDENCE_PATTERN = re.compile(r"(?im)^\s*Confidence:\s*(.+?)\s*$")
SENDER_FINDINGS_PATTERN = re.compile(
    r"(?im)^\s*Sender domain findings:\s*(.+?)\s*$"
)
class ReportWriter:
    """Render and persist a final report using only existing case evidence."""

    name = "Report_Writer"

    def write(
        self,
        case: dict[str, Any],
        reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    ) -> Path:
        """Write one report atomically and return its path."""
        case_id = str(case.get("case_id", ""))
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError("The case has an invalid case_id.")
        directory = Path(reports_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{case_id}.md"
        temporary_path = path.with_suffix(".md.tmp")
        temporary_path.write_text(build_report(case), encoding="utf-8")
        temporary_path.replace(path)
        return path


def write_report(
    case: dict[str, Any],
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
) -> Path:
    """Convenience function for the deterministic Report_Writer."""
    return ReportWriter().write(case, reports_dir)


def build_report(case: dict[str, Any]) -> str:
    """Build every required report section from saved summaries and decisions."""
    verdict_text = normalize_text(str(case.get("current_verdict", "")))
    verdict_match = VERDICT_PATTERN.search(verdict_text)
    confidence_match = CONFIDENCE_PATTERN.search(verdict_text)
    sender_match = SENDER_FINDINGS_PATTERN.search(verdict_text)
    summaries = case.get("pipeline_summaries", [])
    if not isinstance(summaries, list):
        summaries = []
    link_summary = (
        normalize_text(str(summaries[1]))
        if len(summaries) > 1
        else "Not available in the saved Phase 2 summary."
    )
    auth_results = []
    for line in link_summary.splitlines():
        match = re.match(
            r"(?i)^\s*(?:[-*]\s*)?(SPF|DKIM|DMARC)\s*:\s*(.+?)\s*$",
            line,
        )
        if match:
            auth_results.append(f"{match.group(1).upper()}: {match.group(2)}")
    if not auth_results:
        auth_results = ["Not available in the saved Phase 2 summary."]

    policy_matches = case.get("policy_matches", [])
    similar_cases = case.get("similar_past_cases", [])
    analyst_notes = case.get("analyst_notes", [])
    rejection_reasons = case.get("rejection_reasons", [])
    debate_history = case.get("debate_history", [])
    if not isinstance(policy_matches, list):
        policy_matches = []
    if not isinstance(similar_cases, list):
        similar_cases = []
    if not isinstance(analyst_notes, list):
        analyst_notes = []
    if not isinstance(rejection_reasons, list):
        rejection_reasons = []
    if not isinstance(debate_history, list):
        debate_history = []

    debate_summary = [
        f"Rounds used: {case.get('debate_rounds_used', 'Not recorded')}",
        f"Skeptic status: {case.get('skeptic_status', 'Not recorded')}",
        f"Debate attempts saved: {len(debate_history)}",
    ]
    if rejection_reasons:
        debate_summary.append(
            "Human rejection challenges: " + "; ".join(
                normalize_text(str(reason)) for reason in rejection_reasons
            )
        )
    latest_messages = (
        debate_history[-1].get("messages", [])
        if debate_history and isinstance(debate_history[-1], dict)
        else []
    )
    if isinstance(latest_messages, list):
        classifier_point = _last_message(latest_messages, "Classifier")
        skeptic_point = _last_message(latest_messages, "Skeptic")
    else:
        classifier_point = ""
        skeptic_point = ""

    lines = [
        f"# Phishing Desk Case Report - {case.get('case_id', 'unknown')}",
        "",
        "## Verdict",
        f"- Verdict: {verdict_match.group(1).upper() if verdict_match else 'UNKNOWN'}",
        f"- Confidence: {confidence_match.group(1) if confidence_match else 'Not recorded'}",
        "",
        "## Sender domain findings",
        sender_match.group(1) if sender_match else "Not available in the saved verdict.",
        "",
        "## Link findings",
        _code_block(link_summary),
        "",
        "## Authentication results",
        *[f"- {normalize_text(result)}" for result in auth_results],
        "",
        "## Analyst chart",
        *_chart_lines(case),
        "",
        "## Debate summary",
        *[f"- {normalize_text(item)}" for item in debate_summary],
    ]
    if classifier_point:
        lines.extend(
            (
                "",
                "Latest Classifier point:",
                _code_block(classifier_point),
            )
        )
    if skeptic_point:
        lines.extend(
            (
                "",
                "Latest Skeptic point:",
                _code_block(skeptic_point),
            )
        )

    lines.extend(["", "## Policy matches"])
    if policy_matches:
        for match in policy_matches:
            if isinstance(match, dict):
                lines.append(
                    f"- [{normalize_text(str(match.get('rule_id', 'Rule')))}] "
                    f"{normalize_text(str(match.get('rule_text', '')))} "
                    f"Why it matches: "
                    f"{normalize_text(str(match.get('why_matches', '')))}"
                )
    else:
        lines.append("- No relevant policy rules were recorded.")

    lines.extend(["", "## Similar past cases"])
    if similar_cases:
        for match in similar_cases:
            if isinstance(match, dict):
                lines.append(
                    f"- [{normalize_text(str(match.get('case_id', 'Case')))}] "
                    f"{normalize_text(str(match.get('summary', '')))} "
                    f"Why similar: "
                    f"{normalize_text(str(match.get('why_similar', '')))}"
                )
    else:
        lines.append("- No similar sample cases were recorded.")

    lines.extend(["", "## Analyst notes"])
    if analyst_notes:
        lines.extend(_code_block(str(note)) for note in analyst_notes)
    else:
        lines.append("- None")

    lines.extend(
        (
            "",
            "## Approval status",
            f"- {normalize_text(str(case.get('status', 'UNKNOWN')))}",
            "",
            "## Timestamps",
            f"- Created: {case.get('created_at', 'Not recorded')}",
            f"- Updated: {case.get('updated_at', 'Not recorded')}",
            f"- Report generated: {case.get('report_generated_at', _now_utc())}",
            "",
        )
    )
    report_text = normalize_text("\n".join(lines))
    sender_domain = sender_domain_from_email(str(case.get("email_text", "")))
    return mask_pii_text(report_text, sender_domain)


def _code_block(text: str) -> str:
    """Keep email-derived summaries inert in Markdown reports."""
    safe_text = normalize_text(text).replace("```", "` ` `")
    return f"```text\n{safe_text}\n```"


def _chart_lines(case: dict[str, Any]) -> list[str]:
    """Embed the generated chart only when it matches the case-specific name."""
    case_id = str(case.get("case_id", ""))
    chart_path = str(case.get("chart_path", ""))
    expected_name = f"{case_id}_chart.png"
    if not CASE_ID_PATTERN.fullmatch(case_id) or Path(chart_path).name != expected_name:
        return ["- No Analyst chart was generated."]
    return [
        f"![Risk score chart](./{expected_name})",
        f"- Chart file: `reports/{expected_name}`",
    ]


def _last_message(messages: list[Any], speaker: str) -> str:
    """Find a normalized contribution from one saved debate participant."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("name") != speaker:
            continue
        content = message.get("content")
        if isinstance(content, str):
            return normalize_text(content)
    return ""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
