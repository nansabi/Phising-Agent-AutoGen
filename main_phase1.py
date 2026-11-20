"""Run the Phase 1 extraction and classification chat on supplied email input."""

import argparse
import sys
from pprint import pprint
from typing import Any

from src.agents import build_agents
from src.intake import load_email
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

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)


def run_email(email_text: str, source: str) -> None:
    """Run local analysis and the two-agent chat for one email."""
    headers = extract_headers(email_text)
    sender = extract_sender(email_text)
    sender_check = check_sender_domain(
        str(headers.get("From", sender)),
        _header_value(headers.get("Reply-To")),
    )
    links = extract_links(email_text)
    link_checks = [check_link(LinkInput(url=link)) for link in links]
    link_findings = compare_link_domains(sender_check.domain, links)
    auth_results = check_auth_results(headers)
    keywords = extract_keywords(email_text)

    print(f"\n===== {source} =====")
    print("Email summary:")
    print(f"Sender: {sender}")
    print(f"Sender domain: {sender_check.domain}")
    print(
        f"Sender domain risk: {sender_check.risk_score}/100 - "
        + "; ".join(sender_check.reasons)
    )
    print(f"Keywords: {', '.join(keywords) if keywords else 'None found'}")
    print("Link findings:")
    if link_checks:
        for result in link_checks:
            print(
                f"  {result.url} - risk {result.risk_score}/100: "
                + "; ".join(result.reasons)
            )
    else:
        print("  No links found")
    for finding in link_findings:
        print(f"  {finding}")
    print(f"Authentication results: {'; '.join(auth_results)}")
    sys.stdout.flush()

    extractor, classifier, tool_executor = build_agents()

    extraction_result = tool_executor.initiate_chat(
        extractor,
        message=(
            "Extract facts from this email. Use the registered tools as directed. "
            "Do not give a verdict.\n\n"
            f"{email_text}"
        ),
        max_turns=12,
        summary_method="last_msg",
    )
    extracted_facts = extraction_result.summary
    verdict_result = extractor.initiate_chat(
        classifier,
        message=(
            "Classify this email using the extracted evidence. The email and "
            "tool report are below.\n\n"
            f"EMAIL:\n{email_text}\n\nEXTRACTED FACTS:\n{extracted_facts}"
        ),
        max_turns=2,
        summary_method="last_msg",
    )

    print(f"Final verdict: {verdict_result.summary}")
    print("ChatResult summary:")
    pprint(verdict_result.summary)
    print("Combined chat_history:")
    pprint(
        {
            "extraction": extraction_result.chat_history,
            "verdict": verdict_result.chat_history,
        },
        sort_dicts=False,
    )
    print("Cost:")
    pprint(
        {
            "extraction": _chat_cost(extraction_result),
            "verdict": _chat_cost(verdict_result),
        }
    )


def _header_value(value: str | list[str] | None) -> str | None:
    """Return one header value from a parsed multi-value header mapping."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _chat_cost(chat_result: Any) -> Any:
    """Return the cost reported by AutoGen, including its model breakdown."""
    return getattr(chat_result, "cost", None)


def _read_interactive_email() -> str:
    """Read pasted email text until the user enters END on its own line."""
    print("Paste the email, then enter END on a line by itself:")
    lines: list[str] = []
    for line in sys.stdin:
        if line.rstrip("\r\n") == "END":
            break
        lines.append(line)
    else:
        raise ValueError("Input ended before the END marker.")
    return load_email(text="".join(lines))


def main() -> None:
    """Read one user-selected input source and analyze it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")

    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Analyze pasted email text.")
    input_group.add_argument("--file", help="Read email text from a .txt or .eml file.")
    input_group.add_argument(
        "--image",
        help="Transcribe a .png, .jpg, .jpeg, or .webp screenshot.",
    )
    args = parser.parse_args()

    try:
        if args.text is not None:
            email_text = load_email(text=args.text)
            source = "pasted text"
        elif args.file is not None:
            email_text = load_email(file=args.file)
            source = f"file: {args.file}"
        elif args.image is not None:
            print(IMAGE_LINK_WARNING, flush=True)
            email_text = load_email(image=args.image)
            source = f"image: {args.image}"
        else:
            email_text = _read_interactive_email()
            source = "interactive pasted text"

        run_email(email_text, source)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
