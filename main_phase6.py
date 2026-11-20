"""Run Phase 6A policy retrieval, approval, reporting, and observability."""

import argparse
import re
import sys
from datetime import datetime, timezone
from pprint import pprint
from typing import Any, Callable

from main_phase4 import print_case_plan
from main_phase5 import (
    _read_interactive_email,
    build_case_record,
    rerun_saved_case_debate,
)
from src.approval import approval_loop
from src.case_store import (
    DEFAULT_CASES_DIR,
    PROJECT_ROOT,
    load_case,
    save_case,
)
from src.debate import normalize_text
from src.desk_chief import print_nested_steps, run_desk_chief
from src.intake import MAX_EMAIL_CHARS, load_email, trim_email_text
from src.observability import (
    DEFAULT_LOG_DB_PATH,
    format_recent_calls,
    log_agent_message,
    read_recent_calls,
    start_logging,
    stop_logging,
)
from src.rag import (
    PolicyChecker,
    PolicyLookupResult,
    format_policy_appendix,
)
from src.report import write_report

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)
POLICY_SECTION_PATTERN = re.compile(r"(?im)^\s*Policy matches:\s*$")


def _apply_policy_result(
    case: dict[str, Any],
    result: PolicyLookupResult,
) -> dict[str, Any]:
    """Store retrieved evidence and append it to the approval-facing verdict."""
    verdict = normalize_text(str(case.get("current_verdict", "")))
    base_verdict = POLICY_SECTION_PATTERN.split(verdict, maxsplit=1)[0].rstrip()
    case["policy_matches"] = [
        {
            "rule_id": match.rule_id,
            "rule_text": normalize_text(match.rule_text),
            "why_matches": normalize_text(match.why_matches),
            "score": match.score,
        }
        for match in result.policy_matches
    ]
    case["similar_past_cases"] = [
        {
            "case_id": match.case_id,
            "summary": normalize_text(match.summary),
            "why_similar": normalize_text(match.why_similar),
            "score": match.score,
        }
        for match in result.similar_past_cases
    ]
    case["current_verdict"] = (
        f"{base_verdict}\n\n{format_policy_appendix(result)}"
    )
    return case


def _retrieve_policy(
    checker: PolicyChecker,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Run the local Policy_Checker with saved email and Phase 2 summaries."""
    summaries = case.get("pipeline_summaries", [])
    if not isinstance(summaries, list):
        raise RuntimeError("The case does not contain valid Phase 2 summaries.")
    base_verdict = POLICY_SECTION_PATTERN.split(
        normalize_text(str(case.get("current_verdict", ""))),
        maxsplit=1,
    )[0].rstrip()
    result = checker.check(
        str(case.get("email_text", "")),
        [str(summary) for summary in summaries],
        base_verdict,
    )
    return _apply_policy_result(case, result)


def _approve_case(
    case: dict[str, Any],
    checker: PolicyChecker,
    input_fn: Callable[[str], str],
    verbose: bool,
    cases_dir: str,
    offline: bool = False,
    chart_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Reuse Phase 5 approval and refresh policy after any debate rerun."""
    def persist(current_case: dict[str, Any]) -> None:
        save_case(current_case, cases_dir)

    def rerun(
        current_case: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        updated = rerun_saved_case_debate(
            current_case,
            reason,
            verbose=verbose,
            offline=offline,
        )
        _retrieve_policy(checker, updated)
        if chart_callback is not None:
            chart_callback(updated)
        return updated

    return approval_loop(
        case,
        input_fn=input_fn,
        save_fn=persist,
        rerun_debate_fn=rerun,
    )


def _finish_case(
    case: dict[str, Any],
    cases_dir: str,
) -> str:
    """Write the final report, then persist its path and final case state."""
    case["report_generated_at"] = datetime.now(timezone.utc).isoformat()
    report_path = write_report(case)
    log_agent_message(
        "Report_Writer",
        f"Generated report for case {case['case_id']}.",
    )
    case["report_path"] = report_path.relative_to(PROJECT_ROOT).as_posix()
    save_case(case, cases_dir)
    return case["report_path"]


def _new_case(
    email_text: str,
    source_type: str,
    checker: PolicyChecker,
    input_fn: Callable[[str], str],
    verbose: bool,
    cases_dir: str,
    offline: bool = False,
    chart_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run Desk_Chief and policy retrieval before asking for approval."""
    print_case_plan()
    result = run_desk_chief(
        email_text,
        verbose=verbose,
        offline=offline,
    )
    print_nested_steps(
        result.phase2_results,
        result.debate_result,
        verbose=verbose,
    )
    case = build_case_record(email_text, source_type, result)
    _retrieve_policy(checker, case)
    if chart_callback is not None:
        chart_callback(case)
    print("\n=== Desk_Chief ===")
    print(normalize_text(case["current_verdict"]))
    save_case(case, cases_dir)
    return _approve_case(
        case,
        checker,
        input_fn,
        verbose,
        cases_dir,
        offline=offline,
        chart_callback=chart_callback,
    )


def _resume_case(
    case_id: str,
    checker: PolicyChecker,
    input_fn: Callable[[str], str],
    verbose: bool,
    cases_dir: str,
    offline: bool = False,
    chart_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Resume a saved case at approval, without rerunning Phase 2."""
    case = load_case(case_id, cases_dir)
    print(f"Resuming case: {case_id}")
    if not case.get("policy_matches") and not case.get("similar_past_cases"):
        _retrieve_policy(checker, case)
        save_case(case, cases_dir)
    if chart_callback is not None and not case.get("chart_path"):
        chart_callback(case)
        save_case(case, cases_dir)
    print("Last verdict:")
    print(normalize_text(str(case.get("current_verdict", ""))))
    notes = case.get("analyst_notes", [])
    if isinstance(notes, list) and notes:
        print("Analyst notes:")
        for note in notes:
            print(f"- {normalize_text(str(note))}")
    return _approve_case(
        case,
        checker,
        input_fn,
        verbose,
        cases_dir,
        offline=offline,
        chart_callback=chart_callback,
    )


def main(
    argv: list[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Read an email or resume a case, then run Phase 6A."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Analyze pasted email text.")
    input_group.add_argument("--file", help="Read email text from a .txt or .eml file.")
    input_group.add_argument(
        "--image",
        help="Transcribe a .png, .jpg, .jpeg, or .webp screenshot.",
    )
    input_group.add_argument("--resume", metavar="CASE_ID", help="Resume a saved case.")
    parser.add_argument("--verbose", action="store_true", help="Show nested steps.")
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show recent AutoGen calls after the run, or by itself.",
    )
    parser.add_argument(
        "--log-limit",
        type=_positive_int,
        default=20,
        metavar="N",
        help="Number of recent LLM calls to show (default: 20).",
    )
    args = parser.parse_args(argv)
    cases_dir = str(DEFAULT_CASES_DIR)
    has_input = any(
        value is not None
        for value in (args.text, args.file, args.image, args.resume)
    )
    if args.show_log and not has_input:
        print(format_recent_calls(read_recent_calls(args.log_limit)))
        return

    logging_started = False
    try:
        start_logging(DEFAULT_LOG_DB_PATH)
        logging_started = True
        checker = PolicyChecker()
        if args.resume is not None:
            case = _resume_case(
                args.resume,
                checker,
                input_fn,
                args.verbose,
                cases_dir,
            )
        else:
            if args.text is not None:
                email_text = load_email(text=args.text)
                source_type = "text"
            elif args.file is not None:
                email_text = load_email(file=args.file)
                source_type = "file"
            elif args.image is not None:
                print(IMAGE_LINK_WARNING, flush=True)
                email_text = load_email(image=args.image)
                source_type = "image"
            else:
                email_text = _read_interactive_email()
                source_type = "text"

            email_text, trimmed = trim_email_text(email_text)
            if trimmed:
                print(
                    f"Note: email text exceeded {MAX_EMAIL_CHARS} characters; "
                    "headers, body edges, and links were retained."
                )
            case = _new_case(
                email_text,
                source_type,
                checker,
                input_fn,
                args.verbose,
                cases_dir,
            )

        report_path = _finish_case(case, cases_dir)
        print(f"\nCase ID: {case['case_id']}")
        print(f"Case status: {case['status']}")
        print(f"Report_Writer saved: {report_path}")
        if "total_cost" in case:
            print("Total cost across nested chats:")
            pprint(case["total_cost"], sort_dicts=False)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    finally:
        if logging_started:
            stop_logging()

    if args.show_log:
        print("\n=== Recent LLM calls ===")
        print(format_recent_calls(read_recent_calls(args.log_limit)))


def _positive_int(value: str) -> int:
    """Parse a positive integer command-line value."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    main()
