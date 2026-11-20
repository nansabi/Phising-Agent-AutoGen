"""Run email analysis, human approval, and persistent case management."""

import argparse
import sys
from contextlib import redirect_stdout
from io import StringIO
from pprint import pprint
from typing import Any, Callable

from autogen.agentchat.chat import ChatResult

from main_phase4 import print_case_plan
from src.approval import approval_loop
from src.case_store import (
    DEFAULT_CASES_DIR,
    create_case_id,
    format_case_list,
    list_cases,
    load_case,
    save_case,
)
from src.debate import DebateResult, normalize_text, run_debate
from src.desk_chief import (
    DeskChiefResult,
    aggregate_costs,
    format_final_reply,
    format_pipeline_carryover,
    print_nested_steps,
    run_desk_chief,
)
from src.intake import MAX_EMAIL_CHARS, load_email, trim_email_text

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)


def build_case_record(
    email_text: str,
    source_type: str,
    result: DeskChiefResult,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Create the initial open case from the Desk_Chief run."""
    debate = result.debate_result
    return {
        "case_id": case_id or create_case_id(),
        "source_type": source_type,
        "email_text": email_text,
        "pipeline_summaries": [
            normalize_text(chat_result.summary or "")
            for chat_result in result.phase2_results
        ],
        "pipeline_costs": [chat_result.cost for chat_result in result.phase2_results],
        "debate_history": [
            {
                "attempt": 1,
                "human_challenge": "",
                "messages": debate.messages,
            }
        ],
        "debate_costs": [debate.cost],
        "current_verdict": normalize_text(result.final_reply),
        "debate_rounds_used": debate.rounds_used,
        "skeptic_status": "AGREED" if debate.skeptic_agreed else "UNRESOLVED",
        "total_cost": result.total_cost,
        "status": "OPEN",
        "analyst_notes": [],
        "rejection_reasons": [],
    }


def rerun_saved_case_debate(
    case: dict[str, Any],
    reason: str,
    verbose: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    """Continue the saved Phase 3 chat with the analyst's new challenge."""
    summaries = case.get("pipeline_summaries")
    history = case.get("debate_history")
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise RuntimeError("The saved case does not contain Phase 2 summaries.")
    if not isinstance(history, list) or not history:
        raise RuntimeError("The saved case does not contain debate history.")
    previous_messages = history[-1].get("messages")
    if not isinstance(previous_messages, list) or not previous_messages:
        raise RuntimeError("The saved case has no resumable debate messages.")

    phase2_results = [
        ChatResult(summary=str(summary))
        for summary in summaries
    ]
    pipeline_carryover = format_pipeline_carryover(phase2_results)
    if verbose:
        debate_result = run_debate(
            phase2_results,
            pipeline_carryover=pipeline_carryover,
            human_challenge=reason,
            resume_messages=previous_messages,
            offline=offline,
        )
    else:
        with redirect_stdout(StringIO()):
            debate_result = run_debate(
                phase2_results,
                pipeline_carryover=pipeline_carryover,
                human_challenge=reason,
                resume_messages=previous_messages,
                offline=offline,
            )

    case["current_verdict"] = format_final_reply(debate_result)
    case["debate_rounds_used"] = debate_result.rounds_used
    case["skeptic_status"] = (
        "AGREED" if debate_result.skeptic_agreed else "UNRESOLVED"
    )
    case["debate_history"].append(
        {
            "attempt": len(case["debate_history"]) + 1,
            "human_challenge": reason,
            "messages": debate_result.messages,
        }
    )
    case.setdefault("debate_costs", []).append(debate_result.cost)
    case["total_cost"] = aggregate_costs(
        list(case.get("pipeline_costs", []))
        + list(case["debate_costs"])
    )
    return case


def resume_case(
    case_id: str,
    input_fn: Callable[[str], str] = input,
    cases_dir: str = str(DEFAULT_CASES_DIR),
    mask_pii: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Reopen a saved case at the approval step without rerunning Phase 2."""
    case = load_case(case_id, cases_dir)
    print(f"Resuming case: {case_id}")
    print("Last verdict:")
    print(normalize_text(str(case.get("current_verdict", ""))))
    notes = case.get("analyst_notes", [])
    if notes:
        print("Analyst notes:")
        for note in notes:
            print(f"- {normalize_text(str(note))}")

    def persist(current_case: dict[str, Any]) -> None:
        save_case(current_case, cases_dir, mask_pii=mask_pii)

    def rerun(
        current_case: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        return rerun_saved_case_debate(current_case, reason, verbose=verbose)

    return approval_loop(
        case,
        input_fn=input_fn,
        save_fn=persist,
        rerun_debate_fn=rerun,
    )


def _new_case(
    email_text: str,
    source_type: str,
    verbose: bool,
    cases_dir: str,
    mask_pii: bool,
) -> dict[str, Any]:
    """Analyze the email, save an OPEN case, and request approval."""
    print_case_plan()
    result = run_desk_chief(email_text, verbose=verbose)
    print_nested_steps(
        result.phase2_results,
        result.debate_result,
        verbose=verbose,
    )
    print("\n=== Desk_Chief ===")
    print(normalize_text(result.final_reply))

    case = build_case_record(email_text, source_type, result)
    save_case(case, cases_dir, mask_pii=mask_pii)

    def persist(current_case: dict[str, Any]) -> None:
        save_case(current_case, cases_dir, mask_pii=mask_pii)

    def rerun(
        current_case: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        return rerun_saved_case_debate(current_case, reason, verbose=verbose)

    return approval_loop(
        case,
        save_fn=persist,
        rerun_debate_fn=rerun,
    )


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
    """Run one Phase 5 case, list cases, or resume a saved case."""
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
    input_group.add_argument(
        "--list-cases",
        action="store_true",
        help="List saved cases and exit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show nested steps.")
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--mask-pii",
        dest="mask_pii",
        action="store_true",
        default=True,
        help="Mask personal data in saved cases (default).",
    )
    mask_group.add_argument(
        "--no-mask-pii",
        dest="mask_pii",
        action="store_false",
        help="Save case text without PII masking.",
    )
    args = parser.parse_args()
    cases_dir = str(DEFAULT_CASES_DIR)

    try:
        if args.list_cases:
            print(format_case_list(list_cases(cases_dir)))
            return
        if args.resume is not None:
            case = resume_case(
                args.resume,
                cases_dir=cases_dir,
                mask_pii=args.mask_pii,
                verbose=args.verbose,
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
                args.verbose,
                cases_dir,
                args.mask_pii,
            )

        print(f"\nCase ID: {case['case_id']}")
        print(f"Case status: {case['status']}")
        print(f"Saved to: cases\\{case['case_id']}.json")
        if "total_cost" in case:
            print("Total cost across nested chats:")
            pprint(case["total_cost"], sort_dicts=False)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
