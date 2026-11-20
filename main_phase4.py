"""Run the Phase 4 Desk_Chief nested-chat email review."""

import argparse
import sys
from pprint import pprint

from src.debate import normalize_text
from src.desk_chief import CASE_PLAN, print_nested_steps, run_desk_chief
from src.intake import load_email

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)


def print_case_plan() -> None:
    """Print the short code-defined steps before starting the nested chats."""
    print("Case plan:")
    for step_number, step in enumerate(CASE_PLAN, start=1):
        print(f"{step_number}. {step}")


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
    """Read an email in one of four supported ways and run Desk_Chief."""
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print summaries and costs for the nested steps.",
    )
    args = parser.parse_args()

    try:
        if args.text is not None:
            email_text = load_email(text=args.text)
        elif args.file is not None:
            email_text = load_email(file=args.file)
        elif args.image is not None:
            print(IMAGE_LINK_WARNING, flush=True)
            email_text = load_email(image=args.image)
        else:
            email_text = _read_interactive_email()

        print_case_plan()
        result = run_desk_chief(email_text, verbose=args.verbose)
        print_nested_steps(
            result.phase2_results,
            result.debate_result,
            verbose=args.verbose,
        )
        print("\n=== Desk_Chief ===")
        print(normalize_text(result.final_reply))
        print("\nTotal cost across all nested chats:")
        pprint(result.total_cost, sort_dicts=False)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
