"""Run the Phase 3 email-analysis debate."""

import argparse
import sys
from pprint import pprint

from src.debate import DebateResult, normalize_text, run_phase3
from src.intake import load_email

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)


def print_results(result: DebateResult) -> None:
    """Print the debate outcome, unresolved points, history size, and cost."""
    print("\n=== Phase 3 debate ===")
    print(f"Rounds used: {result.rounds_used}")
    if result.skeptic_agreed:
        print("Skeptic status: AGREED")
        print("Final verdict:")
        print(normalize_text(result.final_classifier_message))
    elif result.round_limit_hit:
        print("Skeptic status: round limit hit without agreement")
        print("Final verdict: UNRESOLVED - needs human review")
        print("Classifier's last point:")
        print(normalize_text(result.final_classifier_message))
        print("Skeptic's last point:")
        print(normalize_text(result.skeptic_last_message or "(No Skeptic message.)"))
    else:
        print("Skeptic status: debate ended without agreement")
        print("Final verdict: UNRESOLVED - needs human review")
        print("Classifier's last point:")
        print(normalize_text(result.final_classifier_message))
        print("Skeptic's last point:")
        print(normalize_text(result.skeptic_last_message or "(No Skeptic message.)"))

    print(f"\nFull chat history length: {len(result.messages)}")
    print("Debate cost (Classifier + Skeptic):")
    pprint(result.cost, sort_dicts=False)
    print("\n=== ChatResult ===")
    print(f"Summary: {normalize_text(result.chat_result.summary or '')}")
    print(f"Chat history length: {len(result.chat_result.chat_history or [])}")


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
    """Read email input in one of four supported ways and run Phase 3."""
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

        print(f"Running Phase 3 for {source}.")
        print_results(run_phase3(email_text))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
