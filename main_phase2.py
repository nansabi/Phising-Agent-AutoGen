"""Run the Phase 2 three-chat email analysis pipeline."""

import argparse
import sys
from pprint import pprint
from typing import Any

from autogen.agentchat.chat import ChatResult

from src.intake import load_email
from src.pipeline import STEP_NAMES, run_pipeline

IMAGE_LINK_WARNING = (
    "Links come from visible text only and may not match real URLs."
)


def print_results(results: list[ChatResult]) -> None:
    """Print each step's summary and cost, followed by a compact result list."""
    if len(results) != len(STEP_NAMES):
        raise RuntimeError(
            f"Expected {len(STEP_NAMES)} chat results, received {len(results)}."
        )

    for step_name, result in zip(STEP_NAMES, results):
        print(f"\n=== {step_name} ===")
        print("Summary:")
        print(result.summary)
        print("Cost:")
        pprint(result.cost)

    print("\n=== Final verdict ===")
    print(results[-1].summary)

    result_list: list[dict[str, Any]] = []
    for step_name, result in zip(STEP_NAMES, results):
        result_list.append(
            {
                "step": step_name,
                "summary": result.summary,
                "chat_history_length": len(result.chat_history or []),
                "cost": result.cost,
            }
        )
    print("\n=== ChatResult list ===")
    pprint(result_list, sort_dicts=False)


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
    """Read one of the four supported input forms and run the pipeline."""
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

        print(f"Running Phase 2 for {source}.")
        print_results(run_pipeline(email_text))
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
