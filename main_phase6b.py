"""Run Phase 6B with an optional offline model and risk-score chart."""

import argparse
import sys
import time
from pprint import pprint
from typing import Any, Callable

from main_phase6 import (
    IMAGE_LINK_WARNING,
    _finish_case,
    _new_case,
    _resume_case,
    _positive_int,
)
from src.analyst import run_analyst_chart
from src.case_store import DEFAULT_CASES_DIR, PROJECT_ROOT
from src.debate import normalize_text
from src.observability import (
    DEFAULT_LOG_DB_PATH,
    format_recent_calls,
    read_recent_calls,
    start_logging,
    stop_logging,
)
from src.rag import PolicyChecker
from src.intake import MAX_EMAIL_CHARS, load_email, trim_email_text
from src.desk_chief import run_desk_chief


def main(
    argv: list[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> None:
    """Run a new case, resume approval, or perform a cache demonstration."""
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
        "--no-docker",
        action="store_true",
        help="Opt into LocalCommandLineCodeExecutor instead of Docker.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the local rule-based model client without API calls.",
    )
    parser.add_argument(
        "--cache-demo",
        action="store_true",
        help="Run the same email twice and compare uncached cost and duration.",
    )
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
    has_input = any(
        value is not None
        for value in (args.text, args.file, args.image, args.resume)
    )
    if args.show_log and not has_input and not args.cache_demo:
        print(format_recent_calls(read_recent_calls(args.log_limit)))
        return
    if args.cache_demo and args.resume:
        parser.error("--cache-demo cannot be used with --resume.")
    if args.cache_demo and args.offline:
        parser.error("--cache-demo requires the configured online model cache.")

    logging_started = False
    try:
        start_logging(DEFAULT_LOG_DB_PATH)
        logging_started = True
        if args.resume is not None:
            case = _resume_case(
                args.resume,
                PolicyChecker(),
                input_fn,
                args.verbose,
                str(DEFAULT_CASES_DIR),
                offline=args.offline,
                chart_callback=_chart_callback(args.no_docker),
            )
        else:
            email_text, source_type = _read_email(args, input_fn)
            email_text, trimmed = trim_email_text(email_text)
            if trimmed:
                print(
                    f"Note: email text exceeded {MAX_EMAIL_CHARS} characters; "
                    "headers, body edges, and links were retained."
                )
            if args.cache_demo:
                _run_cache_demo(email_text, args.verbose)
                case = None
            else:
                case = _new_case(
                    email_text,
                    source_type,
                    PolicyChecker(),
                    input_fn,
                    args.verbose,
                    str(DEFAULT_CASES_DIR),
                    offline=args.offline,
                    chart_callback=_chart_callback(args.no_docker),
                )

        if case is not None:
            report_path = _finish_case(case, str(DEFAULT_CASES_DIR))
            print(f"\nCase ID: {case['case_id']}")
            print(f"Case status: {case['status']}")
            print(f"Report_Writer saved: {report_path}")
            if "total_cost" in case:
                print("Total cost across nested chats:")
                pprint(case["total_cost"], sort_dicts=False)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {normalize_text(str(error))}\n")
    finally:
        if logging_started:
            stop_logging()

    if args.show_log:
        print("\n=== Recent LLM calls ===")
        print(format_recent_calls(read_recent_calls(args.log_limit)))


def _read_email(
    args: argparse.Namespace,
    input_fn: Callable[[str], str],
) -> tuple[str, str]:
    """Load one supported input mode and identify its source for the case."""
    if args.text is not None:
        return load_email(text=args.text), "text"
    if args.file is not None:
        return load_email(file=args.file), "file"
    if args.image is not None:
        print(IMAGE_LINK_WARNING, flush=True)
        return load_email(image=args.image), "image"
    if args.cache_demo:
        return _interactive_email(input_fn), "text"
    return _interactive_email(input_fn), "text"


def _interactive_email(input_fn: Callable[[str], str]) -> str:
    """Collect a pasted email ending with END using the injectable input."""
    print("Paste the email. Finish with a line containing only END.")
    lines: list[str] = []
    while True:
        line = input_fn("")
        if line == "END":
            break
        lines.append(line)
    email_text = "\n".join(lines).strip()
    if not email_text:
        raise ValueError("The pasted email is empty.")
    return load_email(text=email_text)


def _chart_callback(no_docker: bool) -> Callable[[dict[str, Any]], None]:
    """Return a hook that stores a successfully created chart in the case."""
    def create_chart(case: dict[str, Any]) -> None:
        path = run_analyst_chart(
            str(case["case_id"]),
            str(case.get("email_text", "")),
            no_docker=no_docker,
        )
        if path is None:
            case.pop("chart_path", None)
        else:
            case["chart_path"] = path.relative_to(PROJECT_ROOT).as_posix()

    return create_chart


def _run_cache_demo(email_text: str, verbose: bool) -> None:
    """Run the same Desk_Chief pipeline twice and report real cache usage."""
    print("Cache seed configured:", _cache_seed())
    for run_number in (1, 2):
        started = time.perf_counter()
        result = run_desk_chief(email_text, verbose=verbose)
        duration = time.perf_counter() - started
        uncached_cost = _uncached_cost(result.total_cost)
        print(
            f"Run {run_number}: cost excluding cached inference "
            f"${uncached_cost:.6f}; total time {duration:.2f} seconds"
        )
        if run_number == 2 and uncached_cost != 0:
            print(
                "Cache note: the second run still reported uncached cost; "
                "confirm that the provider and prompt were unchanged."
            )


def _cache_seed() -> Any:
    """Read the configured cache seed without exposing any credentials."""
    from src.config import llm_config

    return llm_config.get("cache_seed")


def _uncached_cost(total_cost: dict[str, Any]) -> float:
    """Return only AutoGen's usage excluding cached inference."""
    usage = total_cost.get("usage_excluding_cached_inference", {})
    if not isinstance(usage, dict):
        return 0.0
    return float(usage.get("total_cost", 0) or 0)


if __name__ == "__main__":
    main()
