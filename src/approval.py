"""Handle human approval, rejection challenges, and analyst notes."""

from typing import Any, Callable

from src.debate import normalize_text

APPROVAL_PROMPT = (
    "Approve verdict? [a]pprove / [r]eject / [e]dit note / [s]ave and exit"
)


def approval_loop(
    case: dict[str, Any],
    input_fn: Callable[[str], str] = input,
    save_fn: Callable[[dict[str, Any]], Any] | None = None,
    rerun_debate_fn: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    max_rejections: int = 2,
) -> dict[str, Any]:
    """Prompt for a decision, persisting each change through injected callbacks."""
    if max_rejections < 1:
        raise ValueError("max_rejections must be at least one.")
    if save_fn is None:
        from src.case_store import save_case

        save_fn = save_case

    case.setdefault("analyst_notes", [])
    case.setdefault("rejection_reasons", [])
    case.setdefault("status", "OPEN")
    if not isinstance(case["analyst_notes"], list):
        raise ValueError("The case analyst_notes field must be a list.")
    if not isinstance(case["rejection_reasons"], list):
        raise ValueError("The case rejection_reasons field must be a list.")

    while True:
        choice = input_fn(f"{APPROVAL_PROMPT}: ").strip().casefold()
        if choice in {"a", "approve", "approved"}:
            case["status"] = "APPROVED"
            save_fn(case)
            return case

        if choice in {"e", "edit", "note"}:
            note = input_fn("Analyst note: ").strip()
            if note:
                case["analyst_notes"].append(note)
                save_fn(case)
            else:
                print("Analyst note cannot be empty.")
            continue

        if choice in {"s", "save", "exit"}:
            case["status"] = "OPEN"
            save_fn(case)
            return case

        if choice in {"r", "reject", "rejected"}:
            rejection_reasons = case["rejection_reasons"]
            if len(rejection_reasons) >= max_rejections:
                case["status"] = "NEEDS_REVIEW"
                save_fn(case)
                return case
            reason = input_fn("Short rejection reason: ").strip()
            if not reason:
                print("A short rejection reason is required.")
                continue
            rejection_reasons.append(reason)
            case["status"] = "REJECTED"
            save_fn(case)

            if rerun_debate_fn is None:
                raise RuntimeError(
                    "A debate rerun callback is required after rejecting a verdict."
                )
            updated_case = rerun_debate_fn(case, reason)
            if updated_case is not case:
                case = updated_case
            case["status"] = "REJECTED"
            print("Updated verdict:")
            print(normalize_text(str(case.get("current_verdict", ""))))
            save_fn(case)

            if len(case["rejection_reasons"]) >= max_rejections:
                case["status"] = "NEEDS_REVIEW"
                save_fn(case)
                return case
            continue

        print("Choose a, r, e, or s.")
