"""Generate a restricted risk-score chart with an AutoGen code executor."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from autogen import ConversableAgent, UserProxyAgent
from autogen.coding import (
    CodeBlock,
    DockerCommandLineCodeExecutor,
    LocalCommandLineCodeExecutor,
)
from docker.errors import DockerException

from src.case_store import CASE_ID_PATTERN, PROJECT_ROOT
from src.debate import normalize_text
from src.tools import (
    LinkInput,
    check_link,
    check_sender_domain,
    extract_headers,
    extract_links,
    extract_sender,
)

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
CHART_OUTPUT_NAME = "risk_chart.png"
ANALYST_WORK_DIR_NAME = "analyst_work"
ANALYST_PACKAGES_DIR = "analyst_packages"
MATPLOTLIB_SETUP_MARKER = ".matplotlib_setup_complete"
CHART_SETUP_CODE = (
    "python -m pip install --disable-pip-version-check "
    "--no-cache-dir --default-timeout=120 --retries 3 "
    "--upgrade --target analyst_packages matplotlib"
)
LOCAL_CHART_SETUP_CODE = (
    "python -m pip install --disable-pip-version-check --no-cache-dir "
    "--timeout 30 --retries 1 "
    "--upgrade --target analyst_packages matplotlib"
)


class Analyst:
    """Create only fixed plotting code from precomputed local risk scores."""

    name = "Analyst"

    def __init__(self) -> None:
        self.agent = ConversableAgent(
            name=self.name,
            system_message=(
                "You are Analyst. Return the restricted chart code generated "
                "from the supplied risk-score data."
            ),
            llm_config=False,
            human_input_mode="NEVER",
        )
        self.requester = UserProxyAgent(
            name="Chart_Requester",
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.agent.register_reply(
            trigger=UserProxyAgent,
            reply_func=self._write_code,
        )

    def generate_code(
        self,
        risk_data: list[dict[str, Any]],
        case_id: str,
    ) -> str:
        """Ask the local Analyst agent for its fixed, data-only chart script."""
        request = json.dumps(
            {"case_id": case_id, "risk_data": risk_data},
            ensure_ascii=True,
        )
        result = self.requester.initiate_chat(
            self.agent,
            message=request,
            max_turns=1,
            summary_method="last_msg",
            silent=True,
        )
        if not isinstance(result.summary, str) or not result.summary.strip():
            raise RuntimeError("Analyst did not generate chart code.")
        return result.summary

    def _write_code(
        self,
        recipient: ConversableAgent,
        messages: list[dict[str, Any]],
        sender: Any,
        config: Any,
    ) -> tuple[bool, str]:
        """Return a fixed plotting program after validating the numeric input."""
        del recipient, sender, config
        if not messages:
            raise ValueError("Analyst received no chart data.")
        try:
            value = json.loads(str(messages[-1].get("content") or ""))
        except json.JSONDecodeError as error:
            raise ValueError("Analyst received invalid chart data.") from error
        if not isinstance(value, dict) or not isinstance(
            value.get("case_id"), str
        ):
            raise ValueError("Analyst received invalid case chart data.")
        case_id = str(value["case_id"])
        if not CASE_ID_PATTERN.fullmatch(case_id):
            raise ValueError("Analyst received an invalid case ID.")
        validated = _validate_risk_data(value.get("risk_data", []))
        return True, _chart_source(validated, case_id)


def chart_risk_data(email_text: str) -> list[dict[str, Any]]:
    """Recompute sender and link scores using the offline rule-based tools."""
    headers = extract_headers(email_text)
    from_header_value = headers.get("From") or extract_sender(email_text)
    from_header = (
        str(from_header_value[-1])
        if isinstance(from_header_value, list) and from_header_value
        else str(from_header_value)
    )
    reply_value = headers.get("Reply-To")
    reply_to = (
        str(reply_value[-1])
        if isinstance(reply_value, list) and reply_value
        else str(reply_value) if reply_value else None
    )
    sender_result = check_sender_domain(from_header, reply_to)
    data = [
        {"label": "Sender domain", "score": sender_result.risk_score}
    ]
    for index, link in enumerate(extract_links(email_text), start=1):
        result = check_link(LinkInput(url=link))
        data.append({"label": f"Link {index}", "score": result.risk_score})
    return data


def run_analyst_chart(
    case_id: str,
    email_text: str,
    no_docker: bool = False,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    docker_check: Callable[[], bool] | None = None,
) -> Path | None:
    """Create a chart, or clearly skip it if the selected executor is unavailable."""
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("The case has an invalid case_id.")
    is_docker_available = (docker_check or docker_is_available)()
    if not is_docker_available and not no_docker:
        print(
            "Analyst chart skipped: Docker is unavailable. "
            "Use --no-docker to opt into local code execution."
        )
        return None
    if no_docker:
        if is_docker_available:
            print("Analyst is using LocalCommandLineCodeExecutor (--no-docker).")
        else:
            print(
                "Docker is unavailable; --no-docker permits the local executor."
            )

    reports_path = Path(reports_dir)
    work_dir = reports_path / ANALYST_WORK_DIR_NAME
    case_work_dir = work_dir / case_id
    case_work_dir.mkdir(parents=True, exist_ok=True)
    risk_data = chart_risk_data(email_text)
    code = Analyst().generate_code(risk_data, case_id)
    chart_in_workdir = case_work_dir / CHART_OUTPUT_NAME
    chart_path = reports_path / f"{case_id}_chart.png"

    if no_docker:
        result = _run_local_chart(code, work_dir)
    else:
        result = _run_docker_chart(code, work_dir)
    if result is None:
        return None
    if result.exit_code != 0:
        detail = normalize_text(result.output.strip()) or "executor returned an error"
        print(f"Analyst chart skipped: {detail}")
        return None
    if not chart_in_workdir.is_file():
        print("Analyst chart skipped: executor did not create its chart file.")
        return None

    try:
        shutil.copyfile(chart_in_workdir, chart_path)
    except OSError as error:
        print(f"Analyst chart skipped: could not copy the chart ({error}).")
        return None
    print(f"Analyst chart saved: reports\\{chart_path.name}")
    if no_docker:
        print("Chart created via local fallback.")
    else:
        print("Chart created via Docker.")
    return chart_path


def docker_is_available(timeout: int = 15) -> bool:
    """Check that the Docker CLI and daemon respond without displaying details."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _run_docker_chart(code: str, work_dir: Path) -> Any | None:
    """Set up Matplotlib and run only the fixed chart script in Docker."""
    executor: DockerCommandLineCodeExecutor | None = None
    try:
        executor = DockerCommandLineCodeExecutor(
            image="python:3.12-slim",
            timeout=600,
            work_dir=work_dir,
            auto_remove=True,
        )
        if not _matplotlib_installed(work_dir):
            setup = executor.execute_code_blocks(
                [CodeBlock(code=CHART_SETUP_CODE, language="sh")]
            )
            if setup.exit_code != 0:
                detail = normalize_text(setup.output.strip())[-2000:]
                print(
                    "Analyst chart skipped: Docker could not install Matplotlib "
                    f"in the container. {detail}"
                )
                return None
            if not _verify_and_mark_matplotlib(executor):
                return None
        return executor.execute_code_blocks(
            [CodeBlock(code=code, language="python")]
        )
    except (DockerException, OSError, RuntimeError, ValueError) as error:
        print(f"Analyst chart skipped: Docker executor failed ({error}).")
        return None
    finally:
        if executor is not None:
            try:
                executor.stop()
            except (DockerException, OSError) as error:
                print(f"Analyst Docker cleanup warning: {error}.")


def _run_local_chart(code: str, work_dir: Path) -> Any | None:
    """Run the same restricted chart code with the user-opted local executor."""
    original_path = os.environ.get("PATH")
    system_python = _system_python_with_matplotlib()
    python_executable = system_python or Path(sys.executable)
    os.environ["PATH"] = (
        str(python_executable.parent)
        + os.pathsep
        + (original_path or "")
    )
    try:
        executor = LocalCommandLineCodeExecutor(timeout=180, work_dir=work_dir)
        if not _matplotlib_installed(work_dir) and system_python is None:
            setup = executor.execute_code_blocks(
                [CodeBlock(code=LOCAL_CHART_SETUP_CODE, language="sh")]
            )
            if setup.exit_code != 0:
                print(
                    "Analyst chart skipped: local Matplotlib setup failed. "
                    + normalize_text(setup.output.strip())
                )
                return None
            if not _verify_and_mark_matplotlib(executor):
                return None
        return executor.execute_code_blocks(
            [CodeBlock(code=code, language="python")]
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Analyst chart skipped: local executor failed ({error}).")
        return None
    finally:
        if original_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original_path


def _system_python_with_matplotlib() -> Path | None:
    """Find an installed Python 3.12 runtime that already includes Matplotlib."""
    try:
        result = subprocess.run(
            [
                "py",
                "-3.12",
                "-c",
                (
                    "import importlib.util, sys; "
                    "print(sys.executable if "
                    "importlib.util.find_spec('matplotlib') else '')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    executable = Path(result.stdout.strip())
    return executable if executable.is_file() else None


def _validate_risk_data(value: list[Any]) -> list[dict[str, Any]]:
    """Accept only short labels and integer scores from zero to one hundred."""
    validated: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each Analyst chart entry must be an object.")
        label = item.get("label")
        score = item.get("score")
        if (
            not isinstance(label, str)
            or not label
            or len(label) > 40
            or not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 100
        ):
            raise ValueError("Analyst chart entries have invalid labels or scores.")
        validated.append({"label": normalize_text(label), "score": score})
    if not validated:
        raise ValueError("Analyst needs at least one risk score to chart.")
    return validated


def _matplotlib_installed(work_dir: Path) -> bool:
    """Check for the isolated dependency files installed under the work dir."""
    packages_dir = work_dir / ANALYST_PACKAGES_DIR
    return (
        (work_dir / MATPLOTLIB_SETUP_MARKER).is_file()
        and (packages_dir / "matplotlib" / "__init__.py").is_file()
        and (packages_dir / "numpy" / "__init__.py").is_file()
    )


def _verify_and_mark_matplotlib(executor: Any) -> bool:
    """Validate the isolated install and mark it reusable for later cases."""
    verify = executor.execute_code_blocks(
        [
            CodeBlock(
                code=(
                    "import sys\n"
                    "sys.path.insert(0, 'analyst_packages')\n"
                    "import matplotlib.pyplot\n"
                    "import numpy"
                ),
                language="python",
            )
        ]
    )
    if verify.exit_code != 0:
        detail = normalize_text(verify.output.strip())[-2000:]
        print(f"Analyst chart skipped: Matplotlib setup is incomplete. {detail}")
        return False

    marker = executor.execute_code_blocks(
        [
            CodeBlock(
                code=(
                    "from pathlib import Path\n"
                    f"Path({MATPLOTLIB_SETUP_MARKER!r}).touch()"
                ),
                language="python",
            )
        ]
    )
    if marker.exit_code != 0:
        print("Analyst chart skipped: could not verify the Matplotlib setup.")
        return False
    return True


def _chart_source(risk_data: list[dict[str, Any]], case_id: str) -> str:
    """Create the only Python code the Analyst is allowed to execute."""
    encoded_data = repr(json.dumps(risk_data, ensure_ascii=True))
    return "\n".join(
        (
            "import json",
            "import os",
            "import sys",
            f"case_id = {case_id!r}",
            f"packages_dir = os.path.abspath({ANALYST_PACKAGES_DIR!r})",
            "sys.path.insert(0, packages_dir)",
            "os.chdir(case_id)",
            "os.environ['MPLCONFIGDIR'] = '.'",
            "import matplotlib",
            "matplotlib.use('Agg')",
            "import matplotlib.pyplot as plt",
            f"risk_data = json.loads({encoded_data})",
            "labels = [item['label'] for item in risk_data]",
            "scores = [item['score'] for item in risk_data]",
            "width = max(6, len(labels) * 1.2)",
            "fig, axis = plt.subplots(figsize=(width, 4))",
            "bars = axis.bar(labels, scores, color='#3976a8')",
            "axis.set_ylim(0, 100)",
            "axis.set_ylabel('Risk score (0-100)')",
            "axis.set_title('Email risk indicators')",
            "axis.tick_params(axis='x', labelrotation=15)",
            "for bar, score in zip(bars, scores):",
            "    axis.text(bar.get_x() + bar.get_width() / 2, score + 2, str(score), ha='center')",
            "fig.tight_layout()",
            f"fig.savefig({CHART_OUTPUT_NAME!r}, dpi=150)",
            "plt.close(fig)",
        )
    )
