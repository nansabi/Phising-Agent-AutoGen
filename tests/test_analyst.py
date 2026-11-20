"""API- and Docker-free tests for safe Analyst chart behavior."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.analyst import (
    ANALYST_WORK_DIR_NAME,
    CHART_OUTPUT_NAME,
    CHART_SETUP_CODE,
    _chart_source,
    _validate_risk_data,
    run_analyst_chart,
)


class AnalystTests(unittest.TestCase):
    """Exercise chart skipping and data-only plotting code."""

    def test_missing_docker_skips_chart_unless_user_opts_into_local_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            reports_dir = Path(temporary_directory) / "reports"

            result = run_analyst_chart(
                "20261120T045724208818Z-82539024",
                "From: sender@example.test\n\nHello",
                reports_dir=reports_dir,
                docker_check=lambda: False,
            )

            self.assertIsNone(result)
            self.assertFalse(reports_dir.exists())

    def test_chart_code_contains_only_validated_risk_data(self) -> None:
        risk_data = _validate_risk_data(
            [{"label": "Sender domain", "score": 35}]
        )

        source = _chart_source(
            risk_data,
            "20261120T045724208818Z-82539024",
        )

        self.assertIn("axis.bar(labels, scores", source)
        self.assertIn('"score": 35', source)
        self.assertNotIn("requests.", source)
        self.assertNotIn("open(", source)

    def test_invalid_risk_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_risk_data([{"label": "Link", "score": 101}])

    def test_docker_matplotlib_setup_retries_with_long_timeout(self) -> None:
        self.assertIn("--default-timeout=120", CHART_SETUP_CODE)
        self.assertIn("--retries 3", CHART_SETUP_CODE)

    def test_chart_reports_docker_executor(self) -> None:
        case_id = "20261120T045724208818Z-82539024"
        with tempfile.TemporaryDirectory() as temporary_directory:
            reports_dir = Path(temporary_directory) / "reports"
            work_dir = reports_dir / ANALYST_WORK_DIR_NAME

            def create_chart(_code: str, executor_work_dir: Path) -> object:
                output_path = executor_work_dir / case_id / CHART_OUTPUT_NAME
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"test chart")
                return SimpleNamespace(exit_code=0, output="")

            output = io.StringIO()
            with (
                patch("src.analyst._run_docker_chart", side_effect=create_chart),
                redirect_stdout(output),
            ):
                chart_path = run_analyst_chart(
                    case_id,
                    "From: sender@example.test\n\nHello",
                    reports_dir=reports_dir,
                    docker_check=lambda: True,
                )

        self.assertIsNotNone(chart_path)
        self.assertIn("Chart created via Docker.", output.getvalue())

    def test_chart_reports_explicit_local_fallback(self) -> None:
        case_id = "20261120T045724208818Z-82539024"
        with tempfile.TemporaryDirectory() as temporary_directory:
            reports_dir = Path(temporary_directory) / "reports"

            def create_chart(_code: str, executor_work_dir: Path) -> object:
                output_path = executor_work_dir / case_id / CHART_OUTPUT_NAME
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"test chart")
                return SimpleNamespace(exit_code=0, output="")

            output = io.StringIO()
            with (
                patch("src.analyst._run_local_chart", side_effect=create_chart),
                redirect_stdout(output),
            ):
                chart_path = run_analyst_chart(
                    case_id,
                    "From: sender@example.test\n\nHello",
                    no_docker=True,
                    reports_dir=reports_dir,
                    docker_check=lambda: True,
                )

        self.assertIsNotNone(chart_path)
        self.assertIn("Chart created via local fallback.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
