"""Focused tests for email intake and offline analysis tools."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import validate_api_keys
from src.intake import load_email
from src.tools import (
    check_auth_results,
    check_sender_domain,
    compare_link_domains,
    extract_headers,
)


class IntakeTests(unittest.TestCase):
    """Verify each supported input path and its validation behavior."""

    def test_loads_text_and_eml_file(self) -> None:
        text = "From: reader@example.test\nSubject: Update\n\nHello"
        self.assertEqual(load_email(text=text), text)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message.eml"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(load_email(file=path), text)

    def test_requires_exactly_one_nonempty_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_email()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            load_email(text="message", file="message.txt")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            load_email(text="  ")

    def test_uses_groq_vision_transcription(self) -> None:
        transcript = "From: sender@example.test\nSubject: Sample\nBody text"
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=transcript))
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "screenshot.png"
            image_path.write_bytes(b"synthetic-image-bytes")
            with (
                patch("src.intake.GROQ_API_KEY", "test-key"),
                patch("src.intake.OpenAI") as openai_client,
            ):
                openai_client.return_value.chat.completions.create.return_value = (
                    response
                )
                self.assertEqual(load_email(image=image_path), transcript)
                request = (
                    openai_client.return_value.chat.completions.create.call_args.kwargs
                )
                self.assertEqual(request["temperature"], 0)
                self.assertIn("image_url", request["messages"][0]["content"][1])

    def test_rejects_unsupported_and_oversized_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.gif"
            path.write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "Unsupported image type"):
                load_email(image=path)

            oversized_path = Path(directory) / "large.png"
            oversized_path.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(ValueError, "10 MiB"):
                load_email(image=oversized_path)


class OfflineToolTests(unittest.TestCase):
    """Verify selected header, sender-domain, and link-domain indicators."""

    def test_extracts_selected_headers_and_authentication_results(self) -> None:
        text = (
            "From: Example <sender@example.test>\n"
            "Reply-To: reply@other.test\n"
            "Return-Path: <bounce@example.test>\n"
            "Received: first-hop\n"
            "Received: second-hop\n"
            "Authentication-Results: mx.test; spf=pass dkim=fail dmarc=pass\n"
            "\nBody\n"
        )
        headers = extract_headers(text)
        self.assertEqual(headers["From"], "Example <sender@example.test>")
        self.assertEqual(headers["Received"], ["first-hop", "second-hop"])
        self.assertEqual(
            check_auth_results(headers),
            ["SPF: pass", "DKIM: fail", "DMARC: pass"],
        )

    def test_flags_sender_mismatches_and_unrelated_link_domains(self) -> None:
        result = check_sender_domain(
            "PayPal Support <alerts@paypa1.xyz>",
            "reply@unrelated.test",
        )
        self.assertEqual(result.domain, "paypa1.xyz")
        self.assertGreaterEqual(result.risk_score, 90)
        self.assertTrue(
            any("lookalike" in reason for reason in result.reasons)
        )

        findings = compare_link_domains(
            "example.test",
            ["https://outside.invalid/path", "https://mail.example.test/help"],
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("outside.invalid", findings[0])


class ConfigurationTests(unittest.TestCase):
    """Verify Groq is required while the OpenAI fallback remains optional."""

    def test_only_requires_groq_key(self) -> None:
        with patch("src.config.GROQ_API_KEY", "test-key"):
            validate_api_keys()

        with patch("src.config.GROQ_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
                validate_api_keys()


if __name__ == "__main__":
    unittest.main()
