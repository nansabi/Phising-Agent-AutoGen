"""Load model settings and define the primary, fallback, and vision models."""

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.8-27b",
)

# AutoGen tries Groq first and only configures OpenAI when its key is available.
config_list = [
    {
        "model": "openai/gpt-oss-120b",
        "api_type": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "price": [0.00015, 0.0006],
    },
]
if OPENAI_API_KEY:
    config_list.append(
        {
            "model": "gpt-4o-mini",
            "api_key": OPENAI_API_KEY,
        }
    )

# The fixed cache seed makes repeated requests for the same prompt reproducible
# when the model provider and prompt are unchanged.
llm_config = {
    "config_list": config_list,
    "temperature": 0,
    "cache_seed": 42,
}


def validate_api_keys() -> None:
    """Raise a clear error if the required Groq API key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "Missing required API key: GROQ_API_KEY. "
            "Copy .env.example to .env and add your keys."
        )
