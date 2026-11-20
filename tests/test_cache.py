"""Check that the configured AutoGen LLM cache remains enabled."""

import unittest

from src.config import llm_config


class CacheConfigurationTests(unittest.TestCase):
    """Keep the deterministic cache seed required by Phase 1 and later phases."""

    def test_cache_seed_is_42(self) -> None:
        self.assertEqual(llm_config.get("cache_seed"), 42)


if __name__ == "__main__":
    unittest.main()
