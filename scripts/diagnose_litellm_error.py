#!/usr/bin/env python
"""Diagnostic: find the exact exception in litellm.acompletion()."""

from __future__ import annotations

import asyncio
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from hannah.config.loader import load_config
from hannah.providers.litellm_provider import LiteLLMProvider

cfg = load_config()
provider = LiteLLMProvider(config=cfg)

print(f"Model: {cfg.agent.model}")
print(f"_force_local: {provider._force_local()}")
print(f"_hosted_credentials_available: {provider._hosted_credentials_available()}")
print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
print()


async def test():
    import litellm

    litellm.suppress_debug_info = True

    messages = [{"role": "user", "content": "Say hello in one word."}]

    try:
        result = await litellm.acompletion(
            model=cfg.agent.model,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        print(f"SUCCESS: {result.choices[0].message.content}")
    except Exception as exc:
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        print()
        print("Full traceback:")
        traceback.print_exc()


asyncio.run(test())
