#!/usr/bin/env python
"""Diagnostic script: simulate exactly what `hannah agent` does interactively.

Run from the repo root:
    .venv/bin/python scripts/diagnose_provider.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

print("=" * 60)
print("Hannah Provider Diagnostic")
print("=" * 60)

# Step 1: Check CWD and .env location
cwd = Path.cwd()
env_file = cwd / ".env"
print(f"\n1. Working directory: {cwd}")
print(f"   .env exists: {env_file.exists()}")

# Step 2: Check env BEFORE load_dotenv
print(f"\n2. BEFORE load_dotenv():")
print(f"   HANNAH_MODEL: {os.getenv('HANNAH_MODEL', '<NOT SET>')}")
print(f"   OPENAI_API_KEY: {'<SET>' if os.getenv('OPENAI_API_KEY') else '<NOT SET>'}")
print(
    f"   ANTHROPIC_API_KEY: {'<SET>' if os.getenv('ANTHROPIC_API_KEY') else '<NOT SET>'}"
)
print(
    f"   HANNAH_FORCE_LOCAL_PROVIDER: {os.getenv('HANNAH_FORCE_LOCAL_PROVIDER', '<NOT SET>')}"
)

# Step 3: Load dotenv the same way app.py does
try:
    from dotenv import load_dotenv

    result = load_dotenv(override=True)
    print(f"\n3. load_dotenv(override=True) returned: {result}")
except Exception as exc:
    print(f"\n3. load_dotenv() FAILED: {exc}")

# Step 4: Check env AFTER load_dotenv
print(f"\n4. AFTER load_dotenv():")
print(f"   HANNAH_MODEL: {os.getenv('HANNAH_MODEL', '<NOT SET>')}")
print(f"   OPENAI_API_KEY: {'<SET>' if os.getenv('OPENAI_API_KEY') else '<NOT SET>'}")
print(
    f"   ANTHROPIC_API_KEY: {'<SET>' if os.getenv('ANTHROPIC_API_KEY') else '<NOT SET>'}"
)
print(
    f"   HANNAH_FORCE_LOCAL_PROVIDER: {os.getenv('HANNAH_FORCE_LOCAL_PROVIDER', '<NOT SET>')}"
)

# Step 5: Load config
from hannah.config.loader import load_config
from hannah.config.provider_setup import detect_provider_from_model, get_provider_preset

cfg = load_config()
print(f"\n5. Config loaded:")
print(f"   agent.model: {cfg.agent.model}")
print(f"   rlm.enabled: {cfg.rlm.enabled}")
print(f"   rlm.api_base: {cfg.rlm.api_base}")

# Step 6: Provider detection
provider_name = detect_provider_from_model(cfg.agent.model)
print(f"\n6. Provider detection:")
print(f"   detect_provider_from_model('{cfg.agent.model}'): {provider_name}")

if provider_name:
    preset = get_provider_preset(provider_name)
    print(f"   Preset api_key_env_vars: {preset.api_key_env_vars}")
    for key in preset.api_key_env_vars:
        val = os.getenv(key)
        if val:
            print(f"   {key}: <SET> ({val[:12]}...)")
        else:
            print(f"   {key}: <NOT SET>")

# Step 7: LiteLLMProvider checks
from hannah.providers.litellm_provider import LiteLLMProvider

provider = LiteLLMProvider(config=cfg)
print(f"\n7. LiteLLMProvider checks:")
print(f"   _force_local(): {provider._force_local()}")
print(f"   _hosted_credentials_available(): {provider._hosted_credentials_available()}")

# Step 8: Try a direct call
import asyncio


async def test_provider():
    from hannah.providers.local_fallback import LocalCompletion

    result = await provider.complete(
        messages=[{"role": "user", "content": "Say hello in one word."}],
        tools=None,
        temperature=0.0,
        max_tokens=10,
    )

    if isinstance(result, LocalCompletion):
        print(f"\n8. RESULT: LocalCompletion (hosted call was BYPASSED)")
        print(
            f"   This means _force_local() or _hosted_credentials_available() returned False"
        )
    else:
        text = (
            result.choices[0].message.content
            if hasattr(result, "choices")
            else "<unknown>"
        )
        print(f"\n8. RESULT: Hosted model responded: {text}")


asyncio.run(test_provider())

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
