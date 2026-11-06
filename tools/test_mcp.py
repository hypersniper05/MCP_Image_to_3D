"""Check the MCP endpoint with both protocol eras, the way real clients differ.

  python tools/test_mcp.py                       # http://127.0.0.1:8043/mcp
  python tools/test_mcp.py --url http://127.0.0.1:8043/mcp --call idol_status

Verifies that the tool list is complete and that a no-argument tool answers, in the
modern (2026-07-28) and the legacy (2025-11-25) dialect. Prints PASS/FAIL.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import Client

EXPECT = {"idol_generate_3d", "idol_job", "idol_objects", "idol_status"}


async def check(url: str, mode: str, call: str, args: dict) -> bool:
    options = {"mode": "legacy"} if mode == "legacy" else {}
    async with Client(url, **options) as client:
        tools = (await client.list_tools()).tools
        names = {t.name for t in tools}
        print(f"[{mode}] protocol {client.protocol_version}: {len(tools)} tools -> {', '.join(sorted(names))}")
        missing = EXPECT - names
        if missing:
            print(f"[{mode}] MISSING: {sorted(missing)}")
            return False
        if not call:
            return True
        result = await client.call_tool(call, args)
        text = " ".join(getattr(b, "text", "") for b in result.content)
        sc = result.structured_content
        print(f"[{mode}] {call} -> is_error={result.is_error}")
        if sc:
            keys = list(sc.get("result", sc).keys()) if isinstance(sc, dict) else []
            print(f"[{mode}]   fields: {keys}")
        if result.is_error:
            print(f"[{mode}]   error text: {text[:300]}")
        return not result.is_error


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8043/mcp")
    ap.add_argument("--call", default="idol_status")
    ap.add_argument("--args", default="{}")
    a = ap.parse_args()
    args = json.loads(a.args)

    ok = True
    for mode in ("modern", "legacy"):
        try:
            ok &= await check(a.url, mode, a.call, args)
        except Exception as exc:
            print(f"[{mode}] FAILED: {type(exc).__name__}: {exc}")
            ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
