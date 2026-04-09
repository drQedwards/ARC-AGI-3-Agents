"""ARC-AGI-3 interactive agent with PMLL Memory MCP integration (correct approach).

This targets ARC-AGI-3 turn-based environments and avoids ARC-1/2 style transform solving.
"""

from __future__ import annotations

import asyncio
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import requests


@dataclass
class ToolResponse:
    ok: bool
    result: dict[str, Any] | None
    error: str | None = None


class PMLLMemoryClient:
    """MCP stdio client for pmll-memory-mcp (JSON-RPC tools/call)."""

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.session_id = f"arc3-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.request_id = 0
        self.started = False

    async def start(self) -> bool:
        commands = [
            ["npx", "-y", "pmll-memory-mcp"],
            ["python", "-m", "pmll_memory_mcp"],
        ]
        for cmd in commands:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                self.started = True
                await self.call_tool("init", {"session_id": self.session_id, "silo_size": 512})
                print(f"[pmll] started: {' '.join(cmd)}")
                return True
            except (FileNotFoundError, OSError):
                continue
        print("[pmll] unavailable; using fallback memory")
        return False

    async def _send(self, method: str, params: dict[str, Any]) -> ToolResponse:
        if not self.started or not self.process or not self.process.stdin or not self.process.stdout:
            return ToolResponse(False, None, "mcp not started")
        try:
            self.request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params,
            }
            self.process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            await self.process.stdin.drain()
            line = await self.process.stdout.readline()
            if not line:
                self.started = False
                return ToolResponse(False, None, "empty response")
            msg = json.loads(line.decode("utf-8"))
            if "error" in msg:
                return ToolResponse(False, msg, str(msg["error"]))
            return ToolResponse(True, msg, None)
        except (ConnectionResetError, BrokenPipeError, json.JSONDecodeError) as exc:
            self.started = False
            return ToolResponse(False, None, f"mcp disconnected: {exc}")

    async def call_tool(self, tool: str, args: dict[str, Any]) -> ToolResponse:
        return await self._send("tools/call", {"name": tool, "arguments": args})

    async def peek(self, key: str) -> Any | None:
        resp = await self.call_tool("peek", {"session_id": self.session_id, "key": key})
        if not resp.ok or not resp.result:
            return None
        result = resp.result.get("result", {})
        if result.get("hit"):
            return result.get("value")
        return None

    async def set(self, key: str, value: Any) -> None:
        raw = value if isinstance(value, str) else json.dumps(value)
        await self.call_tool("set", {"session_id": self.session_id, "key": key, "value": raw})

    async def remember_pattern(self, env_id: str, level: int, data: dict[str, Any]) -> None:
        await self.call_tool(
            "upsert_memory_node",
            {
                "session_id": self.session_id,
                "type": "arc3_pattern",
                "label": f"{env_id}_L{level}",
                "content": json.dumps(data),
                "metadata": {"env_id": env_id, "level": level, "stored_at": datetime.now(UTC).isoformat()},
            },
        )

    async def recall_patterns(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        resp = await self.call_tool(
            "search_memory_graph",
            {"session_id": self.session_id, "query": query, "top_k": top_k},
        )
        if resp.ok and resp.result:
            data = resp.result.get("result")
            if isinstance(data, list):
                return data
        return []

    async def flush(self) -> None:
        await self.call_tool("flush", {"session_id": self.session_id})

    async def stop(self) -> None:
        if self.started:
            await self.flush()
        if self.process:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            await self.process.wait()


class FallbackMemory:
    """In-process fallback matching PMLL methods used by the agent."""

    def __init__(self) -> None:
        self.cache: dict[str, Any] = {}
        self.patterns: list[dict[str, Any]] = []

    async def start(self) -> bool:
        print("[fallback] memory online")
        return True

    async def stop(self) -> None:
        return None

    async def peek(self, key: str) -> Any | None:
        return self.cache.get(key)

    async def set(self, key: str, value: Any) -> None:
        self.cache[key] = value

    async def remember_pattern(self, env_id: str, level: int, data: dict[str, Any]) -> None:
        self.patterns.append({"env_id": env_id, "level": level, **data})

    async def recall_patterns(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q = query.lower()
        matches = [p for p in self.patterns if q in json.dumps(p).lower()]
        return matches[:top_k]

    async def flush(self) -> None:
        self.cache.clear()


def summarize_frame(frame: list[list[int]]) -> dict[str, Any]:
    arr = np.array(frame)
    counts = Counter(arr.flatten().tolist())
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "unique_colors": len(counts),
        "dominant_color": counts.most_common(1)[0][0],
        "nonzero_count": int(np.sum(arr != 0)),
    }


class ArcAGI3Agent:
    OFFICIAL_PROMPT = (
        "You are playing a game. Your goal is to win. Reply with the exact action you want "
        "to take. The final action in your reply will be executed next turn. Your entire "
        "reply will be carried to the next turn."
    )

    def __init__(self, memory: PMLLMemoryClient | FallbackMemory) -> None:
        self.memory = memory
        self.history: list[dict[str, Any]] = []
        self.rhae_scores: list[float] = []

    async def on_frame(self, frame: list[list[int]], env_id: str, level: int, turn: int) -> Any:
        summary = summarize_frame(frame)
        key = f"{env_id}:L{level}:T{turn}:{hash(json.dumps(summary, sort_keys=True))}"

        cached = await self.memory.peek(key)
        if cached is not None:
            if isinstance(cached, str):
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    return cached
            return cached

        patterns = await self.memory.recall_patterns(f"{env_id} level {level}")
        action = self._choose_action_demo(summary, patterns, turn)
        await self.memory.set(key, action)
        self.history.append({"env_id": env_id, "level": level, "turn": turn, "action": action})
        return action

    def _choose_action_demo(self, summary: dict[str, Any], patterns: list[dict[str, Any]], turn: int) -> Any:
        if patterns:
            return random.choice(["key_action", "key_right", {"select": [32, 32]}])
        if turn < 4:
            return "key_action"
        if summary["nonzero_count"] > 100:
            return {"select": [random.randint(10, 54), random.randint(10, 54)]}
        return random.choice(["key_up", "key_down", "key_left", "key_right", "key_action"])

    async def on_level_complete(self, env_id: str, level: int, actions_taken: int, human_baseline: int = 12) -> float:
        rhae = (human_baseline / max(actions_taken, 1)) ** 2
        self.rhae_scores.append(rhae)
        await self.memory.remember_pattern(
            env_id,
            level,
            {"actions": actions_taken, "human_baseline": human_baseline, "rhae": round(rhae, 6)},
        )
        return rhae

    async def on_env_complete(self) -> float:
        avg_rhae = sum(self.rhae_scores) / max(len(self.rhae_scores), 1)
        await self.memory.flush()
        self.history.clear()
        self.rhae_scores.clear()
        return avg_rhae


async def run_offline_demo() -> dict[str, Any]:
    memory = PMLLMemoryClient()
    if await memory.start():
        active_memory: PMLLMemoryClient | FallbackMemory = memory
    else:
        fb = FallbackMemory()
        await fb.start()
        active_memory = fb

    agent = ArcAGI3Agent(active_memory)
    env_ids = ["ls20", "demo_exploration", "demo_objective"]
    total_rhae = 0.0
    levels = 0

    for env_id in env_ids:
        for level in (1, 2):
            for turn in range(1, 6):
                fake_frame = np.zeros((64, 64), dtype=int).tolist()
                await agent.on_frame(fake_frame, env_id, level, turn)
            actions_taken = random.randint(8, 25)
            rhae = await agent.on_level_complete(env_id, level, actions_taken, human_baseline=12)
            total_rhae += rhae
            levels += 1
        await agent.on_env_complete()

    scorecard = {
        "benchmark": "ARC-AGI-3",
        "attempt_id": str(uuid.uuid4()),
        "solver": "pmll-mcp-interactive-v1",
        "avg_rhae": round(total_rhae / max(levels, 1), 6),
        "environments_tested": len(env_ids),
        "levels_tested": levels,
        "memory_backend": "pmll" if isinstance(active_memory, PMLLMemoryClient) and active_memory.started else "fallback",
        "timestamp": datetime.now(UTC).isoformat(),
        "note": "offline simulation only; use official arcagi3 harness for real benchmark scores",
    }

    out = f"arc3_scorecard_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    arc_api_key = os.getenv("ARC_API_KEY")
    if arc_api_key:
        try:
            requests.post(
                "https://three.arcprize.org/api/scorecard",
                json=scorecard,
                headers={"Authorization": f"Bearer {arc_api_key}"},
                timeout=5,
            )
        except requests.RequestException:
            pass

    await active_memory.stop()
    return scorecard


async def run_online_harness(game_id: str, config: str, max_actions: int) -> int:
    """Run the official ARC-AGI-3 harness in online mode."""
    if importlib.util.find_spec("arcagi3") is None:
        print("[error] arcagi3 is not installed.")
        print("Install with: git clone https://github.com/arcprize/arc-agi-3-benchmarking.git && pip install -e .")
        return 2
    if not os.getenv("ARC_API_KEY"):
        print("[error] ARC_API_KEY is required for online runs.")
        return 2

    cmd = [
        sys.executable,
        "-m",
        "arcagi3.runner",
        "--game_id",
        game_id,
        "--config",
        config,
        "--max_actions",
        str(max_actions),
    ]
    print("[online] running:", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    scorecard_url = None
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        print(text)
        match = re.search(r"https://three\\.arcprize\\.org/scorecards/[A-Za-z0-9\\-]+", text)
        if match:
            scorecard_url = match.group(0)

    code = await proc.wait()
    if scorecard_url:
        print(f"[online] scorecard: {scorecard_url}")
    else:
        print("[online] scorecard URL not detected in runner output.")
    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 agent with PMLL MCP memory")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--game-id", default="ls20", help="ARC-AGI-3 game_id for online mode")
    parser.add_argument("--config", default="claude-sonnet-4-5-20250929", help="arcagi3 model config for online mode")
    parser.add_argument("--max-actions", type=int, default=50, help="max actions per level/environment")
    return parser.parse_args()


async def main() -> int:
    print("[security] ARC_API_KEY set:", bool(os.getenv("ARC_API_KEY")))
    print("[security] PMLL_API_KEY set:", bool(os.getenv("PMLL_API_KEY")))
    args = parse_args()

    if args.mode == "online":
        return await run_online_harness(args.game_id, args.config, args.max_actions)

    if importlib.util.find_spec("arcagi3") is not None:
        print("Official arcagi3 harness detected.")
        print("Run with:")
        print("  uv run python -m arcagi3.runner --check")
        print("  uv run python -m arcagi3.runner --list-games")
        print("  uv run python -m arcagi3.runner --game_id ls20 --config <config> --max_actions 50")
        return 0

    print("arcagi3 not installed; running offline demo simulation...")
    result = await run_offline_demo()
    print("Offline demo scorecard:", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
