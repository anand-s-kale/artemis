"""Jev-driven device loop: Jev decides, a VLM (optional) only *supplies*.

Each step: observe -> Jev picks the action, and says whether the goal is met
or the run is stuck -> act. The loop stops when Jev says the goal is met, when
Jev reports being stuck, on a repeated action, or at the step budget. No LLM
key is needed for the drive itself.

A ``Supplier`` (typically a VLM, or the host model) is consulted only for
things Jev cannot produce:
  * ``supply_text``: the literal text to type into a field Jev chose.
  * ``describe_screen``: a short text description when the element list gives
    Jev too little to be confident (e.g. canvas / image-heavy screens).
It never picks actions and never judges completion.

Run against a connected phone (uses the PR #114 manual-mode session, so no
Artemis LLM credential is required)::

    ARTEMIS_JEV_DRIVER=1 python -m artemis.llm.jev_driver "open Gallery, then go home"
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from artemis.llm.jev_adapter import JevDecision, JevFlashModel

_LINE = re.compile(
    r"^\[(\d+)\]\s+([\w ]+?):\s*'(.*)'\s*\|\s*Bounds:\s*\[(\d+),(\d+)\]\[(\d+),(\d+)\]"
)
_STATUS_BAR_BOTTOM = 45  # normalized 0-1000 y; status-bar icons live above this
_EDITABLE_KINDS = ("input", "edittext", "textfield", "text field")


def parse_elements_text(elements_text: str) -> list[dict[str, Any]]:
    """Parses Artemis's ``[N] Text: '...' | Bounds: [..][..]`` lines.

    Keeps each element's original index (that is what ``click(target=N)``
    resolves against) and drops status-bar noise and empty labels.
    """
    out: list[dict[str, Any]] = []
    for line in (elements_text or "").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        idx, kind, text, _x1, _y1, _x2, y2 = m.groups()
        if int(y2) <= _STATUS_BAR_BOTTOM or not text.strip():
            continue
        out.append(
            {
                "index": int(idx),
                "text": text,
                "kind": kind,
                "editable": kind.strip().lower() in _EDITABLE_KINDS,
            }
        )
    return out


def split_keyboard(elements: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Drops on-screen keyboard keys (many 1-char labels) and reports if it is open.

    Artemis lists every key as an element; left in, they bury the real screen and
    tempt the model into tapping letters instead of typing.
    """
    keys = [e for e in elements if len(e["text"].strip()) <= 1 and e["index"] > 0]
    if len(keys) < 8:
        return elements, False
    drop = {id(e) for e in keys}
    return [e for e in elements if id(e) not in drop], True


class Supplier(Protocol):
    async def supply_text(self, goal: str, field_label: str, history: list[str]) -> str | None: ...

    async def describe_screen(self, screenshot_path: str | None) -> str | None: ...


class QuotedTextSupplier:
    """Key-free supplier: the text to type is whatever the goal puts in quotes."""

    async def supply_text(self, goal: str, field_label: str, history: list[str]) -> str | None:
        m = re.search(r"[\"“'‘]([^\"”'’]+)[\"”'’]", goal)
        return m.group(1) if m else None

    async def describe_screen(self, screenshot_path: str | None) -> str | None:
        return None


@dataclass
class Observation:
    elements_text: str
    screenshot: str | None = None


@dataclass
class DriveResult:
    status: str  # "success" | "failed"
    reason: str
    steps: list[dict[str, Any]] = field(default_factory=list)


ObserveFn = Callable[[], Awaitable[Observation]]
ActFn = Callable[[str, dict[str, Any]], Awaitable[Observation]]


class JevDriver:
    def __init__(
        self,
        observe: ObserveFn,
        act: ActFn,
        jev: JevFlashModel | None = None,
        supplier: Supplier | None = None,
        max_steps: int = 12,
        goal_threshold: float = 0.8,
        stuck_threshold: float = 0.7,
        repeat_limit: int = 3,
        settle_threshold: float = 0.7,
        apps: list[str] | None = None,
    ):
        self.apps = apps or []
        self.observe, self.act = observe, act
        self.jev = jev or JevFlashModel(autonomous=True)
        self.supplier = supplier
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.stuck_threshold = stuck_threshold
        self.repeat_limit = repeat_limit
        self.settle_threshold = settle_threshold

    async def run(self, goal: str) -> DriveResult:
        history: list[str] = []
        steps: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        stuck_streak = 0
        pkg = await self.jev.pick_app(goal, self.apps) if self.apps else None
        if pkg:  # straight to the app over ADB: no screenshot/observe turns to find it
            history.append(f"manage_app launch {pkg}")
            steps.append({"step": 0, "action": f"launch {pkg}", "call": {"name": "manage_app"}})
            obs = await self.act("manage_app", {"action": "launch", "app_name": pkg})
        else:
            obs = await self.observe()

        for n in range(1, self.max_steps + 1):
            elements, keyboard = split_keyboard(parse_elements_text(obs.elements_text))
            d = await self.jev.decide(goal, elements, history, keyboard=keyboard)

            if d.tool_call is None and d.needs_text_for is None and d.escalate_reason:
                d = await self._retry_with_notes(goal, elements, history, obs, d)

            steps.append(_trace(n, d))
            if d.goal_prob >= self.goal_threshold:
                return DriveResult("success", f"Jev: goal met ({d.goal_prob:.2f})", steps)

            stuck_streak = stuck_streak + 1 if d.stuck_prob >= self.stuck_threshold else 0
            if stuck_streak >= 2:
                return DriveResult("failed", "Jev reports stuck twice in a row", steps)

            call = await self._resolve_call(goal, elements, history, d)
            if call is None:
                if d.goal_prob >= self.settle_threshold:  # nothing confident left to do
                    return DriveResult(
                        "success", f"Jev: no action left, goal {d.goal_prob:.2f}", steps
                    )
                return DriveResult("failed", d.escalate_reason or "no actionable decision", steps)

            # Same action on an unchanged screen is a loop; the same swipe on a new
            # page (launcher paging) is legitimate progress.
            key = f"{call['name']}:{call['args']}:{hash(obs.elements_text)}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= self.repeat_limit:
                return DriveResult("failed", f"action repeated {seen[key]}x: {key}", steps)

            history.append(f"{call['name']} {call['args']}")
            obs = await self.act(call["name"], call["args"])

        return DriveResult("failed", f"step budget ({self.max_steps}) exhausted", steps)

    async def _retry_with_notes(self, goal, elements, history, obs, d: JevDecision) -> JevDecision:
        """Low confidence: let the supplier describe the screen, then Jev decides again."""
        if self.supplier is None or "confidence" not in (d.escalate_reason or ""):
            return d
        notes = await self.supplier.describe_screen(obs.screenshot)
        return await self.jev.decide(goal, elements, history, notes=notes or "") if notes else d

    async def _resolve_call(self, goal, elements, history, d: JevDecision) -> dict | None:
        if d.tool_call is not None:
            return d.tool_call
        if d.needs_text_for is None or self.supplier is None:
            return None
        label = next(
            (e["text"] for e in elements if e["index"] == d.needs_text_for), "focused field"
        )
        text = await self.supplier.supply_text(goal, label, history)
        if not text:
            return None
        args = {"text": text}
        if d.needs_text_for > 0:  # 0 = type into whatever is focused
            args["target"] = d.needs_text_for
        return {"name": "input_text", "args": args}


def _trace(n: int, d: JevDecision) -> dict[str, Any]:
    return {
        "step": n,
        "action": d.action,
        "confidence": round(d.action_confidence, 2),
        "goal": round(d.goal_prob, 2),
        "stuck": round(d.stuck_prob, 2),
        "call": d.tool_call and {"name": d.tool_call["name"], "args": d.tool_call["args"]},
        "note": d.escalate_reason,
    }


# ---------------------------------------------------------------------------
# Real-device IO via the PR #114 manual-mode session (no Artemis LLM key).
# ---------------------------------------------------------------------------


def manual_mode_io(device_serial: str | None = None) -> tuple[ObserveFn, ActFn, Callable[[], Any]]:
    from mcp_server.tools import manual_mode as mm

    def fn(tool):  # FastMCP may wrap the coroutine; unwrap when it does
        return getattr(tool, "fn", tool)

    observe_tool, act_tool, end_tool = (
        fn(mm.mobile_observe),
        fn(mm.mobile_act),
        fn(mm.mobile_session_end),
    )

    async def observe() -> Observation:
        r = await observe_tool(device_serial)
        if r.get("status") != "success":
            raise RuntimeError(r.get("error", "observe failed"))
        return Observation(r["elements_text"], r.get("screenshot"))

    async def act(action: str, args: dict[str, Any]) -> Observation:
        r = await act_tool(action, args, device_serial)
        if r.get("status") not in ("success", "ok"):
            raise RuntimeError(f"{action} failed: {r.get('message') or r.get('error')}")
        text = r.get("elements_text")
        if not text:  # some actions return no list; re-observe for the next step
            return await observe()
        return Observation(text, r.get("screenshot"))

    return observe, act, lambda: end_tool(device_serial)


async def list_launchable_apps(serial: str | None = None) -> list[str]:
    """Launchable package names via ADB (one call, no screenshots)."""
    cmd = (
        ["adb"]
        + (["-s", serial] if serial else [])
        + [
            "shell",
            "cmd",
            "package",
            "query-activities",
            "--brief",
            "-a",
            "android.intent.action.MAIN",
            "-c",
            "android.intent.category.LAUNCHER",
        ]
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    out, _ = await proc.communicate()
    pkgs = re.findall(r"^\s+([\w.]+)/", out.decode(errors="ignore"), re.M)
    return sorted(set(pkgs))


async def _main(goal: str) -> int:
    observe, act, end = manual_mode_io()
    try:
        apps = await list_launchable_apps(os.environ.get("ANDROID_SERIAL"))
        result = await JevDriver(observe, act, apps=apps, supplier=QuotedTextSupplier()).run(goal)
    finally:
        end()
    for s in result.steps:
        print(s)
    print(f"\n{result.status.upper()}: {result.reason}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(" ".join(sys.argv[1:]) or "go to the home screen")))
