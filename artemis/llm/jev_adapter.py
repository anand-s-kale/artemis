"""Jev (TypeSafe System One) adapter: turns a screen's element list into a device action.

Jev cannot see screenshots and cannot write text. It answers typed questions
(choice / noul) over a text state in ~70-500ms. Two ways to use it:

* Assist mode (``autonomous=False``, used by the FlashRunner hook): Jev handles
  easy turns and escalates to the VLM on low confidence / goal met / stuck.
* Driver mode (``autonomous=True``, used by ``jev_driver``): Jev is the sole
  decision maker. goal/stuck probabilities are reported on the decision and
  the driver acts on them; a VLM only *supplies* things (text to type, a
  description of a screen with no usable elements).

Verified live: Bearer auth and the ``choice`` criteria shape. Still unverified:
whether swipe ``direction`` "up"/"down" means finger or content direction.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
MAX_ELEMENTS = 40  # Jev's context is 32k tokens; keep the state well under it.
_EDITABLE = ("EditText", "AutoCompleteTextView", "TextInputEditText")
# Low-risk, easily undone moves that Jev may take at lower confidence in driver mode
# (typing into an already-focused field can be cleared; a wrong tap can't be un-sent).
_NAV_ACTIONS = frozenset(
    {"scroll_down", "scroll_up", "swipe_left", "swipe_right", "back", "home", "type_focused"}
)


@dataclass
class JevDecision:
    """One Jev turn, already shaped like a LangChain tool call."""

    tool_call: dict[str, Any] | None
    action: str
    action_confidence: float
    goal_prob: float
    stuck_prob: float
    escalate_reason: str | None = None
    needs_text_for: int | None = None  # Artemis index of a field Jev wants typed into
    raw: dict[str, Any] = field(default_factory=dict)


def _label(el: dict[str, Any]) -> str:
    text = el.get("text") or el.get("content-desc") or el.get("content_desc") or ""
    rid = (el.get("resource_id") or el.get("resource-id") or "").split("/")[-1]
    cls = (el.get("class") or "").split(".")[-1]
    return " | ".join(p for p in (str(text).strip()[:60], rid, cls) if p)


def _ids(elements: list[dict[str, Any]]) -> list[int]:
    """Artemis element index for each element (falls back to 1-based position)."""
    return [int(el.get("index", i)) for i, el in enumerate(elements[:MAX_ELEMENTS], start=1)]


def _is_editable(el: dict[str, Any]) -> bool:
    return any(k in (el.get("class") or "") for k in _EDITABLE) or bool(el.get("editable"))


def build_state(
    goal: str, elements: list[dict[str, Any]], history: list[str], notes: str = ""
) -> str:
    lines = [f"GOAL: {goal}", "RECENT ACTIONS:"]
    lines += [f"- {h}" for h in history[-6:]] or ["- (none)"]
    if notes:
        lines += ["SCREEN NOTES:", notes]
    lines.append("SCREEN ELEMENTS (index: description):")
    lines += [f"{i}: {_label(el)}" for i, el in zip(_ids(elements), elements)]
    return "\n".join(lines)


def build_questions(elements: list[dict[str, Any]] | int, keyboard: bool = False) -> dict[str, Any]:
    if isinstance(elements, int):  # legacy: bare count
        elements = [{} for _ in range(elements)]
    criteria = {f"tap_{i}": f"Tap element {i}" for i in _ids(elements)}
    criteria |= {
        f"type_{i}": f"Focus text field {i} and type text"
        for i, el in zip(_ids(elements), elements)
        if _is_editable(el)
    }
    if keyboard:
        criteria["type_focused"] = "Type text into the text field that is currently focused"
    criteria |= {
        "scroll_down": "Scroll down to reveal more content",
        "scroll_up": "Scroll up",
        "swipe_left": "Swipe left to go to the next page/screen (e.g. launcher page, carousel)",
        "swipe_right": "Swipe right to go to the previous page/screen",
        "back": "Press the Android back button",
        "home": "Press the Android home button to go to the home screen",
    }
    return {
        "action": {
            "type": "choice",
            "instructions": "Pick the single next action that best advances the GOAL.",
            "criteria": criteria,
        },
        "goal_met": {
            "type": "noul",
            "instructions": "Is the GOAL already fully achieved on the current screen?",
        },
        "stuck": {
            "type": "noul",
            "instructions": "Is progress stalled, looping, or the goal unreachable from here?",
        },
    }


def build_app_questions(packages: list[str]) -> dict[str, Any]:
    criteria = {f"app_{i}": f"Launch {pkg}" for i, pkg in enumerate(packages)}
    criteria["none"] = "The goal does not begin by launching an app, or no listed app fits it"
    return {
        "app": {
            "type": "choice",
            "instructions": (
                "Which installed app should be launched first to achieve the GOAL? "
                "Prefer the app the goal names; choose none if unsure."
            ),
            "criteria": criteria,
        }
    }


class JevClient:
    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self._http = httpx.AsyncClient(timeout=timeout)

    async def ask(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        resp = await self._http.post(
            JEV_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": JEV_MODEL, "state": state, "questions": questions},
        )
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._http.aclose()


class JevFlashModel:
    """Turns element lists into tool calls; escalates (returns no call) when unsure."""

    def __init__(
        self,
        client: JevClient | None = None,
        min_confidence: float = 0.75,
        goal_threshold: float = 0.85,
        stuck_threshold: float = 0.5,
        autonomous: bool = False,
        explore_confidence: float = 0.3,
    ):
        self.explore_confidence = explore_confidence
        self.client = client or JevClient()
        self.min_confidence = min_confidence
        self.goal_threshold = goal_threshold
        self.stuck_threshold = stuck_threshold
        self.autonomous = autonomous

    async def decide(
        self,
        goal: str,
        elements: list[dict[str, Any]],
        history: list[str],
        notes: str = "",
        keyboard: bool = False,
    ) -> JevDecision:
        if keyboard:
            notes = ("The on-screen keyboard is open, so a text field is focused. " + notes).strip()
        if not elements:
            return self._escalate("no UI elements (canvas/video/blank screen)")
        raw = await self.client.ask(
            build_state(goal, elements, history, notes), build_questions(elements, keyboard)
        )
        ans = raw.get("answers", {})
        action = ans.get("action", {}).get("choice", "")
        conf = float(ans.get("action", {}).get("confidence", 0.0))
        goal_p = float(ans.get("goal_met", {}).get("noul", 0.0))
        stuck_p = float(ans.get("stuck", {}).get("noul", 0.0))

        def out(call, reason=None):
            return JevDecision(call, action, conf, goal_p, stuck_p, reason, raw=raw)

        if not self.autonomous:
            if goal_p >= self.goal_threshold:
                return out(None, "goal likely met: confirm with VLM before report_task_status")
            if stuck_p >= self.stuck_threshold:
                return out(None, "stuck: needs VLM reasoning")
        # Driver mode: exploring (swipe/scroll/back/home) is cheap and reversible,
        # so it needs far less confidence than a tap or typing, which can act.
        explore = self.autonomous and action in _NAV_ACTIONS and conf >= self.explore_confidence
        if conf < self.min_confidence and not explore:
            return out(None, f"low confidence {conf:.2f}")
        return self._to_call(action, elements, out)

    async def pick_app(
        self, goal: str, packages: list[str], min_confidence: float = 0.8
    ) -> str | None:
        """Jev picks the package to launch from the installed-app list, or None."""
        if not packages:
            return None
        raw = await self.client.ask(f"GOAL: {goal}", build_app_questions(packages))
        ans = raw.get("answers", {}).get("app", {})
        choice, conf = ans.get("choice", "none"), float(ans.get("confidence", 0.0))
        if choice == "none" or conf < min_confidence or not choice.startswith("app_"):
            return None
        i = choice.partition("_")[2]
        return packages[int(i)] if i.isdigit() and int(i) < len(packages) else None

    def _to_call(self, action: str, elements, out) -> JevDecision:
        by_id = dict(zip(_ids(elements), elements))
        if action.startswith(("tap_", "type_")) and action != "type_focused":
            kind, _, num = action.partition("_")
            idx = int(num) if num.isdigit() else -1
            if idx not in by_id:
                return out(None, f"index {idx} not on screen")
            if kind == "type":
                d = out(None, "text needed: supplier must provide it")
                d.needs_text_for = idx
                return d
            if _is_editable(by_id[idx]):
                return out(None, "text field: Jev cannot produce text")
            return out(self._call("click", {"target": idx}))
        if action == "type_focused":
            d = out(None, "text needed: supplier must provide it")
            d.needs_text_for = 0  # 0 = the focused field (no element target)
            return d
        if action == "home":
            return out(self._call("press_key", {"key": "HOME"}))
        if action == "back":
            return out(self._call("press_key", {"key": "BACK"}))
        if action in ("scroll_down", "scroll_up"):
            # Finger swipes up to scroll content down.
            direction = "up" if action == "scroll_down" else "down"
            return out(self._call("swipe", {"direction": direction}))  # UNVERIFIED semantics
        if action in ("swipe_left", "swipe_right"):
            return out(self._call("swipe", {"direction": action.split("_")[1]}))
        return out(None, f"unknown action {action!r}")

    @staticmethod
    def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "args": args, "id": str(uuid.uuid4())}

    @staticmethod
    def _escalate(reason: str) -> JevDecision:
        return JevDecision(None, "", 0.0, 0.0, 0.0, reason)
