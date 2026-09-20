import asyncio

from artemis.llm.jev_adapter import JevFlashModel, build_questions, build_state

ELEMENTS = [
    {"text": "Search", "class": "android.widget.EditText"},
    {"text": "Settings", "class": "android.widget.TextView"},
]


class FakeClient:
    def __init__(self, choice, conf=0.9, goal=0.1, stuck=0.1):
        self.resp = {
            "answers": {
                "action": {"type": "choice", "choice": choice, "confidence": conf},
                "goal_met": {"type": "noul", "noul": goal},
                "stuck": {"type": "noul", "noul": stuck},
            }
        }

    async def ask(self, state, questions):
        assert "GOAL:" in state and "action" in questions
        return self.resp


def decide(client, elements=ELEMENTS):
    return asyncio.run(JevFlashModel(client).decide("open settings", elements, []))


def test_tap_becomes_click_call():
    d = decide(FakeClient("tap_2"))
    assert d.tool_call["name"] == "click" and d.tool_call["args"] == {"target": 2}


def test_low_confidence_escalates():
    assert decide(FakeClient("tap_2", conf=0.4)).tool_call is None


def test_text_field_escalates():
    assert "text" in decide(FakeClient("tap_1")).escalate_reason


def test_goal_met_and_stuck_escalate():
    assert decide(FakeClient("tap_2", goal=0.95)).tool_call is None
    assert decide(FakeClient("tap_2", stuck=0.9)).tool_call is None


def test_no_elements_escalates():
    assert decide(FakeClient("tap_1"), elements=[]).tool_call is None


def test_state_and_questions_shape():
    assert "1: Search" in build_state("g", ELEMENTS, ["clicked x"])
    assert "tap_2" in build_questions(2)["action"]["criteria"]


def test_flash_runner_jev_switch(monkeypatch):
    from types import SimpleNamespace

    from artemis.agents.flash.runner import FlashRunner

    r = FlashRunner.__new__(FlashRunner)
    r.goal, r._jev_history = "open settings", []
    r._jev = JevFlashModel(FakeClient("tap_2"))
    state = SimpleNamespace(indexed_elements=ELEMENTS)

    monkeypatch.delenv("ARTEMIS_FLASH_DECIDER", raising=False)
    assert asyncio.run(r._try_jev(state)) is None  # off by default

    monkeypatch.setenv("ARTEMIS_FLASH_DECIDER", "jev")
    msg = asyncio.run(r._try_jev(state))
    assert msg.tool_calls[0]["name"] == "click" and r._jev_history

    r._jev = JevFlashModel(FakeClient("tap_2", conf=0.1))
    assert asyncio.run(r._try_jev(state)) is None  # escalates to VLM
