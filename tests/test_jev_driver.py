import asyncio

from artemis.llm.jev_adapter import JevFlashModel
from artemis.llm.jev_driver import JevDriver, Observation, parse_elements_text

SCREEN_A = (
    "[1] Text: '4:43' | Bounds: [61,5][129,37]\n"
    "[2] Text: 'Gallery' | Bounds: [106,119][250,184] (WARNING: may overlap)\n"
    "[3] Text: 'Domino's' | Bounds: [331,344][431,389]\n"
    "[4] Input: 'Search' | Bounds: [10,60][900,100]\n"
)
SCREEN_B = "[1] Text: 'Photos' | Bounds: [65,114][321,162]\n"


class Script:
    """Fake Jev client: answers come from a queue of (choice, conf, goal, stuck)."""

    def __init__(self, *steps):
        self.steps = list(steps)

    async def ask(self, state, questions):
        choice, conf, goal, stuck = self.steps.pop(0)
        return {
            "answers": {
                "action": {"choice": choice, "confidence": conf},
                "goal_met": {"noul": goal},
                "stuck": {"noul": stuck},
            }
        }


class Device:
    def __init__(self, screens):
        self.screens, self.calls = list(screens), []

    async def observe(self):
        return Observation(self.screens[0])

    async def act(self, name, args):
        self.calls.append((name, args))
        return Observation(self.screens.pop(1) if len(self.screens) > 1 else self.screens[0])


def drive(script, device, supplier=None, **kw):  # kw: JevDriver options
    jev = JevFlashModel(script, autonomous=True)
    d = JevDriver(device.observe, device.act, jev=jev, supplier=supplier, **kw)
    return asyncio.run(d.run("open gallery"))


def test_parser_keeps_indexes_and_drops_status_bar():
    els = parse_elements_text(SCREEN_A)
    assert [e["index"] for e in els] == [2, 3, 4]
    assert els[1]["text"] == "Domino's" and els[2]["editable"] is True


def test_jev_taps_by_artemis_index_then_declares_goal_met():
    dev = Device([SCREEN_A, SCREEN_B])
    r = drive(Script(("tap_2", 0.99, 0.02, 0.02), ("home", 0.9, 0.95, 0.0)), dev)
    assert r.status == "success" and dev.calls == [("click", {"target": 2})]


def test_goal_met_at_start_takes_no_action():
    dev = Device([SCREEN_B])
    r = drive(Script(("home", 0.9, 0.97, 0.0)), dev)
    assert r.status == "success" and dev.calls == []


def test_supplier_provides_text_but_jev_chooses_field():
    class Sup:
        async def supply_text(self, goal, label, history):
            return "battery"

        async def describe_screen(self, shot):
            return None

    dev = Device([SCREEN_A, SCREEN_B])
    r = drive(Script(("type_4", 0.95, 0.0, 0.0), ("home", 0.9, 0.95, 0.0)), dev, Sup())
    assert dev.calls == [("input_text", {"target": 4, "text": "battery"})]
    assert r.status == "success"


def test_type_without_supplier_fails_cleanly():
    r = drive(Script(("type_4", 0.95, 0.0, 0.0)), Device([SCREEN_A]))
    assert r.status == "failed" and "text" in r.reason


def test_stuck_twice_and_repeat_guard_stop_the_loop():
    r = drive(Script(("tap_2", 0.9, 0.0, 0.9), ("tap_2", 0.9, 0.0, 0.9)), Device([SCREEN_A]))
    assert "stuck" in r.reason
    r = drive(Script(*[("tap_2", 0.9, 0.0, 0.0)] * 3), Device([SCREEN_A]))
    assert "repeated" in r.reason


def test_low_confidence_uses_supplier_notes_then_decides():
    class Sup:
        async def supply_text(self, *a):
            return None

        async def describe_screen(self, shot):
            return "a launcher with a Gallery icon"

    dev = Device([SCREEN_A, SCREEN_B])
    r = drive(
        Script(("tap_2", 0.3, 0.0, 0.0), ("tap_2", 0.97, 0.0, 0.0), ("home", 0.9, 0.95, 0.0)),
        dev,
        Sup(),
    )
    assert r.status == "success" and dev.calls == [("click", {"target": 2})]


def test_jev_picks_app_and_launch_skips_launcher_navigation():
    class AppScript(Script):
        async def ask(self, state, questions):
            if "app" in questions:
                return {"answers": {"app": {"choice": "app_1", "confidence": 0.95}}}
            return await super().ask(state, questions)

    dev = Device([SCREEN_B, SCREEN_B])
    r = drive(
        AppScript(("home", 0.9, 0.95, 0.0)),
        dev,
        apps=["com.a.camera", "com.a.gallery"],
    )
    assert dev.calls == [("manage_app", {"action": "launch", "app_name": "com.a.gallery"})]
    assert r.status == "success"


def test_unsure_app_pick_falls_back_to_normal_loop():
    class AppScript(Script):
        async def ask(self, state, questions):
            if "app" in questions:
                return {"answers": {"app": {"choice": "none", "confidence": 0.9}}}
            return await super().ask(state, questions)

    dev = Device([SCREEN_B])
    r = drive(AppScript(("home", 0.9, 0.95, 0.0)), dev, apps=["com.a.gallery"])
    assert dev.calls == [] and r.status == "success"


def test_keyboard_keys_are_stripped_and_type_focused_uses_no_target():
    from artemis.llm.jev_driver import split_keyboard

    keys = "".join(
        f"[{i}] Text: '{c}' | Bounds: [{i},685][{i + 9},750]\n"
        for i, c in enumerate("qwertyuiop", 10)
    )
    screen = "[1] Text: 'Search history' | Bounds: [50,123][260,143]\n" + keys
    els, kb = split_keyboard(parse_elements_text(screen))
    assert kb and [e["index"] for e in els] == [1]

    class Sup:
        async def supply_text(self, goal, label, history):
            return "battery"

        async def describe_screen(self, shot):
            return None

    dev = Device([screen, SCREEN_B])
    r = drive(Script(("type_focused", 0.95, 0.0, 0.0), ("home", 0.9, 0.95, 0.0)), dev, Sup())
    assert dev.calls == [("input_text", {"text": "battery"})] and r.status == "success"


def test_no_confident_action_but_goal_likely_met_counts_as_success():
    r = drive(Script(("tap_2", 0.2, 0.76, 0.1)), Device([SCREEN_A]))
    assert r.status == "success" and "no action left" in r.reason
    r = drive(Script(("tap_2", 0.2, 0.3, 0.1)), Device([SCREEN_A]))
    assert r.status == "failed"
