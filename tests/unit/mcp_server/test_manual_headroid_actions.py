# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The full manual-mode path under ARTEMIS_DRIVER=headroid, tool call to driver call.

mobile_observe / mobile_act / mobile_session_end run through the real registry,
McpActionExecutor, action session, AdbActuator and UnifiedMobileController onto
a recording fake HeadroidDriver. Every way of reaching the phone directly
(subprocess.run, asyncio.create_subprocess_exec, adbutils shell and device
listing, the screen helper) raises, so any action that would touch display 0
fails the test.
"""

import asyncio
import base64
import os
from pathlib import Path
import subprocess
import sys
import types

import adbutils
import pytest

from artemis.drivers.base import ScreenData
from artemis.mcp import adb_server
from artemis.mcp.actuators import adb as adb_actuator
import artemis.mcp.observation as observation
from mcp_server.manual_session import ManualSessionRegistry
import mcp_server.tools.manual_mode as manual_mode

pytestmark = pytest.mark.asyncio

SERIAL = "SERIAL"
WIDTH, HEIGHT = 720, 1600
# 1x1 PNG, the format headroid's screenshots come in.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cfc0f01f0005000201a5f645400000000049454e44ae426082"
)
# What headroid's tree_to_elements hands ARTEMIS: the uiautomator element format
# AndroidAdbDriver produces (string bounds and "true"/"false" flags).
ELEMENTS = [
    {
        "text": "Wi-Fi",
        "content-desc": "",
        "resource-id": "android:id/title",
        "class": "android.widget.TextView",
        "package": "com.android.settings",
        "bounds": "[40,400][680,520]",
        "clickable": "true",
        "focusable": "true",
        "focused": "false",
        "enabled": "true",
    },
    {
        "text": "Search",
        "content-desc": "",
        "resource-id": "com.example:id/search",
        "class": "android.widget.EditText",
        "package": "com.android.settings",
        "bounds": "[40,100][680,200]",
        "clickable": "true",
        "focusable": "true",
        "focused": "false",
        "enabled": "true",
    },
]


class RecordingHeadroidDriver:
    """Records every call the manual-mode path makes; has no way to reach adb."""

    instances: list["RecordingHeadroidDriver"] = []

    def __init__(self, device_id: str) -> None:
        self._device_id = device_id
        self.calls: list[tuple] = []
        RecordingHeadroidDriver.instances.append(self)

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def screen_size(self) -> tuple[int, int]:
        return (WIDTH, HEIGHT)

    async def connect(self) -> None:
        self.calls.append(("connect",))

    async def disconnect(self) -> None:
        await asyncio.sleep(0)
        self.calls.append(("disconnect",))

    async def get_screen_data(self, skip_settling: bool = False) -> ScreenData:
        self.calls.append(("get_screen_data",))
        return ScreenData(
            screenshot_bytes=PNG,
            screenshot_base64=base64.b64encode(PNG).decode("ascii"),
            ui_hierarchy_xml=None,
            ui_elements=[dict(element) for element in ELEMENTS],
            width=WIDTH,
            height=HEIGHT,
            platform="android",
        )

    async def get_current_package(self) -> str | None:
        self.calls.append(("get_current_package",))
        return "com.example.app"

    async def tap(
        self, x: int, y: int, duration_ms: int = 100, times: int = 1, delay_ms: int = 100
    ) -> bool:
        self.calls.append(("tap", x, y, duration_ms, times))
        return True

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> bool:
        self.calls.append(("long_press", x, y, duration_ms))
        return True

    async def swipe(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration_ms: int = 800
    ) -> bool:
        self.calls.append(("swipe", start_x, start_y, end_x, end_y))
        return True

    async def input_text(self, text: str, clear_existing: bool = True) -> bool:
        self.calls.append(("input_text", text, clear_existing))
        return True

    async def press_key(self, key) -> bool:
        self.calls.append(("press_key", str(key)))
        return True

    async def launch_app(self, package_name: str) -> bool:
        self.calls.append(("launch_app", package_name))
        return True

    async def stop_app(self, package_name: str) -> bool:
        self.calls.append(("stop_app", package_name))
        return True

    async def open_url(self, url: str) -> bool:
        self.calls.append(("open_url", url))
        return True

    async def execute_shell(self, command: str, timeout_seconds: float = 15.0) -> str:
        raise AssertionError(f"execute_shell must not run under headroid: {command!r}")

    async def start_video_recording(self, output_dir=None) -> None:
        raise AssertionError("no video recording on a hidden display")

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


def _boom(*args, **_kwargs):
    raise AssertionError(f"direct device access under ARTEMIS_DRIVER=headroid: {args!r}")


async def _async_boom(*args, **_kwargs):
    raise AssertionError(f"subprocess under ARTEMIS_DRIVER=headroid: {args!r}")


@pytest.fixture(autouse=True)
def _headroid_world(monkeypatch, tmp_path):
    for name in ("ARTEMIS_CLOUD_MODE", "ARTEMIS_MOCK_DRIVER", "ARTEMIS_DEVICE_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARTEMIS_DRIVER", "headroid")
    monkeypatch.setenv("ADB_DEVICE_SERIAL", SERIAL)
    monkeypatch.setenv("ARTEMIS_MCP_INLINE_SCREENSHOT", "1")

    module = types.ModuleType("headroid.integrations.artemis")
    module.HeadroidDriver = RecordingHeadroidDriver
    monkeypatch.setitem(sys.modules, "headroid", types.ModuleType("headroid"))
    monkeypatch.setitem(
        sys.modules, "headroid.integrations", types.ModuleType("headroid.integrations")
    )
    monkeypatch.setitem(sys.modules, "headroid.integrations.artemis", module)
    RecordingHeadroidDriver.instances = []

    monkeypatch.setattr(adb_server, "_CONTROLLERS", {})
    monkeypatch.setattr(adb_server, "_GLOBAL_CONTROLLER", None)
    # The real registry and its default executor factory (_get_controller).
    registry = ManualSessionRegistry()
    monkeypatch.setattr(manual_mode, "_registry", registry)

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _async_boom)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _async_boom)
    monkeypatch.setattr(adbutils.AdbDevice, "shell", _boom)
    monkeypatch.setattr(adb_server.AdbClient, "device_list", _boom)
    monkeypatch.setattr(adb_server, "create_screen_client", _boom)
    # No real screenshots directory; the fallback path still writes a real file.
    monkeypatch.setattr(observation, "get_temp_dir", lambda _name: tmp_path)
    # ensure_focus_at_coords waits 1 s for the keyboard after a focusing tap.
    monkeypatch.setattr(adb_actuator, "asyncio", types.SimpleNamespace(sleep=_no_sleep))
    yield registry
    for key in list(registry._sessions):
        registry._end_key(key)


async def _no_sleep(_seconds):
    await asyncio.sleep(0)


def _driver() -> RecordingHeadroidDriver:
    assert len(RecordingHeadroidDriver.instances) == 1
    return RecordingHeadroidDriver.instances[0]


def _data(result):
    # With inline screenshots the tools return [dict, Image].
    assert isinstance(result, list) and len(result) == 2, result
    return result[0]


async def test_observe_then_every_action_reaches_only_the_hidden_display(monkeypatch):
    observed = await manual_mode.mobile_observe()
    data = _data(observed)
    assert data["status"] == "success", data
    assert "Wi-Fi" in data["elements_text"]
    assert (data["width"], data["height"]) == (WIDTH, HEIGHT)

    driver = _driver()
    assert driver.device_id == SERIAL
    assert driver.names()[:2] == ["connect", "get_screen_data"]

    steps = [
        ("click", {"target": 1}, "tap"),
        ("long_press", {"target": 1, "duration_ms": 900}, "long_press"),
        ("input_text", {"text": "hello", "target": 2}, "input_text"),
        (
            "click_sequence",
            {
                "sequence": [[500, 300], [500, 600]],
                "target_descriptions": ["first key", "second key"],
            },
            "tap",
        ),
        ("swipe", {"direction": "up"}, "swipe"),
        ("press_key", {"key": "back"}, "press_key"),
        ("press_key", {"key": "enter"}, "press_key"),
        ("manage_app", {"action": "launch", "app_name": "com.example.app"}, "launch_app"),
        ("manage_app", {"action": "stop", "app_name": "com.example.app"}, "stop_app"),
        ("wait_for_delay", {"time_in_ms": 1}, None),
        ("wait_for_text", {"text": "Wi-Fi", "timeout_ms": 1000}, "get_screen_data"),
        ("erase_one_char", {}, "press_key"),
        ("focus_and_clear_text", {"target": [500, 93]}, "input_text"),
    ]
    for action, args, expected_call in steps:
        before = len(driver.calls)
        result = _data(await manual_mode.mobile_act(action=action, args=args))
        assert result["status"] == "success", (action, result)
        new_calls = driver.names()[before:]
        if expected_call is not None:
            assert expected_call in new_calls, (action, new_calls)

    assert ("launch_app", "com.example.app") in driver.calls
    assert ("stop_app", "com.example.app") in driver.calls
    assert ("press_key", "back") in driver.calls
    assert ("press_key", "delete") in driver.calls
    assert ("input_text", "", True) in driver.calls  # clearing goes through the driver
    assert ("input_text", "hello", False) in driver.calls
    assert sum(1 for name in driver.names() if name == "tap") >= 3
    # One display for the whole session.
    assert driver.names().count("connect") == 1
    assert len(RecordingHeadroidDriver.instances) == 1

    ended = manual_mode.mobile_session_end()
    assert ended["ended"] is True
    assert driver.names()[-1] == "disconnect"


async def test_open_link_goes_through_the_driver():
    # headroid's open_url, never the controller's `am start` (subprocess and adb raise here).
    await manual_mode.mobile_observe()
    result = _data(
        await manual_mode.mobile_act(action="open_link", args={"url": "https://example.com"})
    )
    assert result["status"] != "error", result
    assert "hidden display" in result["message"]
    assert ("open_url", "https://example.com") in _driver().calls


async def test_inline_screenshot_from_the_real_observe_fallback_is_png():
    observed = await manual_mode.mobile_observe()
    data, image = observed
    path = Path(data["screenshot"])
    assert path.suffix == ".png"
    assert path.read_bytes() == PNG
    content = image.to_image_content()
    assert content.mimeType == "image/png"
    assert base64.b64decode(content.data) == PNG


async def test_omitted_and_explicit_serial_are_one_session(monkeypatch):
    await manual_mode.mobile_observe()
    await manual_mode.mobile_observe(device_serial=SERIAL)
    registry = manual_mode._registry
    assert list(registry._sessions) == [SERIAL]
    driver = _driver()

    # Ending by explicit serial ends the session opened without one, and frees the lock.
    assert manual_mode.mobile_session_end(device_serial=SERIAL)["ended"] is True
    assert driver.names()[-1] == "disconnect"
    assert manual_mode.mobile_session_end()["ended"] is False
    assert registry._sessions == {}
    assert os.environ["ADB_DEVICE_SERIAL"] == SERIAL


async def test_launch_error_from_the_driver_names_the_package_requirement(monkeypatch):
    await manual_mode.mobile_observe()

    async def unknown_package(package_name: str) -> bool:
        raise RuntimeError(f"[-32002] unknown package {package_name}")

    monkeypatch.setattr(_driver(), "launch_app", unknown_package)
    result = _data(
        await manual_mode.mobile_act(
            action="manage_app", args={"action": "launch", "app_name": "Settings"}
        )
    )
    assert result["status"] == "error"
    assert "package name" in result["message"]
    assert "unknown package Settings" in result["message"]
