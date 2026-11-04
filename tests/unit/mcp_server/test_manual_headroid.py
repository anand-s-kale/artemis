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

"""Manual mode under ARTEMIS_DRIVER=headroid: hidden display only, released on end."""

import asyncio
import os
import sys
import types

import adbutils
import pytest

from artemis.mcp import adb_server
from artemis.mcp.action_types import ActionCode
from artemis.mcp.actuators.adb import AdbActuator
from artemis.runtime import device_lock
from artemis.runtime.device_lock import DeviceBusyError
from mcp_server.manual_session import ManualSessionRegistry

try:  # the real error type, when headroid is installed next to the fork
    from headroid.rpc import RpcError as _HeadroidRpcError
except ImportError:
    _HeadroidRpcError = None

SERIAL = "SERIAL"


class FakeHeadroidDriver:
    """Records every call; anything that would shell out raises."""

    instances: list["FakeHeadroidDriver"] = []

    def __init__(self, device_id: str) -> None:
        self._device_id = device_id
        self.calls: list[tuple] = []
        FakeHeadroidDriver.instances.append(self)

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def screen_size(self) -> tuple[int, int]:
        return (720, 1600)

    async def connect(self) -> None:
        self.calls.append(("connect",))

    async def disconnect(self) -> None:
        await asyncio.sleep(0)  # a real await: an un-awaited coroutine never gets here
        self.calls.append(("disconnect",))

    async def launch_app(self, package_name: str) -> bool:
        self.calls.append(("launch_app", package_name))
        return True

    async def stop_app(self, package_name: str) -> bool:
        self.calls.append(("stop_app", package_name))
        return True

    async def open_url(self, url: str) -> bool:
        self.calls.append(("open_url", url))
        self.last_opened_package = "com.android.chrome"
        return True

    async def input_text(self, text: str, clear_existing: bool = True) -> bool:
        self.calls.append(("input_text", text, clear_existing))
        return True

    async def press_key(self, key) -> bool:
        self.calls.append(("press_key", key))
        return True

    async def execute_shell(self, command: str, timeout_seconds: float = 15.0) -> str:
        raise AssertionError(f"execute_shell must not run under headroid: {command!r}")

    def count(self, name: str) -> int:
        return sum(1 for call in self.calls if call[0] == name)


def _register_driver(monkeypatch, driver_cls) -> None:
    module = types.ModuleType("headroid.integrations.artemis")
    module.HeadroidDriver = driver_cls
    monkeypatch.setitem(sys.modules, "headroid", types.ModuleType("headroid"))
    monkeypatch.setitem(
        sys.modules, "headroid.integrations", types.ModuleType("headroid.integrations")
    )
    monkeypatch.setitem(sys.modules, "headroid.integrations.artemis", module)


def _boom(*_args, **_kwargs):
    raise AssertionError("direct device access under ARTEMIS_DRIVER=headroid")


@pytest.fixture(autouse=True)
def _headroid_env(monkeypatch):
    for name in (
        "ARTEMIS_CLOUD_MODE",
        "ARTEMIS_MOCK_DRIVER",
        "ARTEMIS_DEVICE_ID",
        "ADB_DEVICE_SERIAL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARTEMIS_DRIVER", "headroid")
    FakeHeadroidDriver.instances = []
    _register_driver(monkeypatch, FakeHeadroidDriver)
    monkeypatch.setattr(adb_server, "_CONTROLLERS", {})
    monkeypatch.setattr(adb_server, "_GLOBAL_CONTROLLER", None)
    # Nothing on the manual-mode path may reach adb or the screen helper directly.
    monkeypatch.setattr(adb_server, "create_screen_client", _boom)
    monkeypatch.setattr(adb_server.AdbClient, "device_list", _boom)
    monkeypatch.setattr(adbutils.AdbDevice, "shell", _boom)
    import artemis.tools.mobile.launch_app as launch_app_tool
    import artemis.utils.app_launch_utils as app_launch_utils

    monkeypatch.setattr(launch_app_tool, "find_package", _boom)
    monkeypatch.setattr(app_launch_utils, "launch_app_with_retries", _boom)


def test_controller_under_headroid_skips_screen_client():
    controller = adb_server._get_controller(SERIAL)

    driver = controller.driver
    assert isinstance(driver, FakeHeadroidDriver)
    assert driver.device_id == SERIAL
    assert driver.count("connect") == 1
    assert (controller.ctx.device.device_width, controller.ctx.device.device_height) == (
        driver.screen_size
    )
    assert controller.ctx.ui_adb_client is None
    assert controller.ctx.adb_client is None
    assert adb_server._get_controller(SERIAL) is controller


def test_controller_under_headroid_needs_a_serial():
    with pytest.raises(Exception, match="ARTEMIS_DEVICE_ID"):
        adb_server._get_controller(None)
    assert FakeHeadroidDriver.instances == []


def test_session_end_disconnects_driver():
    registry = ManualSessionRegistry()
    registry.get_or_create(SERIAL)
    first = FakeHeadroidDriver.instances[-1]

    assert registry.end(SERIAL) is True

    assert first.count("disconnect") == 1
    rebuilt = adb_server._get_controller(SERIAL)
    assert rebuilt.driver is not first
    assert rebuilt.driver.count("connect") == 1
    adb_server.drop_controller(SERIAL)
    assert rebuilt.driver.count("disconnect") == 1


@pytest.mark.asyncio
async def test_session_end_disconnects_driver_inside_running_loop():
    # The MCP tools call the (sync) registry from inside the server's event loop.
    registry = ManualSessionRegistry()
    registry.get_or_create(SERIAL)
    driver = FakeHeadroidDriver.instances[-1]

    assert registry.end(SERIAL) is True

    assert driver.count("disconnect") == 1
    assert SERIAL not in adb_server._CONTROLLERS
    assert adb_server._GLOBAL_CONTROLLER is None


def test_drop_controller_unknown_serial_is_a_no_op():
    adb_server.drop_controller("never-created")


def test_second_session_same_serial_is_busy(monkeypatch):
    first_server = ManualSessionRegistry()
    second_server = ManualSessionRegistry()
    first_server.get_or_create(SERIAL)
    try:
        # Two servers are two processes: the lock is re-entrant within one pid, so
        # the second server acquires under a different (live) pid.
        other_pid = os.getppid()
        with monkeypatch.context() as patched:
            patched.setattr(device_lock.os, "getpid", lambda: other_pid)
            with pytest.raises(DeviceBusyError) as busy:
                second_server.get_or_create(SERIAL)
        assert "already controlled by PID" in str(busy.value)
        assert len(FakeHeadroidDriver.instances) == 1
    finally:
        first_server.end(SERIAL)


def _actuator() -> tuple[AdbActuator, FakeHeadroidDriver]:
    controller = adb_server._get_controller(SERIAL)
    return AdbActuator(controller.ctx, controller), controller.driver


@pytest.mark.asyncio
async def test_manage_app_launch_uses_driver():
    actuator, driver = _actuator()

    launched = await actuator.manage_app("launch", "com.example.app")
    stopped = await actuator.manage_app("stop", "com.example.app")

    assert launched.ok, launched.message
    assert stopped.ok, stopped.message
    assert ("launch_app", "com.example.app") in driver.calls
    assert ("stop_app", "com.example.app") in driver.calls


@pytest.mark.asyncio
async def test_manage_app_launch_failure_names_package_requirement():
    actuator, driver = _actuator()

    async def fail(package_name: str) -> bool:
        return False

    driver.launch_app = fail
    result = await actuator.manage_app("launch", "Settings")

    assert not result.ok
    assert "package name" in result.message


@pytest.mark.asyncio
async def test_open_link_uses_the_driver():
    # headroid's open_url launches the handling app on the hidden display (muted,
    # stopped at teardown); the controller's open_url would `am start` on display 0.
    actuator, driver = _actuator()

    result = await actuator.open_link("https://example.com")

    assert result.ok, result.message
    # the agent is told which app it is in now
    assert "with com.android.chrome on the hidden display" in result.message
    assert driver.count("open_url") == 1
    assert ("open_url", "https://example.com") in driver.calls


@pytest.mark.asyncio
async def test_open_link_reports_headroids_refusal():
    actuator, driver = _actuator()

    async def no_single_app(url: str) -> bool:
        raise RuntimeError(f"[-32602] url: no single app handles {url}")

    driver.open_url = no_single_app
    result = await actuator.open_link("x:y")

    assert not result.ok
    assert "no single app handles x:y" in result.message


@pytest.mark.asyncio
@pytest.mark.skipif(_HeadroidRpcError is None, reason="headroid is not installed")
async def test_open_link_reports_headroids_rpc_error():
    # the error type the real HeadroidDriver raises; the real one, imported before
    # the fixture replaces the headroid modules with stubs
    actuator, driver = _actuator()

    async def open_on_display_0(url: str) -> bool:
        raise _HeadroidRpcError(
            -32602, "url: com.android.chrome is open on display [0]; close it on the phone first"
        )

    driver.open_url = open_on_display_0
    result = await actuator.open_link("https://example.com")

    assert not result.ok
    assert "is open on display [0]" in result.message


@pytest.mark.asyncio
async def test_stdio_open_link_goes_through_the_driver(monkeypatch):
    # the stdio server's own open_link tool must not reach the controller's `am start`
    driver = adb_server._get_controller(SERIAL).driver
    monkeypatch.setenv("ADB_DEVICE_SERIAL", SERIAL)

    answer = await adb_server.open_link(None, "https://example.com")

    assert answer.startswith("Success"), answer
    assert "hidden display" in answer
    assert ("open_url", "https://example.com") in driver.calls


@pytest.mark.asyncio
async def test_stdio_open_link_reports_a_refusal(monkeypatch):
    driver = adb_server._get_controller(SERIAL).driver

    async def refused(url: str) -> bool:
        raise RuntimeError("[-32602] url: headroid does not open tel: links")

    driver.open_url = refused
    monkeypatch.setenv("ADB_DEVICE_SERIAL", SERIAL)

    answer = await adb_server.open_link(None, "tel:123")

    assert answer.startswith("Failed"), answer
    assert "does not open tel: links" in answer


@pytest.mark.asyncio
async def test_clearing_text_goes_through_the_driver():
    actuator, driver = _actuator()

    typed = await actuator.input_text("hello", clear_exist=True)

    assert typed.ok, typed.message
    assert ("input_text", "", True) in driver.calls
    assert ("input_text", "hello", False) in driver.calls


def test_failed_connect_releases_the_device_lock(monkeypatch):
    class FailingDriver(FakeHeadroidDriver):
        async def connect(self) -> None:
            raise RuntimeError("no hidden display")

    _register_driver(monkeypatch, FailingDriver)
    registry = ManualSessionRegistry()
    with pytest.raises(RuntimeError, match="no hidden display"):
        registry.get_or_create(SERIAL)
    assert SERIAL not in adb_server._CONTROLLERS

    # The lock must be free again: another server (another pid; the lock is
    # re-entrant within one) gets past it to the driver instead of DeviceBusyError.
    other_pid = os.getppid()
    with monkeypatch.context() as patched:
        patched.setattr(device_lock.os, "getpid", lambda: other_pid)
        with pytest.raises(RuntimeError, match="no hidden display"):
            ManualSessionRegistry().get_or_create(SERIAL)
    assert len(FailingDriver.instances) == 2
