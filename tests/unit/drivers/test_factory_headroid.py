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

"""ARTEMIS_DRIVER=headroid selects the headroid hidden-display driver."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from artemis.drivers import factory
from artemis.drivers.android.adb_driver import AndroidAdbDriver


def _ctx() -> SimpleNamespace:
    device = SimpleNamespace(device_id="serial-1", device_width=1080, device_height=2400)
    return SimpleNamespace(device=device, adb_client=None, ui_adb_client=None)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ARTEMIS_DRIVER", "ARTEMIS_CLOUD_MODE", "ARTEMIS_MOCK_DRIVER"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_adb_driver():
    ctx = _ctx()
    with patch.object(factory, "AdbClient"):
        driver = factory.create_driver(ctx)
    assert isinstance(driver, AndroidAdbDriver)


def test_headroid_env_selects_headroid(monkeypatch):
    class HeadroidDriver:
        def __init__(self, device_id: str) -> None:
            self.device_id = device_id

    module = types.ModuleType("headroid.integrations.artemis")
    module.HeadroidDriver = HeadroidDriver
    monkeypatch.setitem(sys.modules, "headroid", types.ModuleType("headroid"))
    monkeypatch.setitem(
        sys.modules, "headroid.integrations", types.ModuleType("headroid.integrations")
    )
    monkeypatch.setitem(sys.modules, "headroid.integrations.artemis", module)
    monkeypatch.setenv("ARTEMIS_DRIVER", "headroid")

    ctx = _ctx()
    driver = factory.create_driver(ctx)

    assert isinstance(driver, HeadroidDriver)
    assert driver.device_id == ctx.device.device_id
    assert ctx.adb_client is None
    assert factory.headroid_selected() is True


def test_headroid_env_without_headroid_fails_clearly(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "headroid.integrations.artemis", None)
    monkeypatch.setenv("ARTEMIS_DRIVER", "headroid")

    with pytest.raises(RuntimeError, match=r'pip install "headroid\[artemis\]"'):
        factory.create_driver(_ctx())
