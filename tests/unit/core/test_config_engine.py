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

"""Unit tests for CascadingConfigEngine and variable interpolation."""

import os
from unittest.mock import patch

from artemis.config.core import CascadingConfigEngine, interpolate_config_value


def test_interpolate_env_variables():
    """Verify ${env:KEY} expressions are properly interpolated."""
    with patch.dict("os.environ", {"TEST_KEY": "secret_123"}):
        result = interpolate_config_value("Bearer ${env:TEST_KEY}")
        assert result == "Bearer secret_123"

        default_result = interpolate_config_value("${env:MISSING_KEY:-fallback_val}")
        assert default_result == "fallback_val"


def test_interpolate_path_variables():
    """Verify ${path:KEY} expressions resolve system standard paths."""
    result = interpolate_config_value("${path:temp_dir}/screenshots")
    assert "/screenshots" in result or "\\screenshots" in result


def test_cascading_config_merge():
    """Verify CLI overrides take precedence in CascadingConfigEngine."""
    engine = CascadingConfigEngine()
    merged = engine.load_raw_config(cli_overrides={"default": {"model": "gemini-test-custom"}})
    assert merged.get("default", {}).get("model") == "gemini-test-custom"


_HEADROID_ENV = {
    "ARTEMIS_DRIVER": "headroid",
    "ARTEMIS_MCP_TOOLS": "mobile_observe,mobile_act,mobile_session_end",
    "ARTEMIS_MCP_INLINE_SCREENSHOT": "1",
    "ADB_DEVICE_SERIAL": "HIDDEN",
    "ARTEMIS_DEVICE_ID": "HIDDEN",
    "ARTEMIS_KEEP_DEVICE_AWAKE": "0",
}
_PROJECT_DOTENV = (
    "ARTEMIS_DRIVER=adb\n"
    "ARTEMIS_MCP_TOOLS=mobile_run_task\n"
    "ARTEMIS_MCP_INLINE_SCREENSHOT=0\n"
    "ADB_DEVICE_SERIAL=OTHER\n"
    "ARTEMIS_DEVICE_ID=OTHER\n"
    "ARTEMIS_KEEP_DEVICE_AWAKE=1\n"
    "SOME_PROJECT_KEY=from-dotenv\n"
)


def test_workspace_dotenv_cannot_override_headroid_settings(tmp_path):
    (tmp_path / ".env").write_text(_PROJECT_DOTENV, encoding="utf-8")
    with patch.dict("os.environ", _HEADROID_ENV, clear=False):
        os.environ.pop("SOME_PROJECT_KEY", None)
        CascadingConfigEngine(workspace_root=tmp_path)
        for key, value in _HEADROID_ENV.items():
            assert os.environ[key] == value, key
        # Everything else in the project .env still loads as before.
        assert os.environ["SOME_PROJECT_KEY"] == "from-dotenv"


def test_workspace_dotenv_still_overrides_without_headroid(tmp_path):
    (tmp_path / ".env").write_text(_PROJECT_DOTENV, encoding="utf-8")
    without_driver = {k: v for k, v in _HEADROID_ENV.items() if k != "ARTEMIS_DRIVER"}
    with patch.dict("os.environ", without_driver, clear=False):
        os.environ.pop("ARTEMIS_DRIVER", None)
        CascadingConfigEngine(workspace_root=tmp_path)
        assert os.environ["ARTEMIS_DRIVER"] == "adb"
        assert os.environ["ADB_DEVICE_SERIAL"] == "OTHER"
        assert os.environ["ARTEMIS_MCP_INLINE_SCREENSHOT"] == "0"
