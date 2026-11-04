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

"""MCP server startup under ARTEMIS_DRIVER=headroid never touches the phone's own screen."""

import subprocess

from mcp_server import server


def _fake_run(monkeypatch):
    started = []

    class FakeMcp:
        def run(self, **kwargs):
            started.append(kwargs)

    monkeypatch.setattr(server, "mcp", FakeMcp())
    monkeypatch.setattr(server, "apply_tool_allowlist", lambda *_args: None)
    return started


def test_main_under_headroid_never_starts_the_awake_service(monkeypatch):
    monkeypatch.setenv("ARTEMIS_DRIVER", "headroid")
    monkeypatch.setenv("ADB_DEVICE_SERIAL", "SERIAL")
    monkeypatch.delenv("ARTEMIS_KEEP_DEVICE_AWAKE", raising=False)
    started = _fake_run(monkeypatch)
    awake_calls = []
    monkeypatch.setattr(server, "start_awake_service", lambda: awake_calls.append("start"))

    def no_adb(*args, **kwargs):
        raise AssertionError(f"adb run at startup under headroid: {args!r}")

    monkeypatch.setattr(subprocess, "run", no_adb)

    server.main()

    assert started == [{"transport": "stdio"}]
    assert awake_calls == []


def test_main_without_headroid_still_starts_the_awake_service(monkeypatch):
    monkeypatch.delenv("ARTEMIS_DRIVER", raising=False)
    _fake_run(monkeypatch)
    threads = []

    class FakeThread:
        def __init__(self, target, daemon, name):
            threads.append((target, name))

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", FakeThread)
    server.main()

    assert threads == [(server.start_awake_service, "artemis-awake-init")]
