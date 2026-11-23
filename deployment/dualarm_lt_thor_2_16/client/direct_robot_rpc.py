#!/usr/bin/env python3
"""Minimal direct send_commend_py client for exact CodeIT command strings."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading


class DirectRobotRpc:
    def __init__(self, host: str, port: int, connect_timeout_ms: int, async_timeout_ms: int):
        vendor = Path(__file__).with_name("vendor")
        if not (vendor / "rpc.so").is_file():
            raise RuntimeError(f"direct RPC library is missing: {vendor / 'rpc.so'}")
        sys.path.insert(0, str(vendor))
        rpc = importlib.import_module("rpc")
        self._rpc = rpc
        self._client = rpc.CPPClient(str(host), int(port), int(connect_timeout_ms))
        if not self._client.IsConnected():
            raise ConnectionError(f"direct robot RPC connection failed: {self._client.GetErrorInfo()}")
        self._async_timeout_ms = int(async_timeout_ms)
        self._sequence = 0
        self._lock = threading.Lock()

    def _message(self, command: str):
        message = self._rpc.Msg(command)
        message.setMsgID(10001)
        self._sequence += 1
        message.setMsgSeqID(self._sequence)
        return message

    def call(self, command: str, timeout_ms: int) -> dict:
        """Send a short configuration command and require its controller response."""
        with self._lock:
            message = self._message(command)
            sequence = self._sequence
            status, responses = self._client.CallAwait(message, int(timeout_ms))
        errors = [str(item.message) for item in (responses or []) if int(item.code) < 0]
        if int(status) != 0 or errors:
            raise RuntimeError(
                f"direct RPC command failed: status={status} errors={errors} command={command}"
            )
        return {
            "success": True, "transport": "direct-5868", "status": int(status),
            "sequence": sequence, "response_count": len(responses or []),
        }

    def submit(self, command: str) -> dict:
        """Submit a streaming command without registering a pending response."""
        def ignore_response(status, responses):
            del status, responses

        with self._lock:
            message = self._message(command)
            sequence = self._sequence
            accepted = self._client.CallAsync(
                message, self._async_timeout_ms, ignore_response, False
            )
        if not accepted:
            raise RuntimeError(f"direct RPC rejected submission: {command}")
        return {
            "success": True, "transport": "direct-5868", "submitted": True,
            "sequence": sequence,
        }

    def close(self) -> None:
        # CPPClient owns its socket; dropping it closes the connection. Deliberately do
        # not send Disable because this executor did not enable the controller.
        self._client = None
