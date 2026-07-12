#!/usr/bin/env python3
"""Minimal Hermes-style coding-agent loop for the onnx-genai OpenAI server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text content to a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories under a path in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the working directory and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def sandbox_path(workdir: Path, requested: str) -> Path:
    candidate = (workdir / requested).resolve() if not os.path.isabs(requested) else Path(requested).resolve()
    try:
        candidate.relative_to(workdir)
    except ValueError as exc:
        raise ValueError(f"path escapes workdir: {requested}") from exc
    return candidate


def execute_tool(workdir: Path, name: str, arguments_json: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid JSON arguments: {exc}"}

    try:
        if name == "read_file":
            path = sandbox_path(workdir, str(args["path"]))
            return {"ok": True, "path": str(path.relative_to(workdir)), "content": path.read_text(encoding="utf-8")}

        if name == "write_file":
            path = sandbox_path(workdir, str(args["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args["content"]), encoding="utf-8")
            return {"ok": True, "path": str(path.relative_to(workdir)), "bytes": path.stat().st_size}

        if name == "list_dir":
            path = sandbox_path(workdir, str(args["path"]))
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return {"ok": True, "path": str(path.relative_to(workdir)), "entries": entries}

        if name == "run_command":
            command = str(args["command"])
            completed = subprocess.run(
                command,
                cwd=workdir,
                shell=True,
                text=True,
                capture_output=True,
                timeout=15,
            )
            return {
                "ok": completed.returncode == 0,
                "command": command,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        return {"ok": False, "error": f"unknown tool: {name}"}
    except Exception as exc:  # tool errors are fed back to the model
        return {"ok": False, "error": str(exc)}


def post_chat(base_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def print_json(label: str, value: Any) -> None:
    print(f"{label}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")


def run_agent(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    repo_root = Path.cwd().resolve()
    try:
        workdir.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"--workdir must be inside the current repository: {workdir}") from exc
    if workdir == repo_root:
        raise SystemExit("--workdir must not be the repository root")

    if args.clean and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a tiny coding agent. Use tools to inspect and modify files. "
                "When you need an action, call exactly one tool. For tool calls use the provided tools. "
                "Never claim command output until you have called run_command and read its tool result. "
                "After tool results show the task is complete, answer briefly with the observed result."
            ),
        },
        {"role": "user", "content": args.task},
    ]

    print(f"WORKDIR: {workdir}")
    print(f"TASK: {args.task}")
    final_content: str | None = None
    transcript: list[dict[str, Any]] = []

    for iteration in range(1, args.max_iterations + 1):
        payload = {
            "model": args.model,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": args.tool_choice,
            "temperature": args.temperature,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
        }
        response = post_chat(args.base_url, payload, args.http_timeout)
        choice = response["choices"][0]
        assistant = choice["message"]
        transcript.append({"iteration": iteration, "assistant": assistant, "finish_reason": choice.get("finish_reason")})

        print(f"\n=== ITERATION {iteration} ASSISTANT ===")
        print_json("message", assistant)
        print(f"finish_reason: {choice.get('finish_reason')}")

        tool_calls = assistant.get("tool_calls") or []
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": assistant.get("content"),
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                result = execute_tool(workdir, name, function.get("arguments", "{}"))
                transcript.append({"tool_call_id": call.get("id"), "tool": name, "result": result})
                print_json(f"TOOL {call.get('id')} {name}", result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue

        final_content = assistant.get("content")
        break
    else:
        print("\nSTOPPED: max iterations reached without final answer")

    verification = verify_task(workdir, args.expect_file, args.expect_contains, args.expect_output)
    print("\n=== FINAL ===")
    print_json("final_content", final_content)
    print_json("verification", verification)
    print("\n=== FULL_TRANSCRIPT_JSON ===")
    print(json.dumps(transcript, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if verification["ok"] else 2


def verify_task(workdir: Path, expect_file: str | None, expect_contains: str | None, expect_output: str | None) -> dict[str, Any]:
    checks: dict[str, Any] = {"ok": True}
    if expect_file:
        path = sandbox_path(workdir, expect_file)
        exists = path.is_file()
        content = path.read_text(encoding="utf-8") if exists else None
        checks["file_exists"] = exists
        checks["file_content"] = content
        if not exists:
            checks["ok"] = False
        if expect_contains is not None:
            contains = bool(content is not None and expect_contains in content)
            checks["file_contains_expected"] = contains
            if not contains:
                checks["ok"] = False
    if expect_output and expect_file:
        path = sandbox_path(workdir, expect_file)
        if path.suffix == ".py" and path.is_file():
            completed = subprocess.run(
                [sys.executable, path.name],
                cwd=workdir,
                text=True,
                capture_output=True,
                timeout=15,
            )
            checks["postcheck_run"] = {
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            if completed.returncode != 0 or expect_output not in completed.stdout:
                checks["ok"] = False
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1", help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", default="qwen2.5-0.5b", help="model id sent to the server")
    parser.add_argument("--workdir", default="target/coding-agent-workspace", help="sandbox directory for file and terminal tools")
    parser.add_argument("--task", default="Create a file named hello.py in the working directory that prints 'Hello, Squad!', then run it and tell me the output.")
    parser.add_argument("--tool-choice", default="auto", choices=["auto", "required", "none"], help="OpenAI tool_choice mode")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--http-timeout", type=int, default=120)
    parser.add_argument("--expect-file", default="hello.py")
    parser.add_argument("--expect-contains", default="Hello, Squad!")
    parser.add_argument("--expect-output", default="Hello, Squad!")
    parser.add_argument("--clean", action="store_true", help="delete workdir before running")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_agent(parse_args()))
