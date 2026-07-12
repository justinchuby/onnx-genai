#!/usr/bin/env python3
"""Minimal Hermes-style coding-agent loop for the onnx-genai OpenAI server.

WARNING: LOCAL TESTING ONLY. This example harness executes model-chosen tools
only inside a throwaway workspace. File tools resolve paths through a workspace
confinement check, and run_command rejects shell syntax, dangerous commands, and
paths that escape the workspace before executing an allow-listed argv without a
shell. This is still a test harness rather than a production isolation boundary.
The onnx-genai server itself never executes tools — it only parses tool calls;
tool execution/sandboxing is the calling agent's responsibility.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
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
            "description": "Run an allow-listed command in the working directory and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]

OUTPUT_LIMIT_BYTES = 64 * 1024
COMMAND_TIMEOUT_SECONDS = 15
DANGEROUS_COMMANDS = {
    "bash",
    "chmod",
    "chown",
    "curl",
    "dd",
    "eval",
    "fish",
    "kill",
    "mkfs",
    "mount",
    "nc",
    "netcat",
    "perl",
    "pkill",
    "python-config",
    "reboot",
    "rm",
    "rsync",
    "scp",
    "sh",
    "shutdown",
    "ssh",
    "sudo",
    "umount",
    "wget",
    "zsh",
}
SHELL_TOKENS = {"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>", "1>", "1>>"}
ALLOWED_COMMANDS = {"cat", "echo", "grep", "head", "ls", "mkdir", "pwd", "python", "python3", "sed", "tail", "touch"}
SAFE_PYTHON_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}
SAFE_PYTHON_NODES = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Del,
    ast.Constant,
    ast.Call,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.If,
    ast.For,
    ast.While,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Return,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.comprehension,
    ast.JoinedStr,
    ast.FormattedValue,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def sandbox_path(workdir: Path, requested: str) -> Path:
    workdir = workdir.resolve()
    if not requested:
        raise ValueError("path is required")
    candidate = (workdir / requested).resolve() if not os.path.isabs(requested) else Path(requested).resolve()
    try:
        candidate.relative_to(workdir)
    except ValueError as exc:
        raise ValueError(f"path escapes workdir: {requested}") from exc
    return candidate


def truncate_output(value: str, limit: int = OUTPUT_LIMIT_BYTES) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    truncated = encoded[:limit].decode("utf-8", errors="replace")
    return f"{truncated}\n...[truncated to {limit} bytes]"


def reject_dangerous_tokens(command: str, argv: list[str]) -> None:
    lowered = command.lower()
    dangerous_fragments = (
        "rm -rf /",
        "rm -fr /",
        "curl ",
        "wget ",
        "nc ",
        "netcat ",
        "sudo ",
        "shutdown",
        "reboot",
        "/dev/",
        "chmod -r",
        "chown -r",
    )
    if any(fragment in lowered for fragment in dangerous_fragments):
        raise ValueError("command rejected: dangerous token")
    for token in argv:
        if token in SHELL_TOKENS or "`" in token or "$(" in token or "${" in token:
            raise ValueError(f"command rejected: shell syntax is not allowed ({token})")


def looks_like_path(token: str) -> bool:
    return token in {".", ".."} or token.startswith(("/", "./", "../", "~")) or "/" in token


def validate_existing_path_arg(workdir: Path, token: str) -> Path:
    path = sandbox_path(workdir, token)
    if not path.exists():
        raise ValueError(f"path does not exist in workdir: {token}")
    return path


def validate_create_path_arg(workdir: Path, token: str) -> Path:
    path = sandbox_path(workdir, token)
    sandbox_path(workdir, str(path.parent))
    return path


def relative_to_workdir(workdir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(workdir.resolve()))


def validate_safe_python_source(source: str, script: str) -> None:
    try:
        tree = ast.parse(source, filename=script)
    except SyntaxError as exc:
        raise ValueError(f"python script rejected: syntax error: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_PYTHON_NODES):
            raise ValueError(f"python script rejected: unsafe syntax {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_PYTHON_BUILTINS:
                raise ValueError("python script rejected: only safe builtin calls are allowed")


def run_sandboxed_python(script: Path, script_args: list[str]) -> None:
    source = script.read_text(encoding="utf-8")
    validate_safe_python_source(source, str(script))
    old_argv = sys.argv
    sys.argv = [str(script), *script_args]
    try:
        exec(compile(source, str(script), "exec"), {"__builtins__": SAFE_PYTHON_BUILTINS}, {})
    finally:
        sys.argv = old_argv


def validate_command(workdir: Path, command: str) -> tuple[list[str], bool]:
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"command rejected: cannot parse command: {exc}") from exc
    if not argv:
        raise ValueError("command rejected: empty command")
    reject_dangerous_tokens(command, argv)

    executable = Path(argv[0]).name
    if executable not in ALLOWED_COMMANDS or executable in DANGEROUS_COMMANDS or Path(argv[0]).parent != Path("."):
        raise ValueError(f"command rejected: command is not allow-listed: {argv[0]}")

    if executable == "pwd":
        if len(argv) != 1:
            raise ValueError("pwd does not accept arguments in this harness")
        return [executable], False

    if executable == "echo":
        return [executable, *argv[1:]], False

    if executable in {"python", "python3"}:
        if len(argv) < 2 or argv[1].startswith("-"):
            raise ValueError("python command rejected: use python <workspace-file.py>")
        script = validate_existing_path_arg(workdir, argv[1])
        if script.suffix != ".py" or not script.is_file():
            raise ValueError("python command rejected: script must be a .py file in the workspace")
        for token in argv[2:]:
            if looks_like_path(token):
                validate_existing_path_arg(workdir, token)
        return [
            sys.executable,
            str(Path(__file__).resolve()),
            "--workdir",
            str(workdir.resolve()),
            "--_sandbox-python",
            relative_to_workdir(workdir, script),
            *argv[2:],
        ], True

    if executable == "mkdir":
        allowed_options = {"-p"}
        paths = [token for token in argv[1:] if token not in allowed_options]
        if not paths or any(token.startswith("-") and token not in allowed_options for token in argv[1:]):
            raise ValueError("mkdir command rejected: only -p and workspace paths are allowed")
        return [executable, *[("-p" if token == "-p" else relative_to_workdir(workdir, validate_create_path_arg(workdir, token))) for token in argv[1:]]], False

    if executable == "touch":
        if not argv[1:] or any(token.startswith("-") for token in argv[1:]):
            raise ValueError("touch command rejected: options are not allowed")
        return [executable, *[relative_to_workdir(workdir, validate_create_path_arg(workdir, token)) for token in argv[1:]]], False

    if executable == "ls":
        allowed_options = {"-l", "-la", "-al", "-a", "-1"}
        validated = [executable]
        for token in argv[1:] or ["."]:
            if token.startswith("-"):
                if token not in allowed_options:
                    raise ValueError("ls command rejected: unsupported option")
                validated.append(token)
            else:
                validated.append(relative_to_workdir(workdir, validate_existing_path_arg(workdir, token)))
        return validated, False

    if executable in {"cat", "head", "tail"}:
        if not argv[1:]:
            raise ValueError(f"{executable} command rejected: at least one workspace file is required")
        allowed_options = {"-n"}
        validated = [executable]
        skip_next = False
        for index, token in enumerate(argv[1:]):
            if skip_next:
                skip_next = False
                continue
            if token == "-n":
                if index + 2 >= len(argv) or not argv[index + 2].lstrip("-").isdigit():
                    raise ValueError(f"{executable} command rejected: -n requires a number")
                validated.extend([token, argv[index + 2]])
                skip_next = True
            elif token.startswith("-"):
                raise ValueError(f"{executable} command rejected: unsupported option")
            else:
                validated.append(relative_to_workdir(workdir, validate_existing_path_arg(workdir, token)))
        return validated, False

    if executable == "grep":
        allowed_options = {"-n", "-i"}
        validated = [executable]
        pattern_seen = False
        file_count = 0
        for token in argv[1:]:
            if token.startswith("-") and not pattern_seen:
                if token not in allowed_options:
                    raise ValueError("grep command rejected: unsupported option")
                validated.append(token)
            elif not pattern_seen:
                validated.append(token)
                pattern_seen = True
            else:
                validated.append(relative_to_workdir(workdir, validate_existing_path_arg(workdir, token)))
                file_count += 1
        if not pattern_seen:
            raise ValueError("grep command rejected: missing pattern")
        if file_count == 0:
            raise ValueError("grep command rejected: at least one workspace file is required")
        return validated, False

    if executable == "sed":
        if len(argv) < 3:
            raise ValueError("sed command rejected: use sed <script> <workspace-file>...")
        if argv[1].startswith("-") or any(ch in argv[1] for ch in "erw"):
            raise ValueError("sed command rejected: only simple read-only scripts are allowed")
        return [executable, argv[1], *[relative_to_workdir(workdir, validate_existing_path_arg(workdir, token)) for token in argv[2:]]], False

    raise ValueError(f"command rejected: unsupported command: {executable}")


def run_allowed_command(workdir: Path, command: str) -> dict[str, Any]:
    argv, _ = validate_command(workdir, command)
    completed = subprocess.run(
        argv,
        cwd=workdir,
        shell=False,
        text=True,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    return {
        "ok": completed.returncode == 0,
        "command": command,
        "argv": argv,
        "exit_code": completed.returncode,
        "stdout": truncate_output(completed.stdout),
        "stderr": truncate_output(completed.stderr),
    }


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
            entries = sorted(p.name + ("/" if p.is_dir() and not p.is_symlink() else "") for p in path.iterdir())
            return {"ok": True, "path": str(path.relative_to(workdir)), "entries": entries}

        if name == "run_command":
            command = str(args["command"])
            return run_allowed_command(workdir, command)

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


def prepare_workdir(workdir_arg: str | None, clean: bool, repo_root: Path) -> Path:
    if workdir_arg:
        workdir = Path(workdir_arg).resolve()
    else:
        parent = repo_root / "target" / "coding-agent-workspaces"
        parent.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="run-", dir=parent)).resolve()
    try:
        workdir.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit(f"--workdir must be inside the current repository: {workdir}") from exc
    if workdir in {repo_root, repo_root / "target"}:
        raise SystemExit("--workdir must not be the repository root or target root")
    if clean and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def run_agent(args: argparse.Namespace) -> int:
    repo_root = Path.cwd().resolve()

    if args._sandbox_python:
        if not args.workdir:
            raise SystemExit("--workdir is required for internal sandboxed python execution")
        workdir = Path(args.workdir).resolve()
        script = sandbox_path(workdir, args._sandbox_python[0])
        try:
            run_sandboxed_python(script, args._sandbox_python[1:])
            return 0
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.self_test:
        return run_self_test(repo_root)

    workdir = prepare_workdir(args.workdir, args.clean, repo_root)

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
            result = run_allowed_command(workdir, f"python {shlex.quote(relative_to_workdir(workdir, path))}")
            checks["postcheck_run"] = result
            if not result["ok"] or expect_output not in result["stdout"]:
                checks["ok"] = False
    return checks


def assert_blocked(result: dict[str, Any], expected: str) -> None:
    if result.get("ok") is not False or expected not in result.get("error", ""):
        raise AssertionError(f"expected blocked {expected!r}, got {result}")


def run_self_test(repo_root: Path) -> int:
    parent = repo_root / "target" / "coding-agent-selftest"
    parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="case-", dir=parent)).resolve()

    execute_tool(workdir, "write_file", json.dumps({"path": "hello.py", "content": "print('Hello, Squad!')\n"}))
    benign = execute_tool(workdir, "run_command", json.dumps({"command": "python hello.py"}))
    if not benign["ok"] or "Hello, Squad!" not in benign["stdout"]:
        raise AssertionError(f"benign python task failed: {benign}")

    blocked_abs = execute_tool(workdir, "read_file", json.dumps({"path": "/etc/hostname"}))
    assert_blocked(blocked_abs, "escapes workdir")
    blocked_cat_abs = execute_tool(workdir, "run_command", json.dumps({"command": "cat /etc/hostname"}))
    assert_blocked(blocked_cat_abs, "escapes workdir")
    blocked_parent = execute_tool(workdir, "write_file", json.dumps({"path": "../escape.txt", "content": "nope"}))
    assert_blocked(blocked_parent, "escapes workdir")
    blocked_rm = execute_tool(workdir, "run_command", json.dumps({"command": "rm -rf /"}))
    assert_blocked(blocked_rm, "dangerous token")
    blocked_shell = execute_tool(workdir, "run_command", json.dumps({"command": "echo ok | sh"}))
    assert_blocked(blocked_shell, "shell syntax")
    execute_tool(workdir, "write_file", json.dumps({"path": "danger.py", "content": "import os\n"}))
    blocked_python = execute_tool(workdir, "run_command", json.dumps({"command": "python danger.py"}))
    if blocked_python.get("ok") is not False or "unsafe syntax Import" not in blocked_python.get("stderr", ""):
        raise AssertionError(f"expected unsafe python to be blocked, got {blocked_python}")

    if hasattr(os, "symlink"):
        outside = parent / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        link = workdir / "outside-link"
        try:
            link.symlink_to(outside, target_is_directory=True)
            blocked_symlink = execute_tool(workdir, "write_file", json.dumps({"path": "outside-link/escape.txt", "content": "nope"}))
            assert_blocked(blocked_symlink, "escapes workdir")
        except FileExistsError:
            pass

    verification = verify_task(workdir, "hello.py", "Hello, Squad!", "Hello, Squad!")
    if not verification["ok"]:
        raise AssertionError(f"verification failed: {verification}")

    print_json("self_test", {"ok": True, "workdir": str(workdir), "benign_stdout": benign["stdout"].strip()})
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1", help="OpenAI-compatible /v1 base URL")
    parser.add_argument("--model", default="qwen2.5-0.5b", help="model id sent to the server")
    parser.add_argument("--workdir", default=None, help="sandbox directory for file and terminal tools; defaults to a fresh target/coding-agent-workspaces directory")
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
    parser.add_argument("--self-test", action="store_true", help="run local sandbox guard self-tests")
    parser.add_argument("--_sandbox-python", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_agent(parse_args()))
