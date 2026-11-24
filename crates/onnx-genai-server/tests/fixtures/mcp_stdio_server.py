"""Local stdio MCP peer used to exercise actual subprocess lifecycle and RPC."""

import argparse
import json
import os
import sys
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--events", required=True)
parser.add_argument("--stderr-bytes", type=int, default=0)
parser.add_argument("--mode", default="normal")
parser.add_argument("--tool-prefix", default="")
parser.add_argument("--only-tool")
args = parser.parse_args()
lock = threading.Lock()
cancelled = set()
ping = threading.Event()


def event(value):
    with open(args.events, "a", encoding="utf-8") as output:
        output.write(json.dumps(value) + "\n")


def send(value):
    with lock:
        sys.stdout.write(json.dumps(value) + "\n")
        sys.stdout.flush()


def reply(request, value):
    if request["id"] not in cancelled:
        send({"jsonrpc": "2.0", "id": request["id"], "result": value})


def call(request):
    params = request["params"]
    name = params["name"].removeprefix(args.tool_prefix)
    values = params.get("arguments", {})
    event({"call": name, "arguments": values, "id": request["id"]})
    if name == "rpc_error":
        send(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "error": {"code": -32602, "message": "RPC_ARGUMENT_ERROR"},
            }
        )
        return
    if name == "big":
        reply(
            request,
            {"content": [{"type": "text", "text": "b" * values.get("bytes", 300000)}]},
        )
        return
    if name == "slow":
        time.sleep(5)
    if name == "crash":
        import subprocess

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        event({"grandchild": child.pid})
        os._exit(7)
    if name == "ping_peer":
        send({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"})
        if not ping.wait(2):
            reply(
                request,
                {
                    "content": [{"type": "text", "text": "ping not answered"}],
                    "isError": True,
                },
            )
            return
    text = json.dumps(
        {"arguments": values, "cwd": os.getcwd(), "env": os.getenv("NXRT_MCP_FIXTURE")}
    )
    reply(
        request,
        {
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "data": "ignored", "mimeType": "image/png"},
                {"type": "text", "text": "second"},
            ],
            "isError": name == "fail",
        },
    )


event({"pid": os.getpid()})
if args.stderr_bytes:
    sys.stderr.write("e" * args.stderr_bytes)
    sys.stderr.flush()
for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if message.get("id") == "server-ping" and "result" in message:
        ping.set()
    elif method == "initialize":
        event({"initialize": message["params"]})
        if args.mode == "slow_init":
            time.sleep(3)
        if args.mode == "exit_init":
            sys.stderr.write("MCP_FIXTURE_BOOT_ERROR\n")
            sys.stderr.flush()
            os._exit(9)
        if args.mode == "malformed":
            sys.stdout.write("{broken\n")
            sys.stdout.flush()
        reply(
            message,
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            },
        )
    elif method == "notifications/initialized":
        event({"initialized": True})
    elif method == "tools/list":
        names = (
            [args.only_tool]
            if args.only_tool
            else ["echo", "fail", "slow", "crash", "ping_peer", "rpc_error", "big"]
        )
        next_cursor = None
        if args.mode == "pagination":
            if message.get("params", {}).get("cursor"):
                names = names[2:]
            else:
                names = names[:2]
                next_cursor = "page-two"
        if args.mode == "repeated_cursor":
            next_cursor = "same-cursor"
        reply(
            message,
            {
                "tools": [
                    {
                        "name": args.tool_prefix + name,
                        "description": name,
                        "inputSchema": {"type": "object"},
                    }
                    for name in names
                ],
                "nextCursor": next_cursor,
            },
        )
    elif method == "tools/call":
        threading.Thread(target=call, args=(message,), daemon=True).start()
    elif method == "notifications/cancelled":
        cancelled.add(message["params"]["requestId"])
        event({"cancelled": message["params"]["requestId"]})
    elif method == "ping":
        reply(message, {})
