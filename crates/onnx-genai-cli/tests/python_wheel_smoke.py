"""Installed Native CLI acceptance; requires a fresh dependency-free environment."""

from contextlib import contextmanager
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
import urllib.error
import urllib.request
import wave


CLI = str(Path(sysconfig.get_path("scripts")) / ("onnx-genai.exe" if os.name == "nt" else "onnx-genai"))

def process_options() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate the owned local process group and reap its direct child."""
    if os.name == "nt":
        if process.poll() is None:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=5)
    finally:
        # The parent may have exited before a descendant; kill the owned group too.
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@contextmanager
def contained_process(command, **kwargs):
    """Own the child before delivering cancellation, then always clean it up."""
    watched = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        watched.append(signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in watched}
    process = None
    pending = None

    def interrupted(number, _frame):
        nonlocal pending
        if process is None:
            pending = number
        else:
            raise KeyboardInterrupt(f"wheel acceptance interrupted by signal {number}")

    try:
        for number in watched:
            signal.signal(number, interrupted)
        process = subprocess.Popen(command, **kwargs, **process_options())
        if pending is not None:
            raise KeyboardInterrupt(f"wheel acceptance interrupted during child creation: {pending}")
        yield process
    finally:
        try:
            for number in watched:
                signal.signal(number, signal.SIG_IGN)
            if process is not None:
                stop_process_tree(process)
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def run_command(command: list[str], environment: dict, *, success=True, timeout=120):
    with contained_process(command, env=environment, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(timeout=timeout)
        if (process.returncode == 0) != success:
            raise RuntimeError(
                f"installed CLI returned {process.returncode} for {command}:\n{stdout}\n{stderr}"
            )
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def http_json(url: str, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=2) as response:
        return json.load(response)


def check_server(fixture: Path, environment: dict, temporary: Path) -> None:
    # The CLI does not expose a bound port for --addr :0. Retry only a proven
    # address-in-use race between choosing an ephemeral port and starting it.
    for attempt in range(3):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        model_id = f"native-wheel-{port}"
        log = temporary / f"server-{attempt}.log"
        with log.open("w+") as output:
            with contained_process(
                [CLI, "serve", str(fixture), "--addr", f"127.0.0.1:{port}",
                 "--model-id", model_id, "--cpu-cores", "1"],
                env=environment, stdout=output, stderr=output,
            ) as process:
                base = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 30
                ready = False
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        health = http_json(base + "/health")
                        if health.get("status") == "ok":
                            models = http_json(base + "/v1/models")
                            if [model.get("id") for model in models.get("data", [])] == [model_id]:
                                ready = True
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(0.05)
                if not ready:
                    output.flush()
                    diagnostic = log.read_text()
                    if "address already in use" in diagnostic.lower() and attempt < 2:
                        continue
                    raise RuntimeError(f"installed Native server did not become healthy: {diagnostic}")
                result = http_json(base + "/completion", {
                    "prompt": "hi", "n_predict": 1, "temperature": 0,
                    "return_tokens": True, "cache_prompt": False,
                })
                if result.get("model") != model_id or result.get("tokens_predicted") != 1 or len(result.get("tokens", [])) != 1:
                    raise RuntimeError(f"installed server did not execute a Native completion: {result}")
                return
    raise RuntimeError("installed server could not reserve a loopback port after three attempts")


def main() -> None:
    fixture = Path(sys.argv[1]).resolve()
    workflow_fixtures = fixture.parent / "onnx_genai_workflows"
    for package in ("onnxruntime", "onnxruntime_ep_mlx"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"wheel acceptance requires no installed {package}; use a fresh venv")
    distribution = importlib.metadata.distribution("onnx-genai-server")
    requirements = distribution.requires or []
    if any("extra ==" not in requirement for requirement in requirements):
        raise RuntimeError(f"Native wheel declares a mandatory dependency: {requirements}")
    specification = importlib.util.find_spec("onnx_genai_server")
    installed = distribution.locate_file("onnx_genai_server/__init__.py").resolve()
    if specification is None or specification.origin is None or Path(specification.origin).resolve() != installed:
        raise RuntimeError("wheel acceptance imported a source checkout instead of the installed launcher")
    environment = os.environ.copy()
    for name in ("ONNX_GENAI_BACKEND", "ONNX_GENAI_ORT_LIB_DIR", "PYTHONPATH", "LLAMA_API_KEY", "LLAMA_ARG_API_KEY_FILE"):
        environment.pop(name, None)
    environment["ONNX_GENAI_EP"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="nxrt-wheel-smoke-") as directory:
        temporary = Path(directory)
        profile = temporary / "profile.json"
        environment["ONNX_GENAI_ORT_LIB"] = str(temporary / "missing-reference-runtime")

        def run(arguments, success=True):
            return run_command([CLI, *arguments], environment, success=success)

        run(["version"])
        run(["--profile-json", str(profile), "generate", str(fixture), "hi",
             "--max-new-tokens", "2", "--greedy", "--no-stats", "--cpu-cores", "1"])
        actual = json.loads(profile.read_text())
        if actual["decode_backend"] != "native" or actual["execution_provider"] != "native":
            raise RuntimeError(f"installed wheel did not execute Native generation: {actual}")
        audio = temporary / "clip.wav"
        samples = [-0.5, -0.5, 0.125, -0.5, 0.0, 0.5, 0.5, 0.125, 0.5]
        with wave.open(str(audio), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"".join(struct.pack("<h", round(value * 32767)) for value in samples))
        transcript = run(["--profile-json", str(profile), "transcribe",
                          str(workflow_fixtures / "transcription_ctc"), str(audio),
                          "--format", "json", "--min-segment-seconds", "0", "--cpu-cores", "1"])
        rows = [json.loads(line) for line in transcript.stdout.splitlines()]
        if len(rows) != 1 or rows[0]["text"] != "bb aa" or rows[0]["index"] != 0:
            raise RuntimeError(f"installed Native transcription returned wrong content: {rows}")
        actual = json.loads(profile.read_text())
        if actual["decode_backend"] != "native" or actual["execution_provider"] != "native-cpu":
            raise RuntimeError(f"installed wheel did not execute Native transcription: {actual}")
        check_server(fixture, environment, temporary)
        rejected = run(["generate", str(fixture), "hi", "--backend", "ort",
                        "--max-new-tokens", "1", "--no-stats", "--cpu-cores", "1"], success=False)
        if "failed to load onnx runtime" not in rejected.stderr.lower():
            raise RuntimeError(f"explicit ORT failed for an unexpected reason: {rejected.stderr}")
    print("Installed wheel: Native generation, transcription and HTTP completion passed without ORT; explicit missing ORT rejected.")


if __name__ == "__main__":
    main()
