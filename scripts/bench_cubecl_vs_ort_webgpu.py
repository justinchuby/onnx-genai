#!/usr/bin/env python3
"""Compare nxrt CubeCL WebGPU plugin EP against Microsoft's ORT WebGPU plugin EP.

The script deliberately drives both plugins through one dlopen'd ONNX Runtime C
API instance. It does not use the Python onnxruntime module for execution,
because the Python wheel may carry a different ORT version than this repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "target" / "bench_cubecl_vs_ort_webgpu"
HELPER_C = BUILD_DIR / "ort_ep_bench_helper.c"
HELPER_BIN = BUILD_DIR / "ort_ep_bench_helper"
MODELS_DIR = BUILD_DIR / "models"
PROFILES_DIR = BUILD_DIR / "profiles"

CUBECL_REG = "onnxruntime_cubecl_webgpu_ep"
CUBECL_EP = "onnxruntime_cubecl_webgpu_ep"
WEBGPU_REG = "webgpu"
WEBGPU_EP = "WebGpuExecutionProvider"


HELPER_SOURCE = r'''
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "onnxruntime_c_api.h"

static const OrtApi* api = NULL;

static void json_escape_print(const char* s) {
  putchar('"');
  for (; s && *s; ++s) {
    if (*s == '"' || *s == '\\') { putchar('\\'); putchar(*s); }
    else if (*s == '\n') printf("\\n");
    else if (*s == '\r') printf("\\r");
    else putchar(*s);
  }
  putchar('"');
}

static int fail_status(OrtStatus* st, const char* stage) {
  if (!st) return 0;
  const char* msg = api && api->GetErrorMessage ? api->GetErrorMessage(st) : "(no api msg)";
  printf("{\"event\":\"error\",\"stage\":");
  json_escape_print(stage);
  printf(",\"message\":");
  json_escape_print(msg ? msg : "(null)");
  printf("}\n");
  fflush(stdout);
  if (api && api->ReleaseStatus) api->ReleaseStatus(st);
  return 1;
}

#define CHECK(stage, expr) do { OrtStatus* _s=(expr); if (fail_status(_s, stage)) return 2; } while(0)

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint16_t f32_to_f16(float f) {
  union { float f; uint32_t u; } v;
  v.f = f;
  uint32_t sign = (v.u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = v.u & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant = (mant | 0x800000u) >> (uint32_t)(1 - exp);
    return (uint16_t)(sign | ((mant + 0x1000u) >> 13));
  }
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000u) >> 13));
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
  uint32_t mant = (uint32_t)h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) out = sign;
    else {
      exp = 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
      mant &= 0x3ffu;
      out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  union { uint32_t u; float f; } v;
  v.u = out;
  return v.f;
}

typedef struct {
  const char* name;
  OrtSession* session;
  int valid;
  char error[1024];
} Arm;

static int elem_type(const char* dtype) {
  return strcmp(dtype, "f16") == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

static size_t elem_size(const char* dtype) {
  return strcmp(dtype, "f16") == 0 ? sizeof(uint16_t) : sizeof(float);
}

static float pattern(size_t i) {
  int v = (int)((i * 37u) % 23u) - 11;
  return (float)v / 8.0f;
}

static void fill_buffer(void* buf, size_t n, const char* dtype, int offset) {
  if (strcmp(dtype, "f16") == 0) {
    uint16_t* p = (uint16_t*)buf;
    for (size_t i = 0; i < n; ++i) p[i] = f32_to_f16(pattern(i + (size_t)offset));
  } else {
    float* p = (float*)buf;
    for (size_t i = 0; i < n; ++i) p[i] = pattern(i + (size_t)offset);
  }
}

static float read_elem(const void* buf, size_t i, const char* dtype) {
  if (strcmp(dtype, "f16") == 0) return f16_to_f32(((const uint16_t*)buf)[i]);
  return ((const float*)buf)[i];
}

static int make_input_value(OrtMemoryInfo* mem, void* data, size_t elems, const int64_t* shape, size_t rank, const char* dtype, OrtValue** out) {
  return fail_status(api->CreateTensorWithDataAsOrtValue(mem, data, elems * elem_size(dtype), shape, rank, elem_type(dtype), out), "CreateTensorWithDataAsOrtValue");
}

static int prepare_inputs(const char* op, const char* dtype, int64_t a, int64_t b, int64_t c,
                          OrtMemoryInfo* mem, OrtValue** inputs, const char*** names_out,
                          size_t* input_count, size_t* output_elems) {
  static const char* elem_names[2] = {"X", "Y"};
  static const char* relu_names[1] = {"X"};
  static const char* mm_names[2] = {"A", "B"};
  if (strcmp(op, "Relu") == 0) {
    size_t n = (size_t)a;
    void* x = calloc(n, elem_size(dtype));
    fill_buffer(x, n, dtype, 0);
    int64_t shape[1] = {a};
    if (make_input_value(mem, x, n, shape, 1, dtype, &inputs[0])) return 1;
    *names_out = relu_names; *input_count = 1; *output_elems = n; return 0;
  }
  if (strcmp(op, "MatMul") == 0) {
    size_t an = (size_t)(a * b), bn = (size_t)(b * c);
    void* x = calloc(an, elem_size(dtype));
    void* y = calloc(bn, elem_size(dtype));
    fill_buffer(x, an, dtype, 0);
    fill_buffer(y, bn, dtype, 7919);
    int64_t ashape[2] = {a, b}, bshape[2] = {b, c};
    if (make_input_value(mem, x, an, ashape, 2, dtype, &inputs[0])) return 1;
    if (make_input_value(mem, y, bn, bshape, 2, dtype, &inputs[1])) return 1;
    *names_out = mm_names; *input_count = 2; *output_elems = (size_t)(a * c); return 0;
  }
  size_t n = (size_t)a;
  void* x = calloc(n, elem_size(dtype));
  void* y = calloc(n, elem_size(dtype));
  fill_buffer(x, n, dtype, 0);
  fill_buffer(y, n, dtype, 7919);
  int64_t shape[1] = {a};
  if (make_input_value(mem, x, n, shape, 1, dtype, &inputs[0])) return 1;
  if (make_input_value(mem, y, n, shape, 1, dtype, &inputs[1])) return 1;
  *names_out = elem_names; *input_count = 2; *output_elems = n; return 0;
}

static int run_once(OrtSession* session, const char** input_names, const OrtValue** inputs, size_t input_count,
                    const char* output_name, OrtValue** output) {
  *output = NULL;
  return fail_status(api->Run(session, NULL, input_names, inputs, input_count, &output_name, 1, output), "Run");
}

static int compare_outputs(OrtValue* ref, OrtValue* got, size_t n, const char* dtype, double rtol, double atol, double* max_abs, double* max_rel) {
  void* r = NULL; void* g = NULL;
  if (fail_status(api->GetTensorMutableData(ref, &r), "GetTensorMutableData(ref)")) return 0;
  if (fail_status(api->GetTensorMutableData(got, &g), "GetTensorMutableData(got)")) return 0;
  int ok = 1;
  *max_abs = 0.0; *max_rel = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double rv = (double)read_elem(r, i, dtype);
    double gv = (double)read_elem(g, i, dtype);
    double abs_err = fabs(gv - rv);
    double rel_err = abs_err / fmax(fabs(rv), 1e-12);
    if (abs_err > *max_abs) *max_abs = abs_err;
    if (rel_err > *max_rel) *max_rel = rel_err;
    if (abs_err > atol + rtol * fabs(rv)) ok = 0;
  }
  return ok;
}

static OrtSession* create_session(OrtEnv* env, const char* model, const OrtEpDevice* device, const char* profile_prefix, char* err, size_t err_len) {
  OrtSessionOptions* so = NULL;
  OrtSession* session = NULL;
  OrtStatus* st = api->CreateSessionOptions(&so);
  if (st) {
    snprintf(err, err_len, "CreateSessionOptions: %s", api->GetErrorMessage(st));
    api->ReleaseStatus(st);
    return NULL;
  }
  if (device) {
    const OrtEpDevice* arr[1] = {device};
    st = api->SessionOptionsAppendExecutionProvider_V2(so, env, arr, 1, NULL, NULL, 0);
    if (st) {
      snprintf(err, err_len, "SessionOptionsAppendExecutionProvider_V2: %s", api->GetErrorMessage(st));
      api->ReleaseStatus(st);
      api->ReleaseSessionOptions(so);
      return NULL;
    }
  }
  if (profile_prefix && profile_prefix[0]) {
    st = api->EnableProfiling(so, profile_prefix);
    if (st) {
      snprintf(err, err_len, "EnableProfiling: %s", api->GetErrorMessage(st));
      api->ReleaseStatus(st);
      api->ReleaseSessionOptions(so);
      return NULL;
    }
  }
  st = api->CreateSession(env, model, so, &session);
  api->ReleaseSessionOptions(so);
  if (st) {
    snprintf(err, err_len, "CreateSession: %s", api->GetErrorMessage(st));
    api->ReleaseStatus(st);
    return NULL;
  }
  return session;
}

static const OrtEpDevice* find_device(OrtEnv* env, const char* ep_name) {
  const OrtEpDevice* const* devices = NULL;
  size_t ndev = 0;
  if (fail_status(api->GetEpDevices(env, &devices, &ndev), "GetEpDevices")) return NULL;
  for (size_t i = 0; i < ndev; ++i) {
    const char* name = api->EpDevice_EpName(devices[i]);
    if (name && strcmp(name, ep_name) == 0) return devices[i];
  }
  return NULL;
}

static void end_profile(OrtSession* session, const char* arm) {
  OrtAllocator* alloc = NULL;
  char* path = NULL;
  if (api->GetAllocatorWithDefaultOptions(&alloc) || api->SessionEndProfiling(session, alloc, &path)) {
    printf("{\"event\":\"profile\",\"arm\":");
    json_escape_print(arm);
    printf(",\"path\":null}\n");
    return;
  }
  printf("{\"event\":\"profile\",\"arm\":");
  json_escape_print(arm);
  printf(",\"path\":");
  json_escape_print(path);
  printf("}\n");
  if (path) api->AllocatorFree(alloc, path);
}

int main(int argc, char** argv) {
  if (argc != 18) {
    fprintf(stderr, "usage: helper <ort> <cubecl_plugin> <webgpu_plugin> <model> <op> <dtype> <a> <b> <c> <warmups> <iterations> <samples> <profile_dir> <rtol> <atol> <cubecl_ep> <webgpu_ep>\n");
    return 2;
  }
  const char* ort_path = argv[1];
  const char* cubecl_plugin = argv[2];
  const char* webgpu_plugin = argv[3];
  const char* model = argv[4];
  const char* op = argv[5];
  const char* dtype = argv[6];
  int64_t a = atoll(argv[7]), b = atoll(argv[8]), c = atoll(argv[9]);
  int warmups = atoi(argv[10]);
  int iterations = atoi(argv[11]);
  int samples = atoi(argv[12]);
  const char* profile_dir = argv[13];
  double rtol = atof(argv[14]);
  double atol = atof(argv[15]);
  const char* cubecl_ep = argv[16];
  const char* webgpu_ep = argv[17];

  void* h = dlopen(ort_path, RTLD_NOW | RTLD_LOCAL);
  if (!h) { printf("{\"event\":\"error\",\"stage\":\"dlopen_ort\",\"message\":"); json_escape_print(dlerror()); printf("}\n"); return 2; }
  const OrtApiBase* (*getbase)(void) = (const OrtApiBase* (*)(void))dlsym(h, "OrtGetApiBase");
  if (!getbase) { printf("{\"event\":\"error\",\"stage\":\"dlsym_OrtGetApiBase\",\"message\":"); json_escape_print(dlerror()); printf("}\n"); return 2; }
  const OrtApiBase* base = getbase();
  api = base->GetApi(ORT_API_VERSION);
  if (!api) { printf("{\"event\":\"error\",\"stage\":\"GetApi\",\"message\":\"null OrtApi\"}\n"); return 2; }

  printf("{\"event\":\"metadata\",\"ort_version\":");
  json_escape_print(base->GetVersionString());
  printf("}\n");

  OrtEnv* env = NULL;
  CHECK("CreateEnv", api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "cubecl_vs_ort_webgpu", &env));
  CHECK("RegisterExecutionProviderLibrary(cubecl)", api->RegisterExecutionProviderLibrary(env, "onnxruntime_cubecl_webgpu_ep", cubecl_plugin));
  CHECK("RegisterExecutionProviderLibrary(webgpu)", api->RegisterExecutionProviderLibrary(env, "webgpu", webgpu_plugin));

  const OrtEpDevice* cubecl_dev = find_device(env, cubecl_ep);
  const OrtEpDevice* webgpu_dev = find_device(env, webgpu_ep);
  printf("{\"event\":\"devices\",\"cubecl_found\":%s,\"webgpu_found\":%s}\n", cubecl_dev ? "true" : "false", webgpu_dev ? "true" : "false");
  if (!cubecl_dev || !webgpu_dev) return 3;

  OrtMemoryInfo* mem = NULL;
  CHECK("CreateCpuMemoryInfo", api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));
  OrtValue* input_values[2] = {NULL, NULL};
  const char** input_names = NULL;
  size_t input_count = 0, output_elems = 0;
  if (prepare_inputs(op, dtype, a, b, c, mem, input_values, &input_names, &input_count, &output_elems)) return 2;
  const OrtValue* inputs[2] = {input_values[0], input_values[1]};
  const char* output_name = strcmp(op, "MatMul") == 0 ? "C" : "Z";

  char err[1024] = {0};
  OrtSession* cpu = create_session(env, model, NULL, NULL, err, sizeof(err));
  if (!cpu) { printf("{\"event\":\"error\",\"stage\":\"CreateSession(cpu)\",\"message\":"); json_escape_print(err); printf("}\n"); return 4; }
  OrtValue* ref = NULL;
  if (run_once(cpu, input_names, inputs, input_count, output_name, &ref)) return 4;

  Arm arms[2] = {{"cubecl", NULL, 0, {0}}, {"official_webgpu", NULL, 0, {0}}};
  const OrtEpDevice* arm_devs[2] = {cubecl_dev, webgpu_dev};
  for (int i = 0; i < 2; ++i) {
    char prefix[2048];
    snprintf(prefix, sizeof(prefix), "%s/%s_%s_%s", profile_dir, arms[i].name, op, dtype);
    OrtSession* prof = create_session(env, model, arm_devs[i], prefix, arms[i].error, sizeof(arms[i].error));
    if (!prof) {
      printf("{\"event\":\"arm_error\",\"arm\":"); json_escape_print(arms[i].name); printf(",\"message\":"); json_escape_print(arms[i].error); printf("}\n");
      continue;
    }
    OrtValue* got = NULL;
    if (run_once(prof, input_names, inputs, input_count, output_name, &got)) {
      api->ReleaseSession(prof);
      continue;
    }
    double max_abs = 0.0, max_rel = 0.0;
    int ok = compare_outputs(ref, got, output_elems, dtype, rtol, atol, &max_abs, &max_rel);
    printf("{\"event\":\"correctness\",\"arm\":");
    json_escape_print(arms[i].name);
    printf(",\"ok\":%s,\"max_abs\":%.17g,\"max_rel\":%.17g}\n", ok ? "true" : "false", max_abs, max_rel);
    end_profile(prof, arms[i].name);
    api->ReleaseValue(got);
    api->ReleaseSession(prof);
    arms[i].valid = ok;
  }

  for (int i = 0; i < 2; ++i) {
    if (!arms[i].valid) continue;
    arms[i].session = create_session(env, model, arm_devs[i], NULL, arms[i].error, sizeof(arms[i].error));
    if (!arms[i].session) {
      printf("{\"event\":\"arm_error\",\"arm\":"); json_escape_print(arms[i].name); printf(",\"message\":"); json_escape_print(arms[i].error); printf("}\n");
      arms[i].valid = 0;
    }
  }

  for (int w = 0; w < warmups; ++w) {
    for (int j = 0; j < 2; ++j) {
      int idx = (w + j) & 1;
      if (!arms[idx].valid) continue;
      OrtValue* out = NULL;
      if (run_once(arms[idx].session, input_names, inputs, input_count, output_name, &out) == 0) api->ReleaseValue(out);
    }
  }

  for (int s = 0; s < samples; ++s) {
    for (int j = 0; j < 2; ++j) {
      int idx = (s + j) & 1;
      if (!arms[idx].valid) continue;
      double t0 = now_s();
      for (int it = 0; it < iterations; ++it) {
        OrtValue* out = NULL;
        if (run_once(arms[idx].session, input_names, inputs, input_count, output_name, &out) == 0) api->ReleaseValue(out);
        else { arms[idx].valid = 0; break; }
      }
      double t1 = now_s();
      printf("{\"event\":\"sample\",\"arm\":");
      json_escape_print(arms[idx].name);
      printf(",\"sample\":%d,\"us_per_run\":%.17g}\n", s, (t1 - t0) * 1000000.0 / (double)iterations);
    }
  }
  fflush(stdout);
  return 0;
}
'''


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)


def find_ort_lib() -> Path:
    explicit = os.environ.get("NXRT_ORT_LIB_DIR")
    if explicit:
        p = Path(explicit) / ("libonnxruntime.dylib" if sys.platform == "darwin" else "libonnxruntime.so")
        if p.exists():
            return p
    candidates = sorted(
        ROOT.glob("target/release/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib/libonnxruntime.dylib")
    ) or sorted(
        ROOT.glob("target/debug/build/onnx-genai-ort-sys-*/out/ort-prebuilt/lib/libonnxruntime.dylib")
    )
    if not candidates:
        raise FileNotFoundError("未找到仓库 ORT dylib；请先确保 onnx-genai-ort-sys 已构建")
    return candidates[0]


def find_ort_root(ort_lib: Path) -> Path:
    return ort_lib.parents[1]


def default_official_plugin() -> Path:
    try:
        import onnxruntime_ep_webgpu as webgpu_ep  # type: ignore

        return Path(webgpu_ep.get_library_path())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "无法 import onnxruntime_ep_webgpu；请在 probe venv 中运行或传 --official-webgpu-plugin"
        ) from exc


def compile_helper(ort_root: Path) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    HELPER_C.write_text(HELPER_SOURCE)
    cmd = [
        "clang",
        "-O2",
        f"-I{ort_root / 'include'}",
        str(HELPER_C),
        "-o",
        str(HELPER_BIN),
    ]
    result = run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"helper 编译失败\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def make_model(path: Path, op: str, dtype: str, dims: tuple[int, int, int]) -> None:
    tensor_type = TensorProto.FLOAT16 if dtype == "f16" else TensorProto.FLOAT
    a, b, c = dims
    if op == "MatMul":
        inputs = [
            helper.make_tensor_value_info("A", tensor_type, [a, b]),
            helper.make_tensor_value_info("B", tensor_type, [b, c]),
        ]
        outputs = [helper.make_tensor_value_info("C", tensor_type, [a, c])]
        nodes = [helper.make_node("MatMul", ["A", "B"], ["C"])]
    elif op == "Relu":
        inputs = [helper.make_tensor_value_info("X", tensor_type, [a])]
        outputs = [helper.make_tensor_value_info("Z", tensor_type, [a])]
        nodes = [helper.make_node("Relu", ["X"], ["Z"])]
    else:
        inputs = [
            helper.make_tensor_value_info("X", tensor_type, [a]),
            helper.make_tensor_value_info("Y", tensor_type, [a]),
        ]
        outputs = [helper.make_tensor_value_info("Z", tensor_type, [a])]
        nodes = [helper.make_node(op, ["X", "Y"], ["Z"])]
    model = helper.make_model(
        helper.make_graph(nodes, f"{op}_{dtype}_{a}_{b}_{c}", inputs, outputs),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def profile_counts(profile_path: str | None) -> dict[str, int]:
    if not profile_path:
        return {}
    p = Path(profile_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    counts: dict[str, int] = {}
    for ev in data:
        args = ev.get("args") or {}
        provider = args.get("provider") or args.get("execution_provider")
        name = ev.get("name", "")
        if provider and (name.endswith("_kernel_time") or ev.get("cat") == "Node"):
            counts[provider] = counts.get(provider, 0) + 1
    return counts


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def cell_specs(include_f16: bool) -> list[dict]:
    dtypes = ["f32", "f16"] if include_f16 else ["f32"]
    specs = []
    for dtype in dtypes:
        for op in ["Add", "Mul", "Relu"]:
            for name, n in [("small", 1024), ("medium", 262144), ("large", 1048576)]:
                specs.append({"name": f"{op.lower()}/{name}/{dtype}", "op": op, "dtype": dtype, "dims": (n, 0, 0)})
        for name, dims in [
            ("gemv", (1, 512, 512)),
            ("small_gemm", (16, 256, 256)),
            ("medium_gemm", (32, 512, 512)),
        ]:
            specs.append({"name": f"matmul/{name}/{dtype}", "op": "MatMul", "dtype": dtype, "dims": dims})
    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ort-lib", type=Path, default=None)
    parser.add_argument("--cubecl-plugin", type=Path, default=ROOT / "target/release/libonnx_runtime_ep_cubecl_plugin.dylib")
    parser.add_argument("--official-webgpu-plugin", type=Path, default=None)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--include-f16", action="store_true", default=True)
    parser.add_argument("--no-f16", dest="include_f16", action="store_false")
    parser.add_argument("--filter", default="")
    args = parser.parse_args()

    if not args.cubecl_plugin.exists():
        raise FileNotFoundError(args.cubecl_plugin)
    official = args.official_webgpu_plugin or default_official_plugin()
    if not official.exists():
        raise FileNotFoundError(official)
    ort_lib = args.ort_lib or find_ort_lib()
    ort_root = find_ort_root(ort_lib)
    compile_helper(ort_root)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    print("# CubeCL WebGPU EP vs official ORT WebGPU EP")
    print()
    print("## Metadata")
    print()
    print(f"- machine: {platform.platform()} / {platform.machine()}")
    print(f"- python: {sys.version.split()[0]}")
    print(f"- ort dylib: `{ort_lib}`")
    print(f"- cubecl plugin: `{args.cubecl_plugin}`")
    print(f"- official webgpu plugin: `{official}`")
    print(f"- warmups: {args.warmups}; iterations/sample: {args.iterations}; samples/arm: {args.samples}")
    print("- f32 tolerance: rtol=1e-5, atol=1e-5")
    print("- f16 tolerance: rtol=5e-2, atol=5e-2（半精度累加/舍入差异门限；所有 cell 仍逐元素 allclose）")
    print()
    print("## Results")
    print()
    print("| cell | arm | status | provider node counts | median us | p90 us | n | correctness | max_abs | max_rel |")
    print("|---|---|---|---|---:|---:|---:|---|---:|---:|")

    rc = 0
    for spec in cell_specs(args.include_f16):
        if args.filter and args.filter not in spec["name"]:
            continue
        safe = spec["name"].replace("/", "_")
        model_path = MODELS_DIR / f"{safe}.onnx"
        make_model(model_path, spec["op"], spec["dtype"], spec["dims"])
        rtol, atol = (1e-5, 1e-5) if spec["dtype"] == "f32" else (5e-2, 5e-2)
        a, b, c = spec["dims"]
        cmd = [
            str(HELPER_BIN),
            str(ort_lib),
            str(args.cubecl_plugin),
            str(official),
            str(model_path),
            spec["op"],
            spec["dtype"],
            str(a),
            str(b),
            str(c),
            str(args.warmups),
            str(args.iterations),
            str(args.samples),
            str(PROFILES_DIR),
            str(rtol),
            str(atol),
            CUBECL_EP,
            WEBGPU_EP,
        ]
        proc = run(cmd)
        events = []
        for line in proc.stdout.splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"<!-- helper non-json stdout: {line} -->")
        if proc.stderr.strip():
            print(f"<!-- helper stderr for {spec['name']}: {proc.stderr.strip()} -->")

        correctness = {e["arm"]: e for e in events if e.get("event") == "correctness"}
        profiles = {e["arm"]: e.get("path") for e in events if e.get("event") == "profile"}
        samples: dict[str, list[float]] = {"cubecl": [], "official_webgpu": []}
        for e in events:
            if e.get("event") == "sample":
                samples[e["arm"]].append(float(e["us_per_run"]))
        arm_errors = {e.get("arm", "?"): e.get("message", "") for e in events if e.get("event") == "arm_error"}
        global_errors = [e for e in events if e.get("event") == "error"]

        for arm, expected_provider in [("cubecl", CUBECL_EP), ("official_webgpu", WEBGPU_EP)]:
            counts = profile_counts(profiles.get(arm))
            node_count = counts.get(expected_provider, 0)
            corr = correctness.get(arm)
            ok = bool(corr and corr.get("ok"))
            vals = samples[arm]
            if global_errors:
                status = "INVALID: " + "; ".join(f"{e.get('stage')}: {e.get('message')}" for e in global_errors[:2])
            elif arm in arm_errors:
                status = "INVALID: " + arm_errors[arm]
            elif node_count == 0:
                status = "INVALID: target EP node_count=0"
            elif not ok:
                status = "INVALID: correctness failed"
            elif not vals:
                status = "INVALID: no timing samples"
            else:
                status = "OK"
            if status != "OK":
                rc = 1
            med = statistics.median(vals) if vals else math.nan
            p90 = percentile(vals, 0.90) if vals else math.nan
            max_abs = corr.get("max_abs", math.nan) if corr else math.nan
            max_rel = corr.get("max_rel", math.nan) if corr else math.nan
            counts_s = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "-"
            corr_s = "PASS" if ok else "FAIL"
            print(
                f"| {spec['name']} | {arm} | {status} | `{counts_s}` | "
                f"{med:.3f} | {p90:.3f} | {len(vals)} | {corr_s} | {max_abs:.6g} | {max_rel:.6g} |"
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
