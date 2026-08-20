# CubeCL WebGPU EP vs 官方 ORT WebGPU EP

## 结论

在这台 Mac、这些 tiny one-op ONNX 形状上，CubeCL WebGPU plugin EP 已能和
Microsoft 官方 `onnxruntime-ep-webgpu` plugin EP 在**同一个 ORT 1.28.0 进程**
中运行，并且所有 cell 都由 ORT profile 确认节点落在目标 EP。端到端每次
`Run` 上，CubeCL 比官方 WebGPU 慢约 1.3–2.1x(两轮独立测量的合并区间)；但这张表主要测到的是 ORT
`Run`/绑定/host-device 边界/同步等固定路径，不是 shader/kernel 质量。精度上，
CubeCL f16 MatMul 与 ORT CPU 参考逐元素相等，而官方 WebGPU f16 MatMul 有
最高约 1.19 的绝对误差。

## 机器与运行元数据

| field | value |
|---|---|
| machine | macOS-26.5.2-arm64-arm-64bit-Mach-O / arm64 |
| ORT dylib | `target/release/build/onnx-genai-ort-sys-3f5a1c371eb81275/out/ort-prebuilt/lib/libonnxruntime.dylib` |
| ORT runtime version | 1.28.0 |
| CubeCL plugin | `target/release/libonnx_runtime_ep_cubecl_plugin.dylib` |
| CubeCL plugin crate version/profile | `onnx-runtime-ep-cubecl-plugin` 0.1.0-dev.5, `release`, `--features webgpu` |
| CubeCL registration / EP name used | registration `onnxruntime_cubecl_webgpu_ep`, EP name `onnxruntime_cubecl_webgpu_ep` |
| official plugin package | `onnxruntime-ep-webgpu==0.2.1` |
| official plugin dylib | `$HOME/.copilot/ortwebgpu-probe/venv/lib/python3.13/site-packages/onnxruntime_ep_webgpu/libonnxruntime_providers_webgpu.dylib` |
| official registration / EP name used | registration `webgpu`, EP name `WebGpuExecutionProvider` |
| Python used to drive harness generation | 3.13.5 |
| harness | `scripts/bench_cubecl_vs_ort_webgpu.py` |
| warmups | 50 interleaved warmup runs per arm/cell |
| samples | 7 samples per arm/cell |
| iterations/sample | 20 `Run` calls |
| f32 tolerance | `rtol=1e-5`, `atol=1e-5` |
| f16 tolerance | `rtol=5e-2`, `atol=5e-2` |

The harness builds tiny ONNX models, loads one ORT C API instance via `dlopen`,
registers both plugin EPs into the same `OrtEnv`, creates a CPU EP reference
session for correctness, then measures the two GPU plugin sessions in an
interleaved order. Each cell also creates a profiled session and marks the cell
invalid if the target provider's node count is zero.

## 重要作废记录：第一轮 MatMul 数字无效

第一轮 full-table run produced a self-contradictory f32 MatMul result and was
explicitly discarded before this report was written:

- `matmul/medium_gemm/f32` CubeCL median was 21111 us while the corresponding
  f16 median was 1036 us. A dtype change alone cannot justify a ~20x difference.
- f32 MatMul shapes had a nonsensical work/time relationship: `gemv`,
  `small_gemm`, and `medium_gemm` differed greatly in work, but reported medians
  clustered around a large fixed cost.
- `matmul/medium_gemm/f32` at 21111 us implies only about 0.056 GB/s for the
  simple read/write byte count, which is not a plausible memory or compute
  interpretation on this machine.
- f32 MatMul p90/median was too wide, consistent with external contention or a
  periodic one-off cost entering measured samples.

Before the rerun, the machine was not fully quiet: load average was still high,
WindowServer/WebKit were active, and a `ReportCrash` process was observed during
the investigation. No `cargo`/`rustc` build process was present at the final
rerun check, but the invalid first table is not used below.

## What this table can and cannot claim

This is an end-to-end `OrtSession::Run` comparison for tiny one-op models. It
includes ORT run overhead, session input/output binding behavior, host/device
boundary handling, GPU submission, synchronization, and plugin-specific fixed
paths. It is **not** a pure kernel or shader benchmark.

The evidence is direct:

- The final timings are relatively flat across very different MatMul workloads.
- Add/Mul/Relu ratios also remain in the same broad band across small/medium/large
  element counts.
- A warmed `matmul/medium_gemm/f32` diagnostic measured CubeCL wall time around
  962 us/run while the profiled node event at the end of the warmed session was
  only about 8–9 us.

That means almost all of the measured wall time is outside the profiled kernel
event. The actionable optimization target from this table is the per-`Run` fixed
path and boundary/synchronization behavior, not a conclusion that CubeCL's
MatMul shader itself is 1.4–1.7x worse.

Do **not** compare profile node durations across the two EPs as a kernel-quality
claim. In the warmed diagnostic, CubeCL's final profiled node events were around
8–9 us and official WebGPU's around 14–16 us, but I did not verify that the two
EPs bracket the same start/stop region in ORT profile JSON. Without that
instrumentation contract, cross-EP node-duration comparisons are not reliable.

## Results

All rows below passed CPU-reference correctness and had target-provider node
count 1 in ORT profile.

| cell | arm | provider node counts | median us | p90 us | n | correctness | max_abs | max_rel |
|---|---|---|---:|---:|---:|---|---:|---:|
| add/small/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 479.750 | 491.700 | 7 | PASS | 0 | 0 |
| add/small/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 319.650 | 338.750 | 7 | PASS | 0 | 0 |
| add/medium/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1062.000 | 1142.250 | 7 | PASS | 0 | 0 |
| add/medium/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 709.400 | 792.250 | 7 | PASS | 0 | 0 |
| add/large/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 2599.200 | 2662.550 | 7 | PASS | 0 | 0 |
| add/large/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 1725.100 | 1754.850 | 7 | PASS | 0 | 0 |
| mul/small/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 552.850 | 602.750 | 7 | PASS | 0 | 0 |
| mul/small/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 333.200 | 354.450 | 7 | PASS | 0 | 0 |
| mul/medium/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1114.700 | 1215.800 | 7 | PASS | 0 | 0 |
| mul/medium/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 705.300 | 752.500 | 7 | PASS | 0 | 0 |
| mul/large/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 2606.900 | 2686.100 | 7 | PASS | 0 | 0 |
| mul/large/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 1736.100 | 1757.400 | 7 | PASS | 0 | 0 |
| relu/small/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 445.050 | 510.200 | 7 | PASS | 0 | 0 |
| relu/small/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 271.000 | 277.800 | 7 | PASS | 0 | 0 |
| relu/medium/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 828.100 | 874.600 | 7 | PASS | 0 | 0 |
| relu/medium/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 596.050 | 638.000 | 7 | PASS | 0 | 0 |
| relu/large/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1818.950 | 1854.950 | 7 | PASS | 0 | 0 |
| relu/large/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 1323.700 | 1347.550 | 7 | PASS | 0 | 0 |
| matmul/gemv/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1042.850 | 1086.900 | 7 | PASS | 0 | 0 |
| matmul/gemv/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 613.100 | 633.400 | 7 | PASS | 0 | 0 |
| matmul/small_gemm/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 710.700 | 743.100 | 7 | PASS | 0 | 0 |
| matmul/small_gemm/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 514.650 | 537.400 | 7 | PASS | 0 | 0 |
| matmul/medium_gemm/f32 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 914.850 | 1086.300 | 7 | PASS | 0 | 0 |
| matmul/medium_gemm/f32 | official_webgpu | `WebGpuExecutionProvider:1` | 666.500 | 743.150 | 7 | PASS | 0 | 0 |
| add/small/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 591.100 | 664.850 | 7 | PASS | 0 | 0 |
| add/small/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 349.500 | 362.200 | 7 | PASS | 0 | 0 |
| add/medium/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 869.200 | 876.750 | 7 | PASS | 0 | 0 |
| add/medium/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 533.400 | 558.000 | 7 | PASS | 0 | 0 |
| add/large/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1754.550 | 1771.450 | 7 | PASS | 0 | 0 |
| add/large/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 1053.250 | 1087.100 | 7 | PASS | 0 | 0 |
| mul/small/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 573.350 | 630.050 | 7 | PASS | 0 | 0 |
| mul/small/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 348.800 | 352.500 | 7 | PASS | 0 | 0 |
| mul/medium/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 864.950 | 920.650 | 7 | PASS | 0 | 0 |
| mul/medium/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 549.950 | 569.000 | 7 | PASS | 0 | 0 |
| mul/large/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1673.100 | 1724.700 | 7 | PASS | 0 | 0 |
| mul/large/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 999.450 | 1033.450 | 7 | PASS | 0 | 0 |
| relu/small/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 485.500 | 518.950 | 7 | PASS | 0 | 0 |
| relu/small/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 270.650 | 281.050 | 7 | PASS | 0 | 0 |
| relu/medium/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 588.150 | 631.550 | 7 | PASS | 0 | 0 |
| relu/medium/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 448.300 | 477.450 | 7 | PASS | 0 | 0 |
| relu/large/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1056.350 | 1077.100 | 7 | PASS | 0 | 0 |
| relu/large/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 815.650 | 821.100 | 7 | PASS | 0 | 0 |
| matmul/gemv/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 861.850 | 996.800 | 7 | PASS | 0 | 0 |
| matmul/gemv/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 520.000 | 531.000 | 7 | PASS | 1.0625 | 0.0121669 |
| matmul/small_gemm/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 754.550 | 891.200 | 7 | PASS | 0 | 0 |
| matmul/small_gemm/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 499.100 | 562.600 | 7 | PASS | 0.4375 | 0.00696594 |
| matmul/medium_gemm/f16 | cubecl | `onnxruntime_cubecl_webgpu_ep:1` | 1066.400 | 1132.050 | 7 | PASS | 0 | 0 |
| matmul/medium_gemm/f16 | official_webgpu | `WebGpuExecutionProvider:1` | 634.750 | 661.850 | 7 | PASS | 1.1875 | 0.0138045 |

## 精度结论

CubeCL 在所有 f16 cell 上对 ORT CPU 参考均为 `max_abs = 0`、`max_rel = 0`。
这包括 f16 MatMul。机制是 CubeCL MatMul 使用 f32 累加，而 ORT CPU EP 对
f16 MatMul 也升到 f32 计算，因此 tiny synthetic 输入下逐元素相等。

官方 WebGPU 的 f16 Add/Mul/Relu 也逐元素相等，但 f16 MatMul 与 ORT CPU
参考不同：`gemv` 的 `max_abs = 1.0625`，`small_gemm` 的 `max_abs = 0.4375`，
`medium_gemm` 的 `max_abs = 1.1875`。这与官方 WebGPU 使用原生 f16 累加
一致。CubeCL 的 f32 累加是一个真实精度优势；同时也是 tradeoff，因为它放弃
了 f16 累加可能带来的速度。

## 独立复现(不同会话、不同负载)

上表由一个 agent 会话产出。之后在同一台机器、**不同时刻、loadavg 12.96**
的条件下,由另一个会话独立重跑了 matmul 组(`--filter matmul --warmups 50
--iterations 20 --samples 5`):

| cell | CubeCL median us | 官方 median us | 比值 |
|---|---:|---:|---:|
| matmul/gemv/f32 | 875.150 | 544.550 | 1.61x |
| matmul/small_gemm/f32 | 523.000 | 392.200 | 1.33x |
| matmul/medium_gemm/f32 | 876.850 | 532.850 | 1.65x |
| matmul/gemv/f16 | 756.950 | 366.100 | 2.07x |
| matmul/small_gemm/f16 | 466.200 | 365.950 | 1.27x |
| matmul/medium_gemm/f16 | 779.350 | 448.450 | 1.74x |

两点结论:

1. **绝对时间不可跨会话比较**(这一轮普遍比上表快 15–30%,尽管 loadavg 更高),
   但**比值是稳健的**。因此本报告的可复用结论是比值,不是绝对微秒数。
2. 比值的实测区间是 **1.27x–2.07x**,比正文摘要里的 1.4–1.7x 更宽。引用时请用
   宽区间。单次测量给出的窄区间不要当成精度。

精度数字则是**完全可复现**的:两轮的 `max_abs` 逐位相同
(`gemv 1.0625` / `small_gemm 0.4375` / `medium_gemm 1.1875`),因为它由累加器
类型决定,与机器负载无关。

## 复现

```bash
# 1. 构建 CubeCL plugin(release,否则测的是 debug 代码)
cargo build --release -p onnx-runtime-ep-cubecl-plugin --features webgpu

# 2. 准备一个装了官方 plugin 的 venv
python -m venv .venv-ortwebgpu
.venv-ortwebgpu/bin/pip install "onnxruntime>=1.24.4" onnxruntime-ep-webgpu

# 3. 跑 harness。默认会自己找仓库内的 ORT dylib;
#    也可以用 NXRT_ORT_LIB_DIR 指定。
.venv-ortwebgpu/bin/python scripts/bench_cubecl_vs_ort_webgpu.py \
  --warmups 50 --iterations 20 --samples 7
```

harness 会把一份运行产物写进 `target/`,那份是一次性的;本文件才是记录。

**重跑前先让机器安静下来。** 第一轮就是因为没做到这一点而全表作废(见上文)。
至少确认没有 `cargo`/`rustc` 在跑、loadavg 已回落,并且在**每次**运行前确认,
而不是只在开始时确认一次。

## 附录:graph capture 与本表的口径

本表**两个 arm 都没有开 graph capture**,因此比较是同口径的。但这一节要记下
两件在后续工作中会反复踩到的事。

### 1. 带前缀的 provider option key 会被静默忽略

官方 dylib 里能 `strings` 出 `ep.webgpuexecutionprovider.enableGraphCapture`,
但那是**旧式 EP 的 session-config 命名**。走 plugin EP API
(`SessionOptionsAppendExecutionProvider_V2`)时,key **不带**前缀,就是
`enableGraphCapture`。

带前缀的 key 传进去不会报错、不会警告,只是不生效。我第一次就是这样测的,
拿到一个"开了 graph capture 也没变化"的数字——那个数字是无效的,因为
**knob 从未生效**。

判别方法(值得对任何 provider option 都做一遍):**传一个非法值,要求它报错。**

```
enableGraphCapture=bogus
  -> Invalid enable graph capture: bogus      # key 被消费了,可信
ep.webgpuexecutionprovider.enableGraphCapture=bogus
  -> 静默通过                                  # key 没被消费,之前的测量作废
```

一个不会因错误输入而失败的开关,也不会因正确输入而生效。

### 2. graph capture 打开后,本 harness 的结果是错的

在 `matmul/medium_gemm/f32` 上打开 `enableGraphCapture=1`,官方 arm 对 CPU
参考的 `max_abs` 是 **115.281**(FAIL)。

这不是官方 EP 的 bug,是 harness 用法不满足 graph capture 的前提:本 harness
每次 Run 传的是 **CPU 内存里的 `OrtValue`**,而 capture 会把 buffer 地址录进
命令流后 replay,后续迭代写入的新输入根本没进到被 replay 的那份 GPU buffer。

因此"给官方开 graph capture"不是加一个参数就完事,需要 harness 改成
`OrtIoBinding` + 常驻 GPU 的输入输出。在 ResNet 那类全静态形状的多节点图上
这件事必须做对,否则官方 arm 要么结果错、要么白白背着它本可以省掉的固定开销,
两种都不是可比的口径。
