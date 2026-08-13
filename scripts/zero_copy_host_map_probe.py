"""Feasibility probe for the #864 hybrid: can a CUDA kernel read weights in place
from a host mmap, and at what bandwidth?

The hybrid proposed on #864 is: keep the hot weight set resident in VRAM, and for
the cold remainder do *not* copy — let the device read it directly from host
memory, because each weight is read exactly once per decode step and a copy
therefore has no reuse to amortize.

That rests on one capability the tree does not currently use anywhere:
``cuMemHostRegister`` on an existing read-only file mapping, plus
``cuMemHostGetDevicePointer``, giving a device-addressable pointer into host RAM.

Three questions, answered cheaply and without touching the engine:

  1. Does ``cuMemHostRegister`` succeed on a large read-only file mapping?
  2. What bandwidth does a device-side read of that memory achieve, versus an
     explicit H2D copy of the same bytes (what the managed path does today),
     versus a read of bytes already resident in VRAM (the ceiling)?
  3. How long does registration itself take? It page-locks host memory, so if it
     were slow it would have to be amortized over the process lifetime rather
     than paid per page-in.

Measured on an RTX 4060 Laptop (2026-08-13), 1 GiB slice of qwen14b-zp:

    device reading mapped host memory (zero-copy)   11.31 GB/s
    explicit HtoD copy (what we do today)           11.36 GB/s
    already-resident VRAM read (ceiling)           111.58 GB/s
    cuMemHostRegister of 1 GiB                       2.8 ms

Interpretation: zero-copy and copy cost the *same per PCIe pass* (1.00x), so the
saving from zero-copy is not bandwidth — it is the copy's second pass plus all
the machinery (pinned staging fill, VRAM alloc, cuMemMap, eviction, synchronize).
The resident hot set is worth ~9.9x, which is the larger prize and should be
maximized first.

Caveat: a DtoD copy *from* mapped host memory measures the rate at which the
device pulls those bytes across PCIe, which is the right proxy for a kernel
reading them in place, but it is not literally a kernel doing strided GEMV
reads. Treat the zero-copy figure as an upper bound.

Usage:  python scripts/zero_copy_host_map_probe.py [path-to-weight-blob]
"""

import ctypes
import mmap
import os
import sys
import time

DEFAULT_BLOB = r"C:\Users\justinchu\dev\models\qwen14b-zp\model.onnx.data"
# Probe a slice rather than the whole blob: registration page-locks host RAM, and
# the question is bandwidth per byte, which a slice answers just as well.
SLICE_BYTES = 1 << 30  # 1 GiB

CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
CU_MEMHOSTREGISTER_READ_ONLY = 0x08
CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19


def load_driver():
    for name in ("nvcuda.dll", "libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    sys.exit("no CUDA driver library found")


CU = load_driver()


def check(code, what):
    if code != 0:
        err = ctypes.c_char_p()
        CU.cuGetErrorString(code, ctypes.byref(err))
        text = err.value.decode() if err.value else "?"
        raise RuntimeError(f"{what} failed: {code} ({text})")


def main() -> int:
    blob = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BLOB
    if not os.path.exists(blob):
        sys.exit(f"weight blob not found: {blob}")

    check(CU.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(CU.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(CU.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")

    name_buf = ctypes.create_string_buffer(128)
    CU.cuDeviceGetName(name_buf, 128, dev)
    print(f"device: {name_buf.value.decode()}")
    for attr, label in (
        (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, "can_map_host_memory"),
        (CU_DEVICE_ATTRIBUTE_INTEGRATED, "integrated"),
    ):
        value = ctypes.c_int()
        check(CU.cuDeviceGetAttribute(ctypes.byref(value), attr, dev), label)
        print(f"{label}: {value.value}")

    size = os.path.getsize(blob)
    print(f"\nsource: {blob}")
    print(f"file size: {size:,} bytes; probing a {SLICE_BYTES:,}-byte slice")
    if size < SLICE_BYTES:
        sys.exit("blob smaller than the probe slice")

    handle = open(blob, "rb")
    mapping = mmap.mmap(handle.fileno(), SLICE_BYTES, access=mmap.ACCESS_READ)
    # numpy exposes the base address of a read-only buffer without copying it;
    # ctypes' from_buffer refuses non-writable buffers.
    import numpy as np

    view = np.frombuffer(mapping, dtype=np.uint8, count=SLICE_BYTES)
    host_ptr = ctypes.c_void_p(view.ctypes.data)
    print(f"host mapping at 0x{host_ptr.value:x}")

    start = time.perf_counter()
    rc = CU.cuMemHostRegister_v2(
        host_ptr,
        ctypes.c_size_t(SLICE_BYTES),
        ctypes.c_uint(CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_READ_ONLY),
    )
    register_s = time.perf_counter() - start
    if rc != 0:
        err = ctypes.c_char_p()
        CU.cuGetErrorString(rc, ctypes.byref(err))
        text = err.value.decode() if err.value else "?"
        print(f"\ncuMemHostRegister FAILED: {rc} ({text})")
        print("=> the zero-copy half of the #864 hybrid is not reachable this way")
        return 0
    print(
        f"\ncuMemHostRegister: OK in {register_s * 1000:.1f} ms "
        f"({SLICE_BYTES / register_s / 1e9:.2f} GB/s of page-locking)"
    )

    mapped_dptr = ctypes.c_void_p()
    check(
        CU.cuMemHostGetDevicePointer_v2(ctypes.byref(mapped_dptr), host_ptr, ctypes.c_uint(0)),
        "cuMemHostGetDevicePointer",
    )
    print(f"device-visible pointer: 0x{mapped_dptr.value:x}")

    vram = ctypes.c_void_p()
    check(CU.cuMemAlloc_v2(ctypes.byref(vram), ctypes.c_size_t(SLICE_BYTES)), "cuMemAlloc")
    vram2 = ctypes.c_void_p()
    check(CU.cuMemAlloc_v2(ctypes.byref(vram2), ctypes.c_size_t(SLICE_BYTES)), "cuMemAlloc2")
    stream = ctypes.c_void_p()
    check(CU.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
    ev_a, ev_b = ctypes.c_void_p(), ctypes.c_void_p()
    check(CU.cuEventCreate(ctypes.byref(ev_a), 0), "cuEventCreate(a)")
    check(CU.cuEventCreate(ctypes.byref(ev_b), 0), "cuEventCreate(b)")

    def timed(fn, label, reps=3):
        best = None
        for _ in range(reps):
            check(CU.cuEventRecord(ev_a, stream), "record(a)")
            fn()
            check(CU.cuEventRecord(ev_b, stream), "record(b)")
            check(CU.cuStreamSynchronize(stream), "sync")
            ms = ctypes.c_float()
            check(CU.cuEventElapsedTime(ctypes.byref(ms), ev_a, ev_b), "elapsed")
            gbps = SLICE_BYTES / (ms.value / 1000.0) / 1e9
            best = gbps if best is None else max(best, gbps)
        print(f"{label:<44} {best:6.2f} GB/s")
        return best

    zero_copy = timed(
        lambda: check(
            CU.cuMemcpyDtoDAsync_v2(vram, mapped_dptr, ctypes.c_size_t(SLICE_BYTES), stream),
            "DtoD from mapped host",
        ),
        "device reading mapped host memory (zero-copy)",
    )
    htod = timed(
        lambda: check(
            CU.cuMemcpyHtoDAsync_v2(vram, host_ptr, ctypes.c_size_t(SLICE_BYTES), stream),
            "HtoD",
        ),
        "explicit HtoD copy (what we do today)",
    )
    resident = timed(
        lambda: check(
            CU.cuMemcpyDtoDAsync_v2(vram2, vram, ctypes.c_size_t(SLICE_BYTES), stream),
            "DtoD in VRAM",
        ),
        "already-resident VRAM read (ceiling)",
    )

    print("\n--- interpretation ---")
    print(f"zero-copy / explicit-copy ratio : {zero_copy / htod:.2f}x")
    print(f"resident / zero-copy ratio      : {resident / zero_copy:.1f}x")
    print(
        "\nA weight read ONCE per step costs ~1 PCIe pass either way, so zero-copy is\n"
        "not a bandwidth win — it removes the copy's second pass and its machinery\n"
        "(staging fill, VRAM alloc, cuMemMap, eviction, synchronize). The resident\n"
        "ratio is what the hot set buys, and it is the larger prize."
    )

    CU.cuMemFree_v2(vram)
    CU.cuMemFree_v2(vram2)
    CU.cuMemHostUnregister(host_ptr)
    # Release the numpy view before closing the mapping, or mmap.close() raises
    # "cannot close exported pointers exist".
    del view
    mapping.close()
    handle.close()
    CU.cuCtxDestroy_v2(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
