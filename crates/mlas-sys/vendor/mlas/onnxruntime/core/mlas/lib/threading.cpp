/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

#include "mlasi.h"

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
// nxrt-mlas-mt: pluggable parallel-for backend implemented in vendor/shim.cpp.
// When a backend is registered (via mlas_set_threading from Rust) these route
// MLAS's own tile partitioning onto a real thread pool (Rayon); otherwise they
// run serially. `work` points to the std::function<void(ptrdiff_t)> closure.
extern "C" void MlasStandaloneParallelFor(std::ptrdiff_t Iterations, void* work);
#endif

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //
    // Execute the routine directly if only one iteration is specified.
    //

    if (Iterations == 1) {
        ThreadedRoutine(Context, 0);
        return;
    }

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // nxrt-mlas-mt: route MLAS's own partitioned iterations onto the registered
    // parallel-for backend (Rayon), mirroring MlasTrySimpleParallel and the
    // upstream ORT MLAS_THREADPOOL::TrySimpleParallelFor path below. Without
    // this the standalone build ran every partition serially on the calling
    // thread, so the NCHWc convolution/pooling/reorder/transpose kernels — which
    // split into MlasGetMaximumThreadCount tiles — executed single-threaded (and
    // paid full partition overhead). MlasStandaloneParallelFor falls back to a
    // serial loop when no backend is registered, preserving the prior behaviour
    // for the mlas-sys unit tests that call the FFI directly.
    //
    // Each partitioned routine writes a disjoint output range keyed off `tid`
    // (this is required for the upstream concurrent TrySimpleParallelFor path),
    // so concurrent invocation is race-free.
    //
    std::function<void(std::ptrdiff_t)> work = [ThreadedRoutine, Context](std::ptrdiff_t tid) {
        ThreadedRoutine(Context, tid);
    };
    MlasStandaloneParallelFor(Iterations, const_cast<void*>(static_cast<const void*>(&work)));
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, [&](ptrdiff_t tid) {
        ThreadedRoutine(Context, tid);
    });
#endif
}


void
MlasTrySimpleParallel(
    MLAS_THREADPOOL * ThreadPool,
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work)
{
    //
    // Execute the routine directly if only one iteration is specified.
    //
    if (Iterations == 1) {
        Work(0);
        return;
    }

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // nxrt-mlas-mt: route MLAS's own partitioned iterations onto the registered
    // parallel-for backend (Rayon), falling back to a serial loop if none is
    // registered.
    //
    MlasStandaloneParallelFor(Iterations, const_cast<void*>(static_cast<const void*>(&Work)));
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, Work);
#endif
}


void
MlasTryBatchParallel(
	MLAS_THREADPOOL * ThreadPool,
	const std::ptrdiff_t Iterations,
	const std::function<void(std::ptrdiff_t tid)>& Work)
{
    //
    // Execute the routine directly if only one iteration is specified.
    //
    if (Iterations == 1) {
        Work(0);
        return;
    }

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // Fallback to OpenMP or a serialized implementation.
    //

    //
    // Execute the routine for the specified number of iterations.
    //
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        Work(tid);
    }
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TryBatchParallelFor(ThreadPool, Iterations, Work, 0);
#endif

}