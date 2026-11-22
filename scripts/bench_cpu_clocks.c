// Developer-only POSIX clock diagnostic, not an inference benchmark.
// The coordinator actively waits for results; worker count excludes it.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define MAX_WORKERS 64
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t quiet = PTHREAD_COND_INITIALIZER;
static _Atomic uint64_t generation;
static uint64_t iterations;
static unsigned width, started, repetitions = 21;
static _Atomic unsigned pending;
static int stopping, spin_mode, observe_mode;
struct worker {
  unsigned id;
  uint64_t quiet_generation, completed_generation, hash, cpu_ns;
  pthread_t thread;
};
static struct worker workers[MAX_WORKERS];

static void thread_check(int error, const char *operation) {
  if (error) {
    fprintf(stderr, "CPU-clock probe %s: %s\n", operation, strerror(error));
    exit(2);
  }
}
static uint64_t clock_ns(clockid_t id) {
  struct timespec t;
  if (clock_gettime(id, &t)) {
    perror(
        "CPU-clock probe clock_gettime: verify requested clock is supported");
    exit(2);
  }
  return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static uint64_t usage_ns(void) {
  struct rusage r;
  if (getrusage(RUSAGE_SELF, &r)) {
    perror("CPU-clock probe getrusage(RUSAGE_SELF)");
    exit(2);
  }
  return (uint64_t)(r.ru_utime.tv_sec + r.ru_stime.tv_sec) *
             UINT64_C(1000000000) +
         (uint64_t)(r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1000;
}
static uint64_t work(uint64_t count, unsigned id) {
  uint64_t x = UINT64_C(0x9e3779b97f4a7c15) ^ id;
  for (uint64_t i = 0; i < count; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
  }
  return x;
}
static void *worker_main(void *arg) {
  struct worker *w = arg;
  uint64_t seen = 0;
  int registered = 0;
  for (;;) {
    thread_check(pthread_mutex_lock(&lock), "lock worker coordination");
    if (!registered) {
      ++started;
      registered = 1;
    }
    // Announce quiescence while holding the mutex, then atomically release
    // it in cond_wait. The caller cannot pass this barrier before parking.
    w->quiet_generation = seen;
    thread_check(pthread_cond_broadcast(&quiet), "announce worker quiescence");
    while (!stopping &&
           atomic_load_explicit(&generation, memory_order_acquire) == seen)
      thread_check(pthread_cond_wait(&ready, &lock),
                   "wait for work generation");
    if (stopping) {
      thread_check(pthread_mutex_unlock(&lock), "unlock worker coordination");
      return NULL;
    }
    seen = atomic_load_explicit(&generation, memory_order_relaxed);
    uint64_t count = iterations;
    int spin = spin_mode, observe = observe_mode;
    thread_check(pthread_mutex_unlock(&lock), "unlock worker coordination");
    uint64_t start = observe ? clock_ns(CLOCK_THREAD_CPUTIME_ID) : 0;
    uint64_t hash = work(count, w->id);
    uint64_t cpu = observe ? clock_ns(CLOCK_THREAD_CPUTIME_ID) - start : 0;
    w->cpu_ns = cpu;
    w->hash = hash;
    w->completed_generation = seen;
    // Completion is an atomic RMW, without an explicit worker CPU-clock
    // query or condition-variable notification in unobserved mode.
    atomic_fetch_sub_explicit(&pending, 1, memory_order_acq_rel);
    // Simulate a pool's active tail after the caller observes completion.
    // A new generation can interrupt the spin to start useful work again.
    if (spin) {
      uint64_t end = clock_ns(CLOCK_MONOTONIC) + UINT64_C(1000000);
      while (clock_ns(CLOCK_MONOTONIC) < end &&
             atomic_load_explicit(&generation, memory_order_acquire) == seen)
        atomic_signal_fence(memory_order_seq_cst);
    }
  }
}
static void await_quiet(void) {
  thread_check(pthread_mutex_lock(&lock), "lock worker coordination");
  for (;;) {
    int all = started == width;
    uint64_t g = atomic_load_explicit(&generation, memory_order_relaxed);
    for (unsigned i = 0; i < width; ++i)
      all &= workers[i].quiet_generation >= g;
    if (all)
      break;
    thread_check(pthread_cond_wait(&quiet, &lock),
                 "wait for worker quiescence");
  }
  thread_check(pthread_mutex_unlock(&lock), "unlock worker coordination");
}
static void sample(uint64_t count, unsigned batch, int spin, int observe,
                   unsigned rep, unsigned arm) {
  await_quiet();
  uint64_t thread_sum = 0, checksum = 0;
  // Nested windows: process clock encloses getrusage, which encloses wall.
  // Includes synchronous job submission/completion; excludes prior/post spin
  // outside the sample. Worker sum measures useful work and its thread-clock
  // bookkeeping; process measurements also include coordination/active tails.
  uint64_t p0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID), r0 = usage_ns();
  uint64_t main0 = observe ? clock_ns(CLOCK_THREAD_CPUTIME_ID) : 0;
  uint64_t t0 = clock_ns(CLOCK_MONOTONIC);
  for (unsigned k = 0; k < batch; ++k) {
    thread_check(pthread_mutex_lock(&lock), "lock worker coordination");
    iterations = count;
    spin_mode = spin;
    observe_mode = observe;
    atomic_store_explicit(&pending, width, memory_order_release);
    atomic_fetch_add_explicit(&generation, 1, memory_order_release);
    thread_check(pthread_cond_broadcast(&ready), "announce work generation");
    thread_check(pthread_mutex_unlock(&lock), "unlock worker coordination");
    while (atomic_load_explicit(&pending, memory_order_acquire))
      atomic_signal_fence(memory_order_seq_cst);
    for (unsigned i = 0; i < width; ++i) {
      if (workers[i].completed_generation !=
          atomic_load_explicit(&generation, memory_order_relaxed)) {
        fprintf(stderr,
                "CPU-clock probe worker %u returned a stale generation; refuse "
                "incomplete work\n",
                i);
        exit(2);
      }
      thread_sum += workers[i].cpu_ns;
      checksum += workers[i].hash;
    }
  }
  uint64_t wall = clock_ns(CLOCK_MONOTONIC) - t0;
  uint64_t main_cpu = observe ? clock_ns(CLOCK_THREAD_CPUTIME_ID) - main0 : 0;
  uint64_t ru = usage_ns() - r0;
  uint64_t process = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - p0;
  // Latency stops at result readiness. Resource accounting continues until
  // every worker parks, so active pool tails cannot leak into another arm.
  await_quiet();
  uint64_t ru_drained = usage_ns() - r0;
  uint64_t process_drained = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - p0;
  uint64_t wall_drained = clock_ns(CLOCK_MONOTONIC) - t0;
  if (!checksum || (observe && (!thread_sum || !main_cpu))) {
    fputs("CPU-clock probe: work checksum/CPU sum is empty; refuse vacuous "
          "sample\n",
          stderr);
    exit(2);
  }
  printf("sample,%u,%d,%d,%" PRIu64 ",%u,%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64
         ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
         "\n",
         width, spin, observe, count, batch, rep, arm, wall, ru, process,
         thread_sum, main_cpu, ru_drained, process_drained, wall_drained,
         checksum);
}
int main(int argc, char **argv) {
  char *end = NULL;
  unsigned long requested =
      (argc == 2 || argc == 3) ? strtoul(argv[1], &end, 10) : 0;
  if ((argc != 2 && argc != 3) || !end || *end || requested < 1 ||
      requested > MAX_WORKERS) {
    fprintf(stderr,
            "usage: %s WORKERS (1..%d) [REPETITIONS (1..101)]; run under the "
            "shared host lock\n",
            argv[0], MAX_WORKERS);
    return 2;
  }
  width = (unsigned)requested;
  if (argc == 3) {
    unsigned long reps = strtoul(argv[2], &end, 10);
    if (!end || *end || reps < 1 || reps > 101) {
      fprintf(stderr, "CPU-clock probe repetitions '%s' rejected: use 1..101\n",
              argv[2]);
      return 2;
    }
    repetitions = (unsigned)reps;
  }
  for (unsigned i = 0; i < width; ++i) {
    workers[i].id = i;
    int error =
        pthread_create(&workers[i].thread, NULL, worker_main, &workers[i]);
    if (error) {
      fprintf(stderr, "CPU-clock probe pthread_create worker %u: %s\n", i,
              strerror(error));
      return 2;
    }
  }
  struct timespec resolution;
  if (clock_getres(CLOCK_PROCESS_CPUTIME_ID, &resolution)) {
    perror("CPU-clock probe process-clock resolution");
    return 2;
  }
  fprintf(stderr, "process_clock_resolution_ns=%" PRIu64 "\n",
          (uint64_t)resolution.tv_sec * UINT64_C(1000000000) +
              (uint64_t)resolution.tv_nsec);
  if (clock_getres(CLOCK_THREAD_CPUTIME_ID, &resolution)) {
    perror("CPU-clock probe thread-clock resolution");
    return 2;
  }
  fprintf(stderr, "thread_clock_resolution_ns=%" PRIu64 "\n",
          (uint64_t)resolution.tv_sec * UINT64_C(1000000000) +
              (uint64_t)resolution.tv_nsec);
  fprintf(stderr, "completion_atomic_lock_free=%d\n",
          atomic_is_lock_free(&pending));
  puts("kind,width,spin,observe,iterations,batch,rep,arm,wall_ns,getrusage_ns,"
       "process_ns,thread_sum_ns,main_thread_ns,getrusage_drained_ns,process_"
       "drained_ns,wall_drained_ns,checksum");
  uint64_t counts[] = {16384, 131072, 1048576};
  unsigned batches[] = {1, 16};
  for (int spin = 0; spin <= 1; ++spin)
    for (int observe = 0; observe <= 1; ++observe)
      for (unsigned c = 0; c < 3; ++c)
        for (unsigned b = 0; b < 2; ++b)
          for (unsigned rep = 0; rep < repetitions; ++rep)
            for (unsigned order = 0; order < 2; ++order)
              sample(counts[c], batches[b], spin, observe, rep,
                     (rep + order) % 2);
  await_quiet();
  uint64_t p0 = clock_ns(CLOCK_PROCESS_CPUTIME_ID), r0 = usage_ns(),
           t0 = clock_ns(CLOCK_MONOTONIC);
  struct timespec sleep_time = {0, 20000000};
  nanosleep(&sleep_time, NULL);
  uint64_t wall = clock_ns(CLOCK_MONOTONIC) - t0;
  uint64_t ru = usage_ns() - r0,
           process = clock_ns(CLOCK_PROCESS_CPUTIME_ID) - p0;
  printf("idle,%u,0,0,0,0,0,0,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",0,0,%" PRIu64
         ",%" PRIu64 ",%" PRIu64 ",0\n",
         width, wall, ru, process, ru, process, wall);
  thread_check(pthread_mutex_lock(&lock), "lock worker coordination");
  stopping = 1;
  thread_check(pthread_cond_broadcast(&ready), "announce work generation");
  thread_check(pthread_mutex_unlock(&lock), "unlock worker coordination");
  for (unsigned i = 0; i < width; ++i)
    thread_check(pthread_join(workers[i].thread, NULL), "join worker");
  return 0;
}
