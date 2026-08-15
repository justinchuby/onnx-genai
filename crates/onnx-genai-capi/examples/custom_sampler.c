/*
 * custom_sampler.c — plug a C-side sampler into onnx-genai.
 *
 * Build (after `cargo build -p onnx-genai-capi --release`):
 *
 *   Linux/macOS:
 *     cc custom_sampler.c -I../include -L../../../target/release \
 *        -lonnx_genai_capi -o custom_sampler
 *   Windows (MSVC):
 *     cl custom_sampler.c /I..\include ^
 *        ..\..\..\target\release\onnx_genai_capi.dll.lib
 *
 * Run:
 *   ./custom_sampler /path/to/model_dir "Hello, world"
 */
#include "onnx_genai.h"
#include <stdio.h>
#include <stdlib.h>

/* A minimal greedy (argmax) sampler implemented entirely in C. The Rust side
 * already applied temperature/top-k/top-p/penalties to `logits`; we just pick
 * the highest-scoring token. Swap this out for beam bookkeeping, a grammar, a
 * watermarking scheme, an RNG of your choice, etc. */
static uint32_t argmax_sample(void *user_data, const float *logits,
                              size_t logits_len, const uint32_t *generated,
                              size_t generated_len, size_t step) {
  (void)user_data;
  (void)generated;
  (void)generated_len;
  (void)step;
  uint32_t best = 0;
  float best_logit = -1.0f / 0.0f; /* -inf */
  for (size_t i = 0; i < logits_len; ++i) {
    if (logits[i] > best_logit) {
      best_logit = logits[i];
      best = (uint32_t)i;
    }
  }
  return best;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model_dir> <prompt>\n", argv[0]);
    return 2;
  }

  OgeEngine *engine = oge_engine_load(argv[1]);
  if (!engine) {
    fprintf(stderr, "load failed: %s\n", oge_last_error());
    return 1;
  }

  OgeSamplerVTable vtable = {0};
  vtable.sample = argmax_sample;
  vtable.name = "c_argmax";
  OgeSampler *sampler = oge_sampler_new(vtable);
  if (!sampler) {
    fprintf(stderr, "sampler creation failed: %s\n", oge_last_error());
    oge_engine_free(engine);
    return 1;
  }

  /* Consumes `sampler`. */
  char *text = oge_engine_generate_with_sampler(engine, argv[2], 64, sampler);
  if (!text) {
    fprintf(stderr, "generate failed: %s\n", oge_last_error());
    oge_engine_free(engine);
    return 1;
  }

  printf("%s\n", text);
  oge_string_free(text);
  oge_engine_free(engine);
  return 0;
}
