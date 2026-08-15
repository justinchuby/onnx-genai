/*
 * onnx_genai.h — C ABI for onnx-genai (crate: onnx-genai-capi)
 *
 * Load a model, generate text, and — the point of this header — plug in your own
 * token sampler. The Rust generation loop still runs the full logit-processor
 * chain (temperature, top-k/top-p, min-p, repetition/frequency/presence
 * penalties, constraints) configured on the request; your sampler only replaces
 * the final token pick that would otherwise be greedy/categorical.
 *
 * Conventions
 * -----------
 *  - Fallible functions return a pointer that is NULL on failure; call
 *    oge_last_error() for a thread-local, human-readable message.
 *  - Every char* returned by this library must be freed with oge_string_free().
 *  - Opaque handles are freed exactly once with their matching *_free function,
 *    which is NULL-tolerant, so `free(x); x = NULL;` prevents double-free.
 *  - No panic/exception crosses the boundary.
 */
#ifndef ONNX_GENAI_H
#define ONNX_GENAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct OgeEngine OgeEngine;
typedef struct OgeSampler OgeSampler;

/*
 * Token-selection callback.
 *
 *  user_data      opaque pointer from OgeSamplerVTable.user_data
 *  logits         post-processor logits for this step (length == vocab size);
 *                 filtered-out tokens are -inf. Read-only, valid only for the
 *                 duration of the call.
 *  logits_len     vocabulary size
 *  generated      token ids generated so far this request (read-only)
 *  generated_len  number of generated tokens
 *  step           0-based decode step index
 *
 * Returns the chosen token id, which must satisfy 0 <= token < logits_len.
 */
typedef uint32_t (*OgeSampleFn)(void *user_data,
                                const float *logits,
                                size_t logits_len,
                                const uint32_t *generated,
                                size_t generated_len,
                                size_t step);

/* Optional destructor for user_data, called once when the sampler is dropped. */
typedef void (*OgeFreeFn)(void *user_data);

/* Description of a foreign sampler. */
typedef struct OgeSamplerVTable {
  void *user_data;   /* passed to every callback; may be NULL              */
  OgeSampleFn sample; /* required                                          */
  const char *name;  /* optional (copied); NULL -> "foreign"              */
  OgeFreeFn free;    /* optional; NULL -> library never frees user_data   */
} OgeSamplerVTable;

/*
 * Create a sampler from a vtable. Returns NULL on failure. Free with
 * oge_sampler_free(), OR pass it to oge_engine_generate_with_sampler(), which
 * takes ownership and frees it for you.
 */
OgeSampler *oge_sampler_new(OgeSamplerVTable vtable);

/* Free a sampler that was NOT consumed by a generate call. NULL-tolerant. */
void oge_sampler_free(OgeSampler *sampler);

/* Load a model directory. Returns NULL on failure. Free with oge_engine_free(). */
OgeEngine *oge_engine_load(const char *model_dir);

/* Free an engine. NULL-tolerant. */
void oge_engine_free(OgeEngine *engine);

/*
 * Generate with the engine's default sampler (greedy/categorical per request
 * options). max_new_tokens == 0 keeps the request default. Returns a heap char*
 * (free with oge_string_free) or NULL on failure.
 */
char *oge_engine_generate(OgeEngine *engine, const char *prompt,
                          size_t max_new_tokens);

/*
 * Generate using your sampler. CONSUMES `sampler` (frees it on success and
 * failure); do not use or free it afterwards. Returns a heap char* (free with
 * oge_string_free) or NULL on failure.
 */
char *oge_engine_generate_with_sampler(OgeEngine *engine, const char *prompt,
                                       size_t max_new_tokens,
                                       OgeSampler *sampler);

/* Free a char* returned by this library. NULL-tolerant. */
void oge_string_free(char *text);

/*
 * Thread-local message for the most recent failure on this thread, or NULL if
 * the last call succeeded. Borrowed — do NOT free; valid until the next
 * fallible call on the same thread.
 */
const char *oge_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ONNX_GENAI_H */
