#include <stddef.h>
#include <stdint.h>

typedef void *OrtStatusPtr;

typedef struct {
  uint32_t ort_version_supported;
} OrtEpFactory;

static OrtEpFactory factory = {
    .ort_version_supported = UINT32_MAX,
};

__attribute__((visibility("default")))
OrtStatusPtr CreateEpFactories(
    const char *registration_name,
    const void *api_base,
    const void *logger,
    OrtEpFactory **factories,
    size_t capacity,
    size_t *factory_count) {
  (void)registration_name;
  (void)api_base;
  (void)logger;
  if (capacity != 0) {
    factories[0] = &factory;
    *factory_count = 1;
  }
  return NULL;
}

__attribute__((visibility("default")))
OrtStatusPtr ReleaseEpFactory(OrtEpFactory *factory_to_release) {
  (void)factory_to_release;
  return NULL;
}
