#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef void *OrtStatusPtr;

typedef struct OrtEpFactory {
  uint32_t ort_version_supported;
  void *get_name;
  void *get_vendor;
  void *get_supported_devices;
  OrtStatusPtr (*create_ep)(
      struct OrtEpFactory *factory,
      const void *const *devices,
      const void *const *metadata,
      size_t device_count,
      const void *session_options,
      const void *logger,
      void **ep);
  void (*release_ep)(struct OrtEpFactory *factory, void *ep);
} OrtEpFactory;

static unsigned char synthetic_ep;

static const char *get_name(const OrtEpFactory *factory) {
  (void)factory;
  return "synthetic_legacy_ep";
}

static OrtStatusPtr create_ep(
    OrtEpFactory *factory,
    const void *const *devices,
    const void *const *metadata,
    size_t device_count,
    const void *session_options,
    const void *logger,
    void **ep) {
  (void)factory;
  (void)devices;
  (void)metadata;
  (void)device_count;
  (void)session_options;
  (void)logger;
  *ep = &synthetic_ep;
  return NULL;
}

static void release_ep(OrtEpFactory *factory, void *ep) {
  (void)factory;
  (void)ep;
}

static OrtEpFactory factory = {
    .ort_version_supported = 1,
    .get_name = get_name,
    .get_vendor = NULL,
    .get_supported_devices = NULL,
    .create_ep = create_ep,
    .release_ep = release_ep,
};

__attribute__((visibility("default")))
OrtStatusPtr CreateEpFactories(
    const char *registration_name,
    const void *api_base,
    const void *logger,
    OrtEpFactory **factories,
    size_t capacity,
    size_t *factory_count) {
  (void)api_base;
  (void)logger;
  if (registration_name == NULL ||
      strcmp(registration_name, "synthetic-registration") != 0 ||
      capacity == 0) {
    *factory_count = 0;
    return NULL;
  }
  factories[0] = &factory;
  *factory_count = 1;
  return NULL;
}

__attribute__((visibility("default")))
OrtStatusPtr ReleaseEpFactory(OrtEpFactory *factory_to_release) {
  (void)factory_to_release;
  return NULL;
}
