#include <cstddef>
#include <new>

#include "re2/re2.h"

extern "C" {

void* nxrt_re2_new(const char* pattern, size_t pattern_len) noexcept {
  try {
    RE2::Options options;
    options.set_log_errors(false);
    return new (std::nothrow) RE2(
        re2::StringPiece(pattern, pattern_len), options);
  } catch (...) {
    return nullptr;
  }
}

bool nxrt_re2_ok(const void* handle) noexcept {
  return handle != nullptr && static_cast<const RE2*>(handle)->ok();
}

const char* nxrt_re2_error(const void* handle, size_t* error_len) noexcept {
  if (handle == nullptr) {
    static const char error[] = "RE2 allocation failed";
    *error_len = sizeof(error) - 1;
    return error;
  }
  const std::string& error = static_cast<const RE2*>(handle)->error();
  *error_len = error.size();
  return error.data();
}

int nxrt_re2_full_match(const void* handle, const char* text,
                        size_t text_len) noexcept {
  try {
    const RE2* regex = static_cast<const RE2*>(handle);
    return regex->Match(re2::StringPiece(text, text_len), 0, text_len,
                        RE2::ANCHOR_BOTH, nullptr, 0)
               ? 1
               : 0;
  } catch (...) {
    return -1;
  }
}

void nxrt_re2_delete(void* handle) noexcept {
  delete static_cast<RE2*>(handle);
}

}  // extern "C"
