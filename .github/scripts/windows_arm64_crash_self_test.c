#include <stdint.h>

__declspec(noinline) void intentional_access_violation_probe(void) {
  volatile uint32_t *invalid = (volatile uint32_t *)(uintptr_t)0;
  *invalid = 0x4e585254;
}

int main(void) {
  intentional_access_violation_probe();
  return 0;
}
