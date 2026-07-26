### 2026-07-25: Cover the second AVX-512 vector's final lane
**By:** Pris
**What:** Corrected PR #168's two-vector NaN test to use a 32-element block and place the NaN at index 31.
**Why:** The prior test wrote index 15, which only exercised the first vector's final lane and did not cover non-finite detection in the second vector.
