# Classify im2col rows once

The valid horizontal samples are exactly the output columns satisfying
`pad_left <= output_col * stride_width + kernel_col * dilation_width < pad_left + input_width`.
Solve the interval once for each channel/kernel-column traversal, then reuse it
for every output row. Invalid rows and horizontal padding are filled with +0.0.
Valid horizontal stride-one spans use a slice copy, regardless of dilation;
other positive strides sample the same span. All indices remain checked Rust
slices. The factory already rejects padded-extent overflow and zero strides.

No new retained plan, allocation, cache, quota, public API or capability gate is
introduced. The staged output, one existing optional im2col buffer and GEMM
reduction order remain. Copying f32 values preserves all sample bits, including
signed zero and nonfinite payloads. The 6912-case independent signed-coordinate
matrix checks the entire im2col buffer after poisoning it, including empty axes,
asymmetric/large padding, dilation, nonunit horizontal/vertical strides and
multiple channels. Existing complete-Conv and allocation tests remain.

This implementation is not inferred from a model/shape identity or a proposed
NCHWc packing mechanism. The deciding property is a contiguous horizontal sample
span; dilation changes the span's starting coordinate without changing its step.
