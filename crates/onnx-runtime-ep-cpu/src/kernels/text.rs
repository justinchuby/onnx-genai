//! Native host-string kernels for ONNX operators.

use std::ffi::{c_char, c_void};
use std::ptr::NonNull;

use onnx_runtime_ep_api::{
    EpError, HostKernelData, HostKernelInput, HostKernelOutput, HostStringInput, Kernel,
    KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{Attribute, DataType, Node};

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(message.into())
}

fn raw_abi_error(op: &str) -> EpError {
    error(format!(
        "{op}: this kernel accepts managed host strings only; dispatch it through \
         Kernel::execute_managed_host instead of the raw tensor ABI"
    ))
}

fn checked_numel(op: &str, input: usize, shape: &[usize], values: &[String]) -> Result<usize> {
    let numel = shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).ok_or_else(|| {
            error(format!(
                "{op}: input {input} shape {shape:?} overflows the host element count"
            ))
        })
    })?;
    if numel != values.len() {
        return Err(error(format!(
            "{op}: input {input} shape {shape:?} contains {numel} elements, but {} strings were supplied",
            values.len()
        )));
    }
    Ok(numel)
}

fn require_inputs<'a, const N: usize>(
    op: &str,
    inputs: &[HostKernelInput<'a>],
) -> Result<[HostStringInput<'a>; N]> {
    let inputs: &[_; N] = inputs.try_into().map_err(|_| {
        error(format!(
            "{op}: expected {N} host string input(s), got {}",
            inputs.len()
        ))
    })?;
    let mut strings = [HostStringInput {
        shape: &[],
        values: &[],
    }; N];
    for (index, input) in inputs.iter().enumerate() {
        let HostKernelInput::Strings(input) = input else {
            return Err(error(format!(
                "{op}: input {index} must be a managed tensor(string)"
            )));
        };
        checked_numel(op, index, input.shape, input.values)?;
        strings[index] = *input;
    }
    Ok(strings)
}

fn broadcast_shape(op: &str, lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut output = vec![1usize; rank];
    for (output_axis, output_extent) in output.iter_mut().enumerate() {
        let lhs_axis = lhs.len() as isize - rank as isize + output_axis as isize;
        let rhs_axis = rhs.len() as isize - rank as isize + output_axis as isize;
        let a = if lhs_axis >= 0 {
            lhs[lhs_axis as usize]
        } else {
            1
        };
        let b = if rhs_axis >= 0 {
            rhs[rhs_axis as usize]
        } else {
            1
        };
        *output_extent = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(error(format!(
                "{op}: input shapes {lhs:?} and {rhs:?} are not bidirectionally broadcastable \
                 (axis {output_axis} has extents {a} and {b})"
            )));
        };
    }
    Ok(output)
}

fn broadcast_strides(input: &[usize], output: &[usize]) -> Vec<usize> {
    let mut dense = vec![0usize; input.len()];
    let mut stride = 1usize;
    for axis in (0..input.len()).rev() {
        dense[axis] = stride;
        stride *= input[axis];
    }
    let lead = output.len() - input.len();
    let mut aligned = vec![0usize; output.len()];
    for (axis, &dim) in input.iter().enumerate() {
        if dim != 1 {
            aligned[lead + axis] = dense[axis];
        }
    }
    aligned
}

pub struct StringConcatKernel;
pub struct StringConcatFactory;

pub(super) fn broadcast_string_pairs<T>(
    op: &str,
    inputs: &[HostKernelInput<'_>],
    mut apply: impl FnMut(&str, &str) -> T,
) -> Result<(Vec<usize>, Vec<T>)> {
    let inputs = require_inputs::<2>(op, inputs)?;
    let shape = broadcast_shape(op, inputs[0].shape, inputs[1].shape)?;
    // No strides or input elements are needed for an empty broadcast.
    if shape.contains(&0) {
        return Ok((shape, Vec::new()));
    }
    let count = shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .ok_or_else(|| error(format!("{op}: broadcast output element count overflowed")))
    })?;
    let lhs_strides = broadcast_strides(inputs[0].shape, &shape);
    let rhs_strides = broadcast_strides(inputs[1].shape, &shape);
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|cause| {
        error(format!(
            "{op}: cannot allocate output of shape {shape:?}: {cause}"
        ))
    })?;
    let mut index = vec![0usize; shape.len()];
    let (mut lhs_offset, mut rhs_offset) = (0usize, 0usize);

    for produced in 0..count {
        values.push(apply(
            &inputs[0].values[lhs_offset],
            &inputs[1].values[rhs_offset],
        ));
        if produced + 1 == count {
            break;
        }
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                lhs_offset += lhs_strides[axis];
                rhs_offset += rhs_strides[axis];
                break;
            }
            index[axis] = 0;
            lhs_offset -= lhs_strides[axis] * (shape[axis] - 1);
            rhs_offset -= rhs_strides[axis] * (shape[axis] - 1);
        }
    }
    Ok((shape, values))
}

impl KernelFactory for StringConcatFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(StringConcatKernel))
    }
}

impl Kernel for StringConcatKernel {
    fn execute_managed_host(
        &self,
        inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        let (shape, values) = broadcast_string_pairs("StringConcat", inputs, |lhs, rhs| {
            let mut value = String::with_capacity(lhs.len() + rhs.len());
            value.push_str(lhs);
            value.push_str(rhs);
            value
        })?;
        Ok(vec![HostKernelOutput {
            shape,
            data: HostKernelData::Strings(values),
        }])
    }

    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
        Err(raw_abi_error("StringConcat"))
    }
}

pub struct RegexFullMatchKernel {
    regex: ExactRe2,
}

pub struct RegexFullMatchFactory;

impl KernelFactory for RegexFullMatchFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let pattern = node
            .attr("pattern")
            .ok_or_else(|| error("RegexFullMatch: required `pattern` attribute is missing"))?
            .as_str()
            .ok_or_else(|| {
                error("RegexFullMatch: `pattern` must be a valid UTF-8 string attribute")
            })?;
        let regex = ExactRe2::compile(pattern)?;
        Ok(Box::new(RegexFullMatchKernel { regex }))
    }
}

impl Kernel for RegexFullMatchKernel {
    fn execute_managed_host(
        &self,
        inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        let inputs = require_inputs::<1>("RegexFullMatch", inputs)?;
        let bytes = inputs[0]
            .values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                self.regex.full_match(value).map(u8::from).map_err(|cause| {
                    error(format!(
                        "RegexFullMatch: failed to match input 0 element {index}: {cause}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(vec![HostKernelOutput {
            shape: inputs[0].shape.to_vec(),
            data: HostKernelData::Raw {
                dtype: DataType::Bool,
                bytes,
            },
        }])
    }

    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
        Err(raw_abi_error("RegexFullMatch"))
    }
}

struct ExactRe2 {
    handle: NonNull<c_void>,
}

// RE2 documents concurrent matching on one compiled object as safe. This
// kernel only exposes immutable matches after construction.
unsafe impl Send for ExactRe2 {}

impl ExactRe2 {
    fn compile(pattern: &str) -> Result<Self> {
        let handle = unsafe { nxrt_re2_new(pattern.as_ptr().cast::<c_char>(), pattern.len()) };
        let handle = NonNull::new(handle).ok_or_else(|| {
            error("RegexFullMatch: Google RE2 allocation failed while compiling `pattern`")
        })?;
        if unsafe { nxrt_re2_ok(handle.as_ptr()) } {
            return Ok(Self { handle });
        }
        let mut error_len = 0usize;
        let error_ptr = unsafe { nxrt_re2_error(handle.as_ptr(), &mut error_len) };
        let detail = if error_ptr.is_null() {
            "unknown RE2 compile error".to_owned()
        } else {
            let bytes = unsafe { std::slice::from_raw_parts(error_ptr.cast::<u8>(), error_len) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        unsafe { nxrt_re2_delete(handle.as_ptr()) };
        Err(error(format!(
            "RegexFullMatch: invalid Google RE2 pattern {pattern:?}: {detail}. \
             Rewrite `pattern` using RE2 release 2022-06-01 syntax."
        )))
    }

    fn full_match(&self, value: &str) -> Result<bool> {
        match unsafe {
            nxrt_re2_full_match(
                self.handle.as_ptr(),
                value.as_ptr().cast::<c_char>(),
                value.len(),
            )
        } {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(error(
                "RegexFullMatch: Google RE2 threw while matching an input string; \
                 the native bridge contained the exception and aborted this kernel execution",
            )),
        }
    }
}

impl Drop for ExactRe2 {
    fn drop(&mut self) {
        unsafe { nxrt_re2_delete(self.handle.as_ptr()) };
    }
}

unsafe extern "C" {
    fn nxrt_re2_new(pattern: *const c_char, pattern_len: usize) -> *mut c_void;
    fn nxrt_re2_ok(handle: *const c_void) -> bool;
    fn nxrt_re2_error(handle: *const c_void, error_len: *mut usize) -> *const c_char;
    fn nxrt_re2_full_match(handle: *const c_void, text: *const c_char, text_len: usize) -> i32;
    fn nxrt_re2_delete(handle: *mut c_void);
}

#[derive(Clone)]
enum SplitDelimiter {
    Whitespace,
    Literal(String),
}

pub struct StringSplitKernel {
    delimiter: SplitDelimiter,
    maxsplit: Option<i64>,
}

pub struct StringSplitFactory;

impl KernelFactory for StringSplitFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let delimiter = match node.attr("delimiter") {
            None => SplitDelimiter::Whitespace,
            Some(attribute) => {
                let value = attribute.as_str().ok_or_else(|| {
                    error("StringSplit: `delimiter` must be a valid UTF-8 string attribute")
                })?;
                if value.is_empty() {
                    SplitDelimiter::Whitespace
                } else {
                    SplitDelimiter::Literal(value.to_owned())
                }
            }
        };
        let maxsplit = match node.attr("maxsplit") {
            None => None,
            Some(Attribute::Int(value)) => Some(*value),
            Some(_) => {
                return Err(error(
                    "StringSplit: `maxsplit` must be an integer attribute when present",
                ));
            }
        };
        Ok(Box::new(StringSplitKernel {
            delimiter,
            maxsplit,
        }))
    }
}

impl Kernel for StringSplitKernel {
    fn execute_managed_host(
        &self,
        inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        let inputs = require_inputs::<1>("StringSplit", inputs)?;
        let mut counts = Vec::with_capacity(inputs[0].values.len());
        let mut width = 0usize;
        for value in inputs[0].values {
            let mut count = 0usize;
            self.visit_parts(value, |_| count += 1);
            width = width.max(count);
            counts.push(i64::try_from(count).map_err(|_| {
                error("StringSplit: substring count does not fit the required int64 output")
            })?);
        }

        let capacity = inputs[0]
            .values
            .len()
            .checked_mul(width)
            .ok_or_else(|| error("StringSplit: padded output element count overflowed"))?;
        let mut substrings = Vec::with_capacity(capacity);
        for value in inputs[0].values {
            let start = substrings.len();
            self.visit_parts(value, |part| substrings.push(part.to_owned()));
            substrings.resize_with(start + width, String::new);
        }

        let mut substring_shape = inputs[0].shape.to_vec();
        substring_shape.push(width);
        let count_bytes = counts
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>();
        Ok(vec![
            HostKernelOutput {
                shape: substring_shape,
                data: HostKernelData::Strings(substrings),
            },
            HostKernelOutput {
                shape: inputs[0].shape.to_vec(),
                data: HostKernelData::Raw {
                    dtype: DataType::Int64,
                    bytes: count_bytes,
                },
            },
        ])
    }

    fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
        Err(raw_abi_error("StringSplit"))
    }
}

impl StringSplitKernel {
    fn visit_parts<'a>(&self, value: &'a str, mut visit: impl FnMut(&'a str)) {
        match &self.delimiter {
            SplitDelimiter::Literal(delimiter) => match self.maxsplit {
                Some(0) => visit(value),
                Some(maxsplit) if maxsplit > 0 => {
                    let fields = usize::try_from(maxsplit)
                        .unwrap_or(usize::MAX)
                        .saturating_add(1);
                    for part in value.splitn(fields, delimiter) {
                        visit(part);
                    }
                }
                _ => {
                    for part in value.split(delimiter) {
                        visit(part);
                    }
                }
            },
            SplitDelimiter::Whitespace => {
                visit_python_whitespace_parts(value, self.maxsplit, visit)
            }
        }
    }
}

fn python_whitespace(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{001c}'..='\u{001f}')
}

fn visit_python_whitespace_parts<'a>(
    value: &'a str,
    maxsplit: Option<i64>,
    mut visit: impl FnMut(&'a str),
) {
    let mut cursor = value
        .char_indices()
        .find_map(|(offset, ch)| (!python_whitespace(ch)).then_some(offset))
        .unwrap_or(value.len());
    if cursor == value.len() {
        return;
    }
    if matches!(maxsplit, Some(0)) {
        visit(&value[cursor..]);
        return;
    }

    let limit = maxsplit
        .filter(|&value| value > 0)
        .map(|value| value as u64);
    let mut splits = 0u64;
    loop {
        if limit.is_some_and(|limit| splits == limit) {
            visit(&value[cursor..]);
            return;
        }
        let separator = value[cursor..]
            .char_indices()
            .find_map(|(relative, ch)| python_whitespace(ch).then_some(cursor + relative));
        let Some(separator) = separator else {
            visit(&value[cursor..]);
            return;
        };
        visit(&value[cursor..separator]);
        splits += 1;
        cursor = value[separator..]
            .char_indices()
            .find_map(|(relative, ch)| (!python_whitespace(ch)).then_some(separator + relative))
            .unwrap_or(value.len());
        if cursor == value.len() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::NodeId;

    fn strings<'a>(shape: &'a [usize], values: &'a [String]) -> HostKernelInput<'a> {
        HostKernelInput::Strings(HostStringInput { shape, values })
    }

    fn string_data(output: &HostKernelOutput) -> &[String] {
        match &output.data {
            HostKernelData::Strings(values) => values,
            HostKernelData::Raw { .. } => panic!("expected string output"),
        }
    }

    fn make_split(delimiter: Option<&str>, maxsplit: Option<i64>) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "StringSplit", vec![], vec![]);
        if let Some(delimiter) = delimiter {
            node.attributes.insert(
                "delimiter".into(),
                Attribute::String(delimiter.as_bytes().to_vec()),
            );
        }
        if let Some(maxsplit) = maxsplit {
            node.attributes
                .insert("maxsplit".into(), Attribute::Int(maxsplit));
        }
        StringSplitFactory.create(&node, &[]).unwrap()
    }

    #[test]
    fn string_concat_broadcasts_without_materializing_inputs() {
        let lhs = ["cat", "dog", "蛇"].map(str::to_owned);
        let rhs = ["s", "!"].map(str::to_owned);
        let outputs = StringConcatKernel
            .execute_managed_host(&[strings(&[3, 1], &lhs), strings(&[1, 2], &rhs)])
            .unwrap();
        assert_eq!(outputs[0].shape, [3, 2]);
        assert_eq!(
            string_data(&outputs[0]),
            ["cats", "cat!", "dogs", "dog!", "蛇s", "蛇!"]
        );
    }

    #[test]
    fn string_concat_handles_scalars_and_zero_extent_broadcasts() {
        let lhs = ["cat".to_owned()];
        let rhs = ["s".to_owned()];
        let outputs = StringConcatKernel
            .execute_managed_host(&[strings(&[], &lhs), strings(&[], &rhs)])
            .unwrap();
        assert!(outputs[0].shape.is_empty());
        assert_eq!(string_data(&outputs[0]), ["cats"]);

        let empty: [String; 0] = [];
        let singleton = ["suffix".to_owned()];
        let outputs = StringConcatKernel
            .execute_managed_host(&[strings(&[2, 0, 3], &empty), strings(&[1, 1, 3], &singleton)])
            .err()
            .expect("shape/value mismatch must fail");
        assert!(
            outputs.to_string().contains("but 1 strings were supplied"),
            "shape/value validation must run before zero-extent broadcasting"
        );

        let rhs = ["a", "b", "c"].map(str::to_owned);
        let outputs = StringConcatKernel
            .execute_managed_host(&[strings(&[2, 0, 3], &empty), strings(&[1, 1, 3], &rhs)])
            .unwrap();
        assert_eq!(outputs[0].shape, [2, 0, 3]);
        assert!(string_data(&outputs[0]).is_empty());
    }

    #[test]
    fn regex_is_full_match_and_uses_re2_ascii_word_classes() {
        let mut node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        node.attributes.insert(
            "pattern".into(),
            Attribute::String(br"www\.[\w.-]+\.\bcom\b".to_vec()),
        );
        let kernel = RegexFullMatchFactory.create(&node, &[]).unwrap();
        let values = ["www.google.com", "www.bbc.co.uk", "www.中.com", "x\0y"].map(str::to_owned);
        let outputs = kernel
            .execute_managed_host(&[strings(&[4], &values)])
            .unwrap();
        match &outputs[0].data {
            HostKernelData::Raw { dtype, bytes } => {
                assert_eq!(*dtype, DataType::Bool);
                assert_eq!(bytes, &[1, 0, 0, 0]);
            }
            HostKernelData::Strings(_) => panic!("expected bool output"),
        }
    }

    #[test]
    fn regex_uses_exact_re2_alternation_and_character_class_semantics() {
        let mut node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        node.attributes.insert(
            "pattern".into(),
            Attribute::String(br"(?:a|ab)|[\D]".to_vec()),
        );
        let kernel = RegexFullMatchFactory.create(&node, &[]).unwrap();
        let values = ["ab", "é", "7"].map(str::to_owned);
        let outputs = kernel
            .execute_managed_host(&[strings(&[3], &values)])
            .unwrap();
        match &outputs[0].data {
            HostKernelData::Raw { bytes, .. } => assert_eq!(bytes, &[1, 1, 0]),
            HostKernelData::Strings(_) => panic!("expected bool output"),
        }
    }

    #[test]
    fn regex_re2_ascii_whitespace_excludes_vertical_tab() {
        let mut node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        node.attributes
            .insert("pattern".into(), Attribute::String(br"\s".to_vec()));
        let kernel = RegexFullMatchFactory.create(&node, &[]).unwrap();
        let values = [" ", "\t", "\n", "\u{000c}", "\u{000b}"].map(str::to_owned);
        let outputs = kernel
            .execute_managed_host(&[strings(&[5], &values)])
            .unwrap();
        match &outputs[0].data {
            HostKernelData::Raw { bytes, .. } => assert_eq!(bytes, &[1, 1, 1, 1, 0]),
            HostKernelData::Strings(_) => panic!("expected bool output"),
        }
    }

    #[test]
    fn regex_re2_byte_escape_and_quoted_literal_are_exact() {
        let mut bytes_node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        bytes_node
            .attributes
            .insert("pattern".into(), Attribute::String(br"\C\C".to_vec()));
        let bytes_kernel = RegexFullMatchFactory.create(&bytes_node, &[]).unwrap();
        let values = ["é", "a", "\0x"].map(str::to_owned);
        let outputs = bytes_kernel
            .execute_managed_host(&[strings(&[3], &values)])
            .unwrap();
        match &outputs[0].data {
            HostKernelData::Raw { bytes, .. } => assert_eq!(bytes, &[1, 0, 1]),
            HostKernelData::Strings(_) => panic!("expected bool output"),
        }

        let mut quote_node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        quote_node
            .attributes
            .insert("pattern".into(), Attribute::String(br"\Qa+b[0]\E".to_vec()));
        let quote_kernel = RegexFullMatchFactory.create(&quote_node, &[]).unwrap();
        let values = ["a+b[0]", "ab0"].map(str::to_owned);
        let outputs = quote_kernel
            .execute_managed_host(&[strings(&[2], &values)])
            .unwrap();
        match &outputs[0].data {
            HostKernelData::Raw { bytes, .. } => assert_eq!(bytes, &[1, 0]),
            HostKernelData::Strings(_) => panic!("expected bool output"),
        }
    }

    #[test]
    fn regex_reports_re2_compile_errors() {
        let mut node = Node::new(NodeId(0), "RegexFullMatch", vec![], vec![]);
        node.attributes
            .insert("pattern".into(), Attribute::String(br"(?=a)".to_vec()));
        let error = RegexFullMatchFactory
            .create(&node, &[])
            .err()
            .expect("RE2-invalid look-ahead must fail");
        assert!(error.to_string().contains("invalid Google RE2 pattern"));
    }

    #[test]
    fn string_split_matches_python_whitespace_maxsplit_and_padding() {
        let kernel = make_split(None, Some(2));
        let values =
            ["hello world", "def.net", "o n n x", "the quick brown fox"].map(str::to_owned);
        let outputs = kernel
            .execute_managed_host(&[strings(&[2, 2], &values)])
            .unwrap();
        assert_eq!(outputs[0].shape, [2, 2, 3]);
        assert_eq!(
            string_data(&outputs[0]),
            [
                "hello",
                "world",
                "",
                "def.net",
                "",
                "",
                "o",
                "n",
                "n x",
                "the",
                "quick",
                "brown fox"
            ]
        );
        match &outputs[1].data {
            HostKernelData::Raw { dtype, bytes } => {
                assert_eq!(*dtype, DataType::Int64);
                let counts: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                assert_eq!(counts, [2, 1, 3, 3]);
            }
            HostKernelData::Strings(_) => panic!("expected count output"),
        }
    }

    #[test]
    fn string_split_handles_python_whitespace_unicode_controls_and_nul() {
        let whitespace = make_split(None, Some(0));
        let values = ["\u{001c}\u{2003}alpha  beta ", " \t"].map(str::to_owned);
        let outputs = whitespace
            .execute_managed_host(&[strings(&[2], &values)])
            .unwrap();
        assert_eq!(outputs[0].shape, [2, 1]);
        assert_eq!(string_data(&outputs[0]), ["alpha  beta ", ""]);

        let nul = make_split(Some("\0"), None);
        let values = ["a\0b\0", "\0x"].map(str::to_owned);
        let outputs = nul.execute_managed_host(&[strings(&[2], &values)]).unwrap();
        assert_eq!(outputs[0].shape, [2, 3]);
        assert_eq!(string_data(&outputs[0]), ["a", "b", "", "", "x", ""]);
    }

    #[test]
    fn string_split_empty_tensor_has_runtime_zero_width() {
        let outputs = make_split(Some("."), None)
            .execute_managed_host(&[strings(&[0], &[])])
            .unwrap();
        assert_eq!(outputs[0].shape, [0, 0]);
        assert!(string_data(&outputs[0]).is_empty());
        assert_eq!(outputs[1].shape, [0]);
    }

    #[test]
    fn text_descriptors_are_native_only_by_slot_dtype_property() {
        let (_, descriptors) = crate::kernels::build_cpu_registry_with_descriptors();
        let native_only: Vec<&str> = descriptors
            .iter()
            .filter(|descriptor| !descriptor.supports_raw_tensor_abi)
            .map(|descriptor| descriptor.op_type.as_str())
            .collect();
        assert_eq!(
            native_only,
            ["RegexFullMatch", "StringConcat", "StringSplit"],
            "only kernels with a required string-only slot may be excluded from the raw ABI"
        );

        for op in native_only {
            let constraints = crate::kernels::input_dtype_constraints_for_op(op, "")
                .iter()
                .chain(crate::kernels::output_dtype_constraints_for_op(op, ""));
            assert!(
                constraints
                    .into_iter()
                    .any(|(_, dtypes)| { *dtypes == [DataType::String] }),
                "{op} must derive native-only capability from a required string-only slot"
            );
        }
    }
}
