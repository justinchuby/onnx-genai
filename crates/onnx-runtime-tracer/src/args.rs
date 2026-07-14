//! Structured [`Args`] payloads attached to trace events.
//!
//! Chrome Trace Event Format lets every event carry an arbitrary JSON `args`
//! object. Perfetto and `chrome://tracing` render these as key/value rows when
//! a span is selected, so this is where runtime-specific metadata lives:
//! the device a kernel ran on, its input shapes, byte counts moved by a
//! transfer, estimated FLOPs, and so on.
//!
//! [`Args`] is a thin, ergonomic builder over a [`serde_json::Map`]. The
//! common runtime fields have named helpers; anything else can be attached
//! with [`Args::with`]. All helpers take `self` and return `Self`, so they
//! chain:
//!
//! ```
//! use onnx_runtime_tracer::Args;
//!
//! let args = Args::new()
//!     .device("cuda:0")
//!     .shapes(vec![vec![1_i64, 4096, 4096]])
//!     .flops(34_359_738_368_u64)
//!     .with("fused", true);
//! assert_eq!(args["device"], "cuda:0");
//! ```

use serde_json::{Map, Value};

/// A builder for the JSON `args` object attached to a trace event.
///
/// Cheap to construct (`Args::new` allocates an empty map) and cheap to move.
/// Convert to a [`serde_json::Value`] with [`Args::into_value`] or via `From`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Args(Map<String, Value>);

impl Args {
    /// Create an empty args object.
    #[must_use]
    pub fn new() -> Self {
        Self(Map::new())
    }

    /// Insert an arbitrary key/value pair.
    ///
    /// This is the escape hatch for metadata without a named helper (op type,
    /// stream id, cache hit/miss, …). Later inserts of the same key overwrite
    /// earlier ones.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.0.insert(key.into(), value.into());
        self
    }

    /// Record the device a span ran on, e.g. `"cpu"` or `"cuda:0"`.
    #[must_use]
    pub fn device(self, device: impl Into<String>) -> Self {
        self.with("device", device.into())
    }

    /// Record the input tensor shapes involved in a span.
    ///
    /// Accepts anything convertible into a JSON value, so
    /// `vec![vec![1_i64, 4096, 4096]]` (a list of shapes) works directly.
    #[must_use]
    pub fn shapes(self, shapes: impl Into<Value>) -> Self {
        self.with("shapes", shapes.into())
    }

    /// Record a byte count (e.g. the size of a host↔device transfer).
    #[must_use]
    pub fn bytes(self, bytes: u64) -> Self {
        self.with("bytes", bytes)
    }

    /// Record an estimated floating-point-operation count for a kernel.
    #[must_use]
    pub fn flops(self, flops: u64) -> Self {
        self.with("flops", flops)
    }

    /// Record the source endpoint of a transfer, e.g. `"cuda:0"`.
    #[must_use]
    pub fn src(self, src: impl Into<String>) -> Self {
        self.with("src", src.into())
    }

    /// Record the destination endpoint of a transfer, e.g. `"cpu"`.
    #[must_use]
    pub fn dst(self, dst: impl Into<String>) -> Self {
        self.with("dst", dst.into())
    }

    /// Whether any metadata has been attached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of attached keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Consume the builder and return the underlying JSON object value.
    #[must_use]
    pub fn into_value(self) -> Value {
        Value::Object(self.0)
    }
}

impl<K, V> FromIterator<(K, V)> for Args
where
    K: Into<String>,
    V: Into<Value>,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Map::new();
        for (key, value) in iter {
            map.insert(key.into(), value.into());
        }
        Self(map)
    }
}

impl From<Args> for Value {
    fn from(args: Args) -> Self {
        args.into_value()
    }
}

impl std::ops::Index<&str> for Args {
    type Output = Value;

    /// Index by key for convenient assertions in tests. Panics if the key is
    /// absent, matching [`serde_json::Value`]'s own indexing contract.
    fn index(&self, key: &str) -> &Self::Output {
        &self.0[key]
    }
}
