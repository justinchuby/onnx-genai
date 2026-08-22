//! The ComfyUI API-format graph, read structurally.
//!
//! A ComfyUI *"Save (API Format)"* export is a flat map
//! `{node_id: {"class_type": str, "inputs": {port: value | link}}}` where a
//! value of the form `[src_id, slot]` is a *link* to another node's output.
//!
//! Everything in this module is pure graph reading. It knows what a node, a
//! link, and a reachable set are; it knows nothing about diffusion, samplers,
//! or the canonical workflow IR. Recognition lives in
//! [`crate::recognize`](../recognize/index.html) and lowering lives in
//! [`crate::lower`](../lower/index.html), so a Comfy-specific spelling can
//! never leak past the recognizer into emitted metadata.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::ComfyUiConfigError;

/// A resolved reference to one output slot of one node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Link {
    /// Producing node id.
    pub node: String,
    /// Output slot index on the producing node.
    pub slot: usize,
}

/// One ComfyUI node with its class and its already-classified input ports.
#[derive(Debug, Clone)]
pub struct Node {
    /// Node id as spelled in the workflow document.
    pub id: String,
    /// ComfyUI `class_type`.
    pub class: String,
    /// Ports whose value is a link to another node's output.
    pub links: BTreeMap<String, Link>,
    /// Ports whose value is a literal widget value.
    pub literals: BTreeMap<String, Value>,
}

impl Node {
    /// The link on `port`, if the port is connected.
    pub fn link(&self, port: &str) -> Option<&Link> {
        self.links.get(port)
    }

    /// The literal value on `port`, if the port carries one.
    pub fn literal(&self, port: &str) -> Option<&Value> {
        self.literals.get(port)
    }

    /// A required literal string widget.
    pub fn string(&self, port: &str) -> Result<String, ComfyUiConfigError> {
        self.literal(port)
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| self.missing(port, "a string widget value"))
    }

    /// A required literal integer widget.
    pub fn integer(&self, port: &str) -> Result<i64, ComfyUiConfigError> {
        self.literal(port)
            .and_then(Value::as_i64)
            .ok_or_else(|| self.missing(port, "an integer widget value"))
    }

    /// A required literal float widget. Integers widen, because ComfyUI writes
    /// `1` for a float widget left at a whole number.
    pub fn float(&self, port: &str) -> Result<f64, ComfyUiConfigError> {
        self.literal(port)
            .and_then(Value::as_f64)
            .ok_or_else(|| self.missing(port, "a numeric widget value"))
    }

    /// An optional literal float widget, absent when the port is not present.
    pub fn optional_float(&self, port: &str) -> Option<f64> {
        self.literal(port).and_then(Value::as_f64)
    }

    /// An optional literal integer widget.
    pub fn optional_integer(&self, port: &str) -> Option<i64> {
        self.literal(port).and_then(Value::as_i64)
    }

    /// Fail-closed error naming this node, the port, and what was expected.
    pub fn missing(&self, port: &str, expected: &str) -> ComfyUiConfigError {
        ComfyUiConfigError::Unrepresentable {
            node: self.id.clone(),
            class: self.class.clone(),
            detail: format!("port '{port}' does not carry {expected}"),
            remedy: format!(
                "re-export the workflow with '{port}' set to a literal widget value on \
                 node {} ({}), or convert the upstream primitive node into a widget",
                self.id, self.class
            ),
        }
    }

    /// Fail-closed error naming this node and an unsupported structural fact.
    pub fn unsupported(
        &self,
        detail: impl Into<String>,
        remedy: impl Into<String>,
    ) -> ComfyUiConfigError {
        ComfyUiConfigError::Unrepresentable {
            node: self.id.clone(),
            class: self.class.clone(),
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

/// A whole ComfyUI workflow document, indexed by node id.
#[derive(Debug, Clone)]
pub struct ComfyGraph {
    nodes: BTreeMap<String, Node>,
}

impl ComfyGraph {
    /// Read a parsed workflow document.
    ///
    /// A `{"prompt": {...}}` wrapper (what the `/prompt` HTTP endpoint posts) is
    /// unwrapped, because it carries exactly the same node map.
    pub fn from_value(document: &Value) -> Result<Self, ComfyUiConfigError> {
        let object = document
            .as_object()
            .ok_or_else(|| ComfyUiConfigError::NotAWorkflow {
                detail: "the document root is not a JSON object".to_owned(),
            })?;
        let raw = match object.get("prompt") {
            Some(Value::Object(inner)) => inner,
            _ => object,
        };
        if raw.is_empty() {
            return Err(ComfyUiConfigError::NotAWorkflow {
                detail: "the workflow contains no nodes".to_owned(),
            });
        }
        let mut nodes = BTreeMap::new();
        for (id, node) in raw {
            nodes.insert(id.clone(), read_node(id, node)?);
        }
        Ok(Self { nodes })
    }

    /// Every node, in stable id order.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// The node with `id`, or a fail-closed dangling-link error.
    pub fn node(&self, id: &str) -> Result<&Node, ComfyUiConfigError> {
        self.nodes
            .get(id)
            .ok_or_else(|| ComfyUiConfigError::DanglingLink {
                node: id.to_owned(),
            })
    }

    /// Follow `link` to the node that produces it.
    pub fn target(&self, link: &Link) -> Result<&Node, ComfyUiConfigError> {
        self.node(&link.node)
    }

    /// Follow a required port of `node` to the node it is connected to.
    pub fn follow(&self, node: &Node, port: &str) -> Result<(&Node, usize), ComfyUiConfigError> {
        let link = node
            .link(port)
            .ok_or_else(|| node.missing(port, "a link to another node's output"))?;
        Ok((self.target(link)?, link.slot))
    }

    /// Follow an optional port, yielding `None` when it is not connected.
    pub fn follow_optional(
        &self,
        node: &Node,
        port: &str,
    ) -> Result<Option<(&Node, usize)>, ComfyUiConfigError> {
        match node.link(port) {
            Some(link) => Ok(Some((self.target(link)?, link.slot))),
            None => Ok(None),
        }
    }

    /// Every node id reachable upstream from `roots`, inclusive.
    ///
    /// This is the *output path*: the set of nodes whose values can reach the
    /// workflow's image sink. A node outside this set provably cannot change
    /// the produced image, which is the only reason the converter is allowed to
    /// ignore an unrecognized class.
    pub fn upstream_closure(
        &self,
        roots: &[String],
    ) -> Result<BTreeSet<String>, ComfyUiConfigError> {
        let mut seen = BTreeSet::new();
        let mut pending: Vec<String> = roots.to_vec();
        while let Some(id) = pending.pop() {
            if !seen.insert(id.clone()) {
                continue;
            }
            let node = self.node(&id)?;
            for link in node.links.values() {
                // Validate every edge eagerly: a dangling link on the output
                // path is a broken workflow, not something to skip quietly.
                self.node(&link.node)?;
                pending.push(link.node.clone());
            }
        }
        Ok(seen)
    }

    /// Every node whose class is one of `classes`, in stable id order.
    pub fn by_class<'a>(&'a self, classes: &'a [&'a str]) -> impl Iterator<Item = &'a Node> + 'a {
        self.nodes
            .values()
            .filter(move |node| classes.contains(&node.class.as_str()))
    }
}

fn read_node(id: &str, node: &Value) -> Result<Node, ComfyUiConfigError> {
    let object = node
        .as_object()
        .ok_or_else(|| ComfyUiConfigError::NotAWorkflow {
            detail: format!("node '{id}' is not a JSON object"),
        })?;
    let class = object
        .get("class_type")
        .and_then(Value::as_str)
        .ok_or_else(|| ComfyUiConfigError::NotAWorkflow {
            detail: format!("node '{id}' has no 'class_type'"),
        })?
        .to_owned();
    let inputs = match object.get("inputs") {
        Some(Value::Object(inputs)) => inputs.clone(),
        None => Map::new(),
        Some(_) => {
            return Err(ComfyUiConfigError::NotAWorkflow {
                detail: format!("node '{id}' has a non-object 'inputs'"),
            });
        }
    };
    let mut links = BTreeMap::new();
    let mut literals = BTreeMap::new();
    for (port, value) in inputs {
        match as_link(&value) {
            Some(link) => {
                links.insert(port, link);
            }
            None => {
                literals.insert(port, value);
            }
        }
    }
    Ok(Node {
        id: id.to_owned(),
        class,
        links,
        literals,
    })
}

/// A `[src_id, slot]` pair references another node's output.
///
/// ComfyUI writes the source id as a string and the slot as a number. Anything
/// else is a widget value, including a two-element list of numbers, which is
/// why the id must be a string for the pair to count as a link.
fn as_link(value: &Value) -> Option<Link> {
    let array = value.as_array()?;
    if array.len() != 2 {
        return None;
    }
    let node = array[0].as_str()?.to_owned();
    let slot = usize::try_from(array[1].as_u64()?).ok()?;
    Some(Link { node, slot })
}
