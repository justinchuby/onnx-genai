//! Borrowed or shared ownership for stateful resources derived from an ORT session.

use std::ops::Deref;
use std::sync::Arc;

use crate::{Allocator, IoBinding, Result, Session};

/// Immutable session ownership carried by a stateful ORT runner.
///
/// The shared form contains only the `Session`. Mutable bindings, allocators,
/// device values, KV state, and graph-capture state remain owned by each runner.
pub(crate) enum OrtSessionOwner<'a> {
    Borrowed(&'a Session),
    Shared(Arc<Session>),
}

impl<'a> OrtSessionOwner<'a> {
    pub(crate) fn borrowed(session: &'a Session) -> Self {
        Self::Borrowed(session)
    }

    pub(crate) fn shared(session: Arc<Session>) -> OrtSessionOwner<'static> {
        OrtSessionOwner::Shared(session)
    }

    /// Create a binding with the same ownership edge as this owner.
    pub(crate) fn binding(&self) -> Result<IoBinding<'a>> {
        match self {
            Self::Borrowed(session) => IoBinding::new(session),
            Self::Shared(session) => IoBinding::for_shared_session(Arc::clone(session)),
        }
    }

    /// Query the session device allocator with the same ownership edge as this owner.
    pub(crate) fn device_allocator(&self) -> Result<Option<Allocator<'a>>> {
        match self {
            Self::Borrowed(session) => session.device_kv_allocator(),
            Self::Shared(session) => Session::shared_device_allocator(session),
        }
    }
}

impl Deref for OrtSessionOwner<'_> {
    type Target = Session;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(session) => session,
            Self::Shared(session) => session,
        }
    }
}
