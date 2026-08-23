//! What a package cannot do, said once and typed.
//!
//! A caller asking for something the loaded package does not declare is not a
//! server fault and not a malformed request — it is a mismatch between what was
//! asked for and what this package supports. A front end has to tell those
//! apart to answer with the right status, and matching on prose to do it is a
//! guess that goes stale the first time a message is reworded.

/// A capability the loaded package does not declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PackageCapabilityError {
    /// The package publishes a token stream but declares no session-scoped
    /// state anything can carry, so a session over it would restart every turn.
    #[error(
        "this package publishes a token stream but declares no session-scoped workflow state \
         anything carries, so a session could not continue a conversation: every turn would \
         restart from its own prompt. Declare the conversation in `pipeline.workflow.state` with \
         `scope: session` — carried by the loop, held by a state service group, or with a \
         `session.continuation` naming the prompt input it rejoins — and re-validate the package. \
         Until then, use stateless generation and send the conversation in the prompt."
    )]
    NoSessionState,
}

/// The package capability a failure names, if it names one.
///
/// Reads the whole chain, because the refusal is raised deep in the engine and
/// reaches a front end wrapped in whatever context the path added.
pub fn package_capability_error(error: &anyhow::Error) -> Option<PackageCapabilityError> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<PackageCapabilityError>().copied())
}
