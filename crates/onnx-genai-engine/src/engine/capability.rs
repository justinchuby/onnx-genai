//! What a package cannot do, said once and typed.
//!
//! A caller asking for something the loaded package does not declare is not a
//! server fault and not a malformed request — it is a mismatch between what was
//! asked for and what this package supports. A front end has to tell those
//! apart to answer with the right status, and matching on prose to do it is a
//! guess that goes stale the first time a message is reworded.

/// A request the loaded package cannot serve as asked.
///
/// Every variant is the caller and the package disagreeing, never the server
/// failing, so a front end answers all of them with a 4xx it can choose from the
/// variant rather than from the wording.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
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
    /// The conversation this turn would leave exceeds the bound the package
    /// declares for it. The caller can shorten the turn, ask for fewer tokens,
    /// or start a new session.
    #[error(
        "this turn would leave a conversation of {reached} tokens and '{cell}' declares a bound \
         of {bound}; reset or close the session to start a new one, or ask for fewer tokens"
    )]
    ConversationOverBound {
        /// The state cell whose declared bound would be exceeded.
        cell: String,
        /// Tokens the conversation would hold.
        reached: usize,
        /// The bound the package declares.
        bound: usize,
    },
    /// A turn is already in flight for this session, whose lease is declared
    /// exclusive. Running both would let one read a conversation the other is
    /// about to replace.
    #[error(
        "session {session} already has a turn in flight, and its workflow declares the \
         conversation an exclusive lease; run the turns one after another so neither reads a \
         conversation the other is about to replace"
    )]
    SessionBusy {
        /// The session with a turn already running.
        session: String,
    },
}

impl PackageCapabilityError {
    /// Whether this is a transient conflict the same request can succeed at
    /// later, as opposed to a package that will never serve it.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::SessionBusy { .. })
    }
}

/// The package capability a failure names, if it names one.
///
/// Reads the whole chain, because the refusal is raised deep in the engine and
/// reaches a front end wrapped in whatever context the path added. Matching the
/// type is what keeps a status code from depending on the wording of a message.
pub fn package_capability_error(error: &anyhow::Error) -> Option<PackageCapabilityError> {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<PackageCapabilityError>().cloned())
}

/// What a session already holds in front of the next turn's prompt.
///
/// Two numbers, because two different questions are asked of the same session
/// and they have different answers:
///
/// * `attended` is what the model's window has to fit *besides* this request —
///   what a context cap, an output budget and `usage.prompt_tokens` are about;
/// * `reprefilled` is how much of that this turn will compute again — what a
///   "prompt tokens processed" metric is about.
///
/// They coincide only for a package that carries its conversation by prepending
/// it. A decode core that appends each turn to a retained sequence attends to
/// everything it holds and re-prefills none of it, because the rest is served
/// from its KV.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionPrefillCarry {
    /// Tokens the model holds in front of this turn's prompt.
    pub attended: usize,
    /// Of those, the tokens this turn will prefill again.
    pub reprefilled: usize,
}
