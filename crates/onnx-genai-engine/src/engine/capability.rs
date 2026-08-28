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
        "this turn would leave a conversation of {requested} tokens and '{cell}' declares a bound \
         of {bound}; reset or close the session to start a new one, or ask for fewer tokens"
    )]
    ConversationOverBound {
        /// The state cell whose declared bound would be exceeded.
        cell: String,
        /// Tokens the conversation would hold if this turn ran.
        requested: usize,
        /// The bound the package declares for that cell.
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
    ExclusiveLeaseConflict {
        /// The session whose exclusive lease is already held.
        session: String,
    },
    /// The package declares a canonical candidate tree this runtime can
    /// validate but cannot execute transactionally.
    #[error(
        "package declares canonical candidate-tree speculation \
         (onnx-genai.speculative@{version}), but this runtime has no candidate-tree \
         package-dispatch capability or executor. Refusing to silently run plain or MTP \
         generation without the declared proposer, target verification, accepted-prefix \
         commit, and rollback participants. Upgrade to a runtime that implements \
         candidate-tree dispatch, or re-export this package with a supported canonical \
         proposal execution."
    )]
    CandidateTreeExecutionUnavailable {
        /// Exact canonical speculation contract version declared by the package.
        version: String,
    },
    /// The package declares a DFlash ABI/backend pair this runtime has not
    /// implemented. The production driver accepts only the base v1 contract
    /// through the portable ORT component execution seam.
    #[error(
        "package declares DFlash flat-block speculation \
         (onnx-genai.dflash-flat-block@{version}) and requires capability '{capability}', but this \
         runtime implements only the exact v1/base contract through the ORT component backend. \
         Refusing before workflow mutation rather than silently running a different speculative \
         mode. Re-export the package as v1/base for the ORT backend or use a runtime that \
         implements this version/backend pair."
    )]
    DFlashExecutionUnavailable {
        /// Exact DFlash contract version declared by the package.
        version: String,
        /// Derived capability that requires the unavailable execution driver.
        capability: String,
    },
    /// DFlash components cannot be executed as an ordinary workflow pass:
    /// doing so would bypass verification and accepted-prefix commit.
    #[error(
        "public workflow operation '{operation}' cannot execute a DFlash package because raw \
         target/proposer execution would bypass verifier-owned acceptance and the S3 \
         accepted-prefix transaction. Use `Engine::generate`, \
         `Engine::generate_in_session`, or `Engine::generate_with_pipeline_request`; the \
         runtime-owned DFlash driver is the only execution authority."
    )]
    DFlashRawWorkflowApi {
        /// Public operation that would have bypassed the DFlash driver.
        operation: String,
    },
}

impl PackageCapabilityError {
    /// Whether this is a transient conflict the same request can succeed at
    /// later, as opposed to a package that will never serve it.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::ExclusiveLeaseConflict { .. })
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
/// `attended` contributes to context admission and usage. `reprefilled`
/// contributes to work metrics. They differ for an ORT decode core, whose
/// retained sequence is attended from KV without being recomputed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionPrefillCarry {
    /// Tokens the model holds in front of this turn's prompt.
    pub attended: usize,
    /// Of those tokens, how many this turn will prefill again.
    pub reprefilled: usize,
}
