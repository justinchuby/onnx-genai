use crate::{DeviceBindingOutputs, DeviceIoBinding, InferenceSession, Result, Tensor};

/// Exclusive runner retaining the executor's StepScoped workspace across a
/// bounded active execution phase. It does not expose a replaceable session.
///
/// Call [`Self::finish`] before scheduling yield or phase transition. Returning
/// the owner to its provider does not refund retained physical capacity; ordinary
/// provider completion, capture pinning and quarantine rules continue to apply.
///
/// The session cannot be used while its exclusive runner is alive:
/// ```compile_fail
/// use onnx_runtime_session::{InferenceSession, Result};
/// fn overlapping(session: &mut InferenceSession) -> Result<()> {
///     let scope = session.execution_scope()?;
///     session.run(&[])?;
///     scope.finish()
/// }
/// ```
/// Nor can a runner be dereferenced into a replaceable session:
/// ```compile_fail
/// use onnx_runtime_session::{ExecutionScope, InferenceSession};
/// fn replaceable(scope: &mut ExecutionScope<'_>) {
///     let _: &mut InferenceSession = &mut **scope;
/// }
/// ```
#[must_use = "call finish() to observe completion/return errors before ending the active phase"]
pub struct ExecutionScope<'a> {
    session: Option<&'a mut InferenceSession>,
}

impl<'a> ExecutionScope<'a> {
    pub(crate) fn new(session: &'a mut InferenceSession) -> Self {
        Self {
            session: Some(session),
        }
    }

    fn session(&mut self) -> &mut InferenceSession {
        self.session
            .as_deref_mut()
            .expect("the exclusive session borrow lives until consuming finish")
    }

    pub fn prepare_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<onnx_runtime_ep_api::WorkspaceRequirement> {
        self.session()
            .prepare_with_device_bindings(inputs, bindings)
    }

    pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        self.session().run(inputs)
    }

    pub fn run_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceBindingOutputs> {
        self.session().run_with_device_bindings(inputs, bindings)
    }

    /// Settle completion, then return the owner unless an independent capture
    /// pin still needs its address. A failed boundary retains ownership and
    /// disables reuse. Provider return errors consume the owner once.
    pub fn finish(mut self) -> Result<()> {
        self.session
            .take()
            .expect("finish consumes the active borrow")
            .exec
            .finish_execution_scope()
    }
}

impl Drop for ExecutionScope<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take()
            && let Err(error) = session.exec.finish_execution_scope()
        {
            eprintln!("[onnx-runtime-session] active execution scope cleanup failed: {error}");
        }
    }
}
