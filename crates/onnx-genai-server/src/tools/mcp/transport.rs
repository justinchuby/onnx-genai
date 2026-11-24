//! SDK transport adapter: SDK request/notification work retains canonical host credit.
use super::*;
use rmcp::{
    RoleClient,
    model::JsonRpcMessage,
    service::{RxJsonRpcMessage, TxJsonRpcMessage},
    transport::Transport,
};

// The SDK may retain notifications in handler tasks. A lease attached to each
// message's extensions follows that task instead of counting only wire bytes.
// Responses are separately covered by the actor's one outstanding RPC staging.
pub(super) type Fault = Arc<std::sync::Mutex<Option<anyhow::Error>>>;
pub(super) struct CreditedTransport<T> {
    inner: T,
    memory: LedgerGovernor,
    frame_bytes: usize,
    fault: Fault,
    connection: Arc<actor::ConnectionCredit>,
}
struct MessageResources {
    _frame: MemoryLease,
    _connection: Arc<actor::ConnectionCredit>,
}
impl<T> CreditedTransport<T> {
    pub fn new(
        inner: T,
        memory: LedgerGovernor,
        frame_bytes: usize,
        fault: Fault,
        connection: Arc<actor::ConnectionCredit>,
    ) -> Self {
        Self {
            inner,
            memory,
            frame_bytes,
            fault,
            connection,
        }
    }
}
impl<T> Transport<RoleClient> for CreditedTransport<T>
where
    T: Transport<RoleClient> + Send + 'static,
{
    type Error = T::Error;
    fn send(
        &mut self,
        item: TxJsonRpcMessage<RoleClient>,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send + 'static {
        let sending = self.inner.send(item);
        let connection = self.connection.clone();
        async move {
            let _connection = connection;
            sending.await
        }
    }
    async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
        // Yield before another ready pipe read so completed notification handlers
        // can release credit; an unsolicited burst cannot allocate without it.
        tokio::task::yield_now().await;
        let bytes = self.frame_bytes.checked_mul(32)?.checked_add(64 * 1024)?;
        let credit = match self.memory.reserve(
            Tier::Host,
            u64::try_from(bytes).ok()?,
            MemoryRole::Activation,
            HolderId::new(0),
        ) {
            Ok(credit) => credit,
            Err(error) => {
                tracing::warn!(%error, "MCP incoming message exceeds available host credit; closing transport; increase --tools-max-host-bytes or reduce MCP output/concurrency");
                *self.fault.lock().unwrap_or_else(|p|p.into_inner())=Some(anyhow::Error::new(error).context("MCP incoming message exceeded host credit; increase --tools-max-host-bytes or reduce MCP output/concurrency"));
                return None;
            }
        };
        let mut message = self.inner.receive().await?;
        if matches!(
            message,
            JsonRpcMessage::Request(_) | JsonRpcMessage::Notification(_)
        ) {
            message.insert_extension(Arc::new(MessageResources {
                _frame: credit,
                _connection: self.connection.clone(),
            }));
        }
        Some(message)
    }
    fn close(&mut self) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        self.inner.close()
    }
}
