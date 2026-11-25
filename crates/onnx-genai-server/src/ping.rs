//! Validated llama-server SSE idle-ping policy.
use axum::response::{
    IntoResponse, Response, Sse,
    sse::{Event, KeepAlive},
};
use std::{convert::Infallible, num::NonZeroU32, time::Duration};

/// Idle interval in seconds. Zero and -1 disable comment pings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsePingInterval(Option<NonZeroU32>);
impl Default for SsePingInterval {
    fn default() -> Self {
        Self(NonZeroU32::new(30))
    }
}
impl std::str::FromStr for SsePingInterval {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value=value.parse::<i32>().map_err(|_| "sse_ping_interval must be integer seconds from -1 through 2147483647; use -1 or 0 to disable pings".to_string())?;
        if value < -1 {
            return Err("sse_ping_interval must be -1, 0, or positive integer seconds; use -1 to disable pings".into());
        }
        Ok(Self(NonZeroU32::new(value.max(0) as u32)))
    }
}
impl SsePingInterval {
    pub(crate) fn seconds(self) -> i32 {
        self.0.map_or(-1, |seconds| seconds.get() as i32)
    }
    pub(crate) fn parse(
        value: &serde_json::Value,
    ) -> Result<Option<Self>, crate::routes::ApiError> {
        crate::types::parse_llama_integer(value, "sse_ping_interval", i64::from(i32::MAX))
            .map(|number| number.map(|seconds| Self(NonZeroU32::new(seconds.max(0) as u32))))
            .map_err(crate::routes::ApiError::invalid_request)
    }
    pub(crate) fn resolve(
        request: &crate::types::ChatCompletionRequest,
        default: Self,
    ) -> Result<Self, crate::routes::ApiError> {
        match request.extensions.get("sse_ping_interval") {
            Some(value) => Ok(Self::parse(value)?.unwrap_or(default)),
            None => Ok(default),
        }
    }
    pub(crate) fn response(
        self,
        stream: impl tokio_stream::Stream<Item = Result<Event, Infallible>> + Send + 'static,
    ) -> Response {
        match self.0 {
            Some(seconds) => Sse::new(stream)
                .keep_alive(
                    KeepAlive::new()
                        .interval(Duration::from_secs(u64::from(seconds.get())))
                        .text(""),
                )
                .into_response(),
            None => Sse::new(stream).into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio_stream::{StreamExt, wrappers::ReceiverStream};
    #[test]
    fn request_ping_policy_inherits_and_overrides_validated_seconds() {
        let default = SsePingInterval::default();
        assert_eq!(default, "30".parse().unwrap());
        let mut request: crate::types::ChatCompletionRequest =
            serde_json::from_value(json!({"model":"test","messages":[]})).unwrap();
        assert_eq!(
            SsePingInterval::resolve(&request, default).unwrap(),
            default
        );
        for seconds in [-1, 0, 1, 2147483647] {
            request
                .extensions
                .insert("sse_ping_interval".into(), json!(seconds));
            assert_eq!(
                SsePingInterval::resolve(&request, default).unwrap(),
                seconds.to_string().parse().unwrap()
            );
        }
        assert_eq!(
            "-1".parse::<SsePingInterval>().unwrap(),
            "0".parse().unwrap()
        );
        for value in [
            json!(-2),
            json!(2147483648u64),
            json!(-2.1),
            json!(true),
            json!("1"),
            json!(u64::MAX),
        ] {
            assert!(SsePingInterval::parse(&value).is_err(), "{value}");
        }
        for (value, expected) in [
            (json!(-1.9), -1),
            (json!(0.5), 0),
            (json!(1.5), 1),
            (json!(2147483647.9), i32::MAX),
        ] {
            request.extensions.insert("sse_ping_interval".into(), value);
            assert_eq!(
                SsePingInterval::resolve(&request, default).unwrap(),
                expected.to_string().parse().unwrap()
            );
        }
        request
            .extensions
            .insert("sse_ping_interval".into(), json!(null));
        for configured in ["7", "-1"] {
            let configured = configured.parse().unwrap();
            assert_eq!(
                SsePingInterval::resolve(&request, configured).unwrap(),
                configured
            );
        }
        assert_eq!(SsePingInterval::parse(&json!(null)).unwrap(), None);
    }
    #[tokio::test]
    async fn idle_pings_are_comments_reset_after_data_and_end_with_the_source() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let mut body = "1"
            .parse::<SsePingInterval>()
            .unwrap()
            .response(ReceiverStream::new(rx))
            .into_body()
            .into_data_stream();
        tokio::time::sleep(Duration::from_millis(700)).await;
        tx.send(Ok(Event::default().data("hello"))).await.unwrap();
        let data = body.next().await.unwrap().unwrap();
        assert!(std::str::from_utf8(&data).unwrap().contains("data: hello"));
        assert!(
            tokio::time::timeout(Duration::from_millis(500), body.next())
                .await
                .is_err(),
            "data must reset the idle timer"
        );
        let ping = tokio::time::timeout(Duration::from_secs(4), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(ping.starts_with(b":") && ping.ends_with(b"\n\n"));
        assert!(!std::str::from_utf8(&ping).unwrap().contains("data:"));
        drop(tx);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), body.next())
                .await
                .unwrap()
                .is_none()
        );
    }
    #[tokio::test]
    async fn disabled_pings_wait_for_data_without_changing_the_body() {
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let mut body = "-1"
            .parse::<SsePingInterval>()
            .unwrap()
            .response(ReceiverStream::new(rx))
            .into_body()
            .into_data_stream();
        assert!(
            tokio::time::timeout(Duration::from_millis(100), body.next())
                .await
                .is_err()
        );
        tx.send(Ok(Event::default().data("hello"))).await.unwrap();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: hello\n\n");
        drop(tx);
        assert!(body.next().await.is_none());
    }
}
