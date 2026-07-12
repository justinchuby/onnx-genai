//! ORT error handling.

#[derive(Debug, thiserror::Error)]
pub enum OrtError {
    #[error("ORT error: {message} (code: {code})")]
    Runtime { code: i32, message: String },
    #[error("Null pointer returned from ORT API")]
    NullPointer,
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Session creation failed: {0}")]
    SessionCreation(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OrtError>;
