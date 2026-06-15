use thiserror::Error;

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("invalid RGB buffer length: expected {expected}, got {actual}")]
    InvalidImageBuffer { expected: usize, actual: usize },
    #[error("layout model is not available; export the ONNX model and rebuild with model assets")]
    ModelUnavailable,
    #[error("unsupported YOLO layout image size: expected {expected}, got {actual}")]
    UnsupportedImageSize { expected: u32, actual: u32 },
    #[error("invalid YOLO output tensor: {0}")]
    InvalidModelOutput(String),
}
