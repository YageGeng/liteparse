pub mod tesseract;

/// A single word-level OCR result with bounding box and confidence.
#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    /// Bounding box in pixel coordinates: [x1, y1, x2, y2] (left, top, right, bottom).
    pub bbox: [f32; 4],
    /// Confidence score in 0.0–1.0 range.
    pub confidence: f32,
}

pub struct OcrOptions {
    pub language: String,
}

pub trait OcrEngine {
    fn name(&self) -> &str;
    fn recognize(
        &self,
        image_data: &[u8],
        width: u32,
        height: u32,
        options: &OcrOptions,
    ) -> Result<Vec<OcrResult>, Box<dyn std::error::Error>>;
}
