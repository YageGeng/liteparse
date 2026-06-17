use std::{io::Cursor, pin::Pin, time::Duration};

use image::ImageFormat;
use reqwest::{
    Client,
    multipart::{Form, Part},
};
use serde::{Deserialize, Serialize};

use crate::ocr::{OcrEngine, OcrOptions, OcrResult};

#[derive(Debug, Serialize, Deserialize)]
pub struct HttpOcrResponseItem {
    text: String,
    bbox: [f32; 4],
    confidence: f32,
    /// Optional 4-point polygon [[x,y]×4] of the (possibly rotated) detection,
    /// ordered top-left → top-right → bottom-right → bottom-left in the
    /// glyphs' upright reading frame.
    #[serde(default)]
    polygon: Option<[[f32; 2]; 4]>,
}

impl HttpOcrResponseItem {
    fn into_ocr_result(self) -> OcrResult {
        OcrResult {
            text: self.text,
            bbox: self.bbox,
            confidence: self.confidence,
            polygon: self.polygon,
        }
    }
}

/// A single detection from the LlamaParse prod OCR worker, which emits
/// EasyOCR/PaddleOCR-style positional tuples: `[polygon, text, confidence]`,
/// where `polygon` is a list of `[x, y]` points (4 for both engines) ordered
/// top-left → top-right → bottom-right → bottom-left.
#[derive(Debug, Deserialize)]
struct ProdOcrItem(Vec<[f32; 2]>, String, f32);

impl ProdOcrItem {
    fn into_ocr_result(self) -> OcrResult {
        let ProdOcrItem(poly, text, confidence) = self;
        // Axis-aligned bbox from the polygon's min/max extents.
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for [x, y] in &poly {
            min_x = min_x.min(*x);
            min_y = min_y.min(*y);
            max_x = max_x.max(*x);
            max_y = max_y.max(*y);
        }
        // Forward the raw polygon only when it's exactly 4 points, so the
        // projector can recover rotation for sideways text.
        let polygon = match poly.as_slice() {
            [a, b, c, d] => Some([*a, *b, *c, *d]),
            _ => None,
        };
        OcrResult {
            text,
            bbox: [min_x, min_y, max_x, max_y],
            confidence,
            polygon,
        }
    }
}

/// Accepts either the LiteParse standard response or the LlamaParse prod OCR
/// worker response. Untagged: serde tries `Standard` first (keyed on
/// `results` with object items), then falls back to `Prod` (keyed on `result`
/// with positional-tuple items).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum HttpOcrResponse {
    Standard { results: Vec<HttpOcrResponseItem> },
    Prod { result: Vec<ProdOcrItem> },
}

impl HttpOcrResponse {
    fn into_results(self) -> Vec<OcrResult> {
        match self {
            HttpOcrResponse::Standard { results } => {
                results.into_iter().map(|i| i.into_ocr_result()).collect()
            }
            HttpOcrResponse::Prod { result } => {
                result.into_iter().map(|i| i.into_ocr_result()).collect()
            }
        }
    }
}

/// HTTP-based OCR engine that conforms to LiteParse OCR API specification.
/// The server must implement the API defined in OCR_API_SPEC.md:
///     - POST /ocr endpoint
///     - Accepts multipart/form-data with 'file' and 'language' fields
///     - Returns JSON: { results: [{ text, bbox: [x1,y1,x2,y2], confidence }] }
/// See ocr/easyocr/ and ocr/paddleocr/ for example server implementations.
pub struct HttpOcrEngine {
    pub name: String,
    server_url: String,
    /// Extra headers (name, value) sent with every request, e.g. auth tokens.
    headers: Vec<(String, String)>,
}

impl HttpOcrEngine {
    pub fn new(server_url: String) -> Self {
        Self::with_headers(server_url, Vec::new())
    }

    pub fn with_headers(server_url: String, headers: Vec<(String, String)>) -> Self {
        Self {
            name: "http-ocr".to_string(),
            server_url,
            headers,
        }
    }
}

impl OcrEngine for HttpOcrEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn recognize<'a, 'b: 'a, 'c: 'a>(
        &'a self,
        image_data: &'c [u8],
        width: u32,
        height: u32,
        options: &'b OcrOptions,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<Vec<OcrResult>, Box<dyn std::error::Error + Send + Sync>>>
                + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            // Encode raw RGB bytes as PNG for the server
            let img: image::RgbImage =
                image::ImageBuffer::from_raw(width, height, image_data.to_vec())
                    .ok_or("failed to create image buffer from raw RGB data")?;
            let mut png_bytes = Vec::new();
            img.write_to(&mut Cursor::new(&mut png_bytes), ImageFormat::Png)?;

            let client = Client::new();
            let form = Form::new()
                .part(
                    "file",
                    Part::bytes(png_bytes)
                        .file_name("image.png")
                        .mime_str("image/png")?,
                )
                .text("language", options.language.clone());
            let mut request = client
                .post(&self.server_url)
                .multipart(form)
                .timeout(Duration::from_millis(60000));
            for (name, value) in &self.headers {
                request = request.header(name.as_str(), value.as_str());
            }
            let raw = request.send().await?.error_for_status()?.text().await?;
            // Parse from the buffered body (rather than `.json()`) so a
            // malformed/unexpected response can surface a snippet of what the
            // server actually returned.
            let response: HttpOcrResponse = serde_json::from_str(&raw).map_err(|e| {
                let snippet: String = raw.chars().take(200).collect();
                format!("OCR server returned unparseable response: {e}; body starts: {snippet}")
            })?;
            let results = response.into_results();
            if std::env::var("LITEPARSE_DEBUG_OCR").is_ok() {
                eprintln!(
                    "[ocr-http] {} bytes -> {} result(s)",
                    raw.len(),
                    results.len()
                );
            }
            Ok(results)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_sets_name_and_url() {
        let e = HttpOcrEngine::new("http://example.com/ocr".into());
        assert_eq!(e.name(), "http-ocr");
        assert_eq!(e.server_url, "http://example.com/ocr");
    }

    #[test]
    fn test_response_deserializes() {
        let raw = r#"{"results":[{"text":"hi","bbox":[1.0,2.0,3.0,4.0],"confidence":0.85}]}"#;
        let parsed: HttpOcrResponse = serde_json::from_str(raw).unwrap();
        let results = parsed.into_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hi");
        assert_eq!(results[0].bbox, [1.0, 2.0, 3.0, 4.0]);
        assert!((results[0].confidence - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_response_deserializes_empty() {
        let raw = r#"{"results":[]}"#;
        let parsed: HttpOcrResponse = serde_json::from_str(raw).unwrap();
        assert!(parsed.into_results().is_empty());
    }

    #[test]
    fn test_prod_response_deserializes() {
        // LlamaParse prod worker shape: `result` tuples + `document_angle`.
        let raw = r#"{"document_angle":-90,"result":[[[[10.0,20.0],[60.0,20.0],[60.0,40.0],[10.0,40.0]],"hi",0.85]]}"#;
        let parsed: HttpOcrResponse = serde_json::from_str(raw).unwrap();
        let results = parsed.into_results();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "hi");
        // bbox is the polygon's min/max extents.
        assert_eq!(results[0].bbox, [10.0, 20.0, 60.0, 40.0]);
        assert!((results[0].confidence - 0.85).abs() < 1e-6);
        // 4-point polygon forwarded as-is (TL → TR → BR → BL).
        assert_eq!(
            results[0].polygon,
            Some([[10.0, 20.0], [60.0, 20.0], [60.0, 40.0], [10.0, 40.0]])
        );
    }

    #[test]
    fn test_prod_response_empty() {
        let raw = r#"{"document_angle":null,"result":[]}"#;
        let parsed: HttpOcrResponse = serde_json::from_str(raw).unwrap();
        assert!(parsed.into_results().is_empty());
    }

    #[tokio::test]
    async fn test_recognize_network_error() {
        let e = HttpOcrEngine::new("http://127.0.0.1:1/ocr".into());
        let opts = OcrOptions {
            language: "eng".into(),
            dpi: 150.0,
        };
        let r = e.recognize(&[0u8; 4], 1, 1, &opts).await;
        assert!(r.is_err());
    }
}
