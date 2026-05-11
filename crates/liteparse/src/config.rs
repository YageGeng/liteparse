use serde::{Deserialize, Serialize};

/// Configuration for LiteParse document parsing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteParseConfig {
    /// OCR language code (Tesseract format: "eng", "fra", "deu", etc.).
    pub ocr_language: String,
    /// Whether OCR is enabled. When true, runs on text-sparse pages and embedded images.
    pub ocr_enabled: bool,
    /// Path to tessdata directory. Falls back to TESSDATA_PREFIX env var if not set.
    pub tessdata_path: Option<String>,
    /// Maximum number of pages to parse.
    pub max_pages: usize,
    /// Specific pages to parse (e.g., "1-5,10,15-20"). None means all pages.
    pub target_pages: Option<String>,
    /// DPI for rendering pages (used for OCR and screenshots).
    pub dpi: f32,
    /// Output format.
    pub output_format: OutputFormat,
    /// Keep very small text that would normally be filtered out.
    pub preserve_very_small_text: bool,
    /// Password for encrypted/protected documents.
    pub password: Option<String>,
    /// Suppress progress output.
    pub quiet: bool,
}

/// Supported output formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Text,
}

impl Default for LiteParseConfig {
    fn default() -> Self {
        Self {
            ocr_language: "eng".to_string(),
            ocr_enabled: true,
            tessdata_path: None,
            max_pages: 1000,
            target_pages: None,
            dpi: 150.0,
            output_format: OutputFormat::Json,
            preserve_very_small_text: false,
            password: None,
            quiet: false,
        }
    }
}

/// Parse a target pages string like "1-5,10,15-20" into a sorted list of page numbers.
pub fn parse_target_pages(s: &str) -> Result<Vec<u32>, String> {
    let mut pages = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let bounds: Vec<&str> = part.splitn(2, '-').collect();
            let start: u32 = bounds[0]
                .trim()
                .parse()
                .map_err(|_| format!("invalid page number: {}", bounds[0]))?;
            let end: u32 = bounds[1]
                .trim()
                .parse()
                .map_err(|_| format!("invalid page number: {}", bounds[1]))?;
            if start > end {
                return Err(format!("invalid range: {}-{}", start, end));
            }
            for p in start..=end {
                pages.push(p);
            }
        } else {
            let p: u32 = part
                .parse()
                .map_err(|_| format!("invalid page number: {}", part))?;
            pages.push(p);
        }
    }
    pages.sort();
    pages.dedup();
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_target_pages() {
        assert_eq!(parse_target_pages("1-5,10,15-20").unwrap(), vec![1, 2, 3, 4, 5, 10, 15, 16, 17, 18, 19, 20]);
        assert_eq!(parse_target_pages("3").unwrap(), vec![3]);
        assert_eq!(parse_target_pages("1,1,2").unwrap(), vec![1, 2]);
        assert!(parse_target_pages("5-3").is_err());
        assert!(parse_target_pages("abc").is_err());
    }
}
