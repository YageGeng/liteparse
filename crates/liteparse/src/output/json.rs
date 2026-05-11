use crate::types::ParsedPage;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct JsonTextItem {
    pub text: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_size: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct JsonPage {
    pub page: usize,
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub text_items: Vec<JsonTextItem>,
}

#[derive(Debug, Serialize)]
pub struct ParseResultJson {
    pub pages: Vec<JsonPage>,
}

/// Build structured JSON output from parsed pages.
pub fn build_json(pages: &[ParsedPage]) -> ParseResultJson {
    ParseResultJson {
        pages: pages
            .iter()
            .map(|page| JsonPage {
                page: page.page_number,
                width: page.page_width,
                height: page.page_height,
                text: page.text.clone(),
                text_items: page
                    .text_items
                    .iter()
                    .map(|item| JsonTextItem {
                        text: item.text.clone(),
                        x: item.x,
                        y: item.y,
                        width: item.width,
                        height: item.height,
                        font_name: item.font_name.clone(),
                        font_size: item.font_size,
                        confidence: item.confidence.or(Some(1.0)),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Format parsed pages as pretty-printed JSON string.
pub fn format_json(pages: &[ParsedPage]) -> Result<String, serde_json::Error> {
    let result = build_json(pages);
    serde_json::to_string_pretty(&result)
}
