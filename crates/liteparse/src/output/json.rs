use crate::types::{LayoutBlock, ParsedPage};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct JsonTextItem {
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

/// JSON representation of one detected layout block.
#[derive(Debug, Serialize)]
pub(crate) struct JsonLayoutBlock {
    /// Stable page-local block id.
    pub id: usize,
    /// Public layout class label.
    pub label: String,
    /// Model confidence score for the block.
    pub confidence: f32,
    /// Left position in page coordinates.
    pub x: f32,
    /// Top position in page coordinates.
    pub y: f32,
    /// Block width in page coordinates.
    pub width: f32,
    /// Block height in page coordinates.
    pub height: f32,
}

#[derive(Debug, Serialize)]
pub(crate) struct JsonPage {
    pub page: usize,
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub text_items: Vec<JsonTextItem>,
    /// Layout blocks detected on this page, kept separate from text items.
    pub layout_blocks: Vec<JsonLayoutBlock>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ParseResultJson {
    pub pages: Vec<JsonPage>,
}

/// Build structured JSON output from parsed pages.
pub(crate) fn build_json(pages: &[ParsedPage]) -> ParseResultJson {
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
                // CONTEXT: Keep layout blocks as a sibling array so existing
                // JSON consumers can continue reading `text_items` unchanged.
                layout_blocks: page
                    .layout_blocks
                    .iter()
                    .map(JsonLayoutBlock::from_layout_block)
                    .collect(),
            })
            .collect(),
    }
}

impl JsonLayoutBlock {
    /// Create the JSON representation for a detected layout block.
    fn from_layout_block(block: &LayoutBlock) -> Self {
        Self {
            id: block.id,
            label: block.label.clone(),
            confidence: block.confidence,
            x: block.x,
            y: block.y,
            width: block.width,
            height: block.height,
        }
    }
}

/// Format parsed pages as pretty-printed JSON string.
pub fn format_json(pages: &[ParsedPage]) -> Result<String, serde_json::Error> {
    let result = build_json(pages);
    serde_json::to_string_pretty(&result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LayoutBlock, ParsedPage, TextItem};

    // Build a compact text item for JSON output tests.
    fn item(text: &str, conf: Option<f32>) -> TextItem {
        TextItem {
            text: text.into(),
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            font_name: Some("Helv".into()),
            font_size: Some(10.0),
            confidence: conf,
            ..Default::default()
        }
    }

    // Build a parsed page with no layout blocks by default.
    fn page(items: Vec<TextItem>) -> ParsedPage {
        ParsedPage {
            page_number: 1,
            page_width: 612.0,
            page_height: 792.0,
            text: "txt".into(),
            text_items: items,
            layout_blocks: vec![],
            projected_lines: vec![],
            regions: crate::types::Region::default(),
            graphics: vec![],
            figures: vec![],
            struct_nodes: vec![],
            image_refs: vec![],
        }
    }

    // Verifies native text gets the historical JSON confidence default.
    #[test]
    fn test_build_json_native_text_defaults_confidence_to_one() {
        let j = build_json(&[page(vec![item("hi", None)])]);
        assert_eq!(j.pages.len(), 1);
        assert_eq!(j.pages[0].page, 1);
        assert_eq!(j.pages[0].text_items[0].confidence, Some(1.0));
        assert_eq!(j.pages[0].text_items[0].font_name.as_deref(), Some("Helv"));
        assert!(j.pages[0].layout_blocks.is_empty());
    }

    // Verifies OCR confidence is preserved instead of replaced by the default.
    #[test]
    fn test_build_json_preserves_ocr_confidence() {
        let j = build_json(&[page(vec![item("hi", Some(0.42))])]);
        assert_eq!(j.pages[0].text_items[0].confidence, Some(0.42));
    }

    // Verifies JSON formatting remains pretty-printed.
    #[test]
    fn test_format_json_pretty() {
        let s = format_json(&[page(vec![item("hi", None)])]).unwrap();
        assert!(s.contains("\n"));
        assert!(s.contains("\"text\": \"hi\""));
        assert!(s.contains("\"page\": 1"));
    }

    // Verifies empty page input serializes to an empty page array.
    #[test]
    fn test_build_json_empty() {
        let j = build_json(&[]);
        assert!(j.pages.is_empty());
    }

    // Verifies layout blocks are additive and do not replace text items.
    #[test]
    fn test_build_json_keeps_text_items_when_layout_blocks_exist() {
        let mut page = page(vec![item("hi", None)]);
        page.layout_blocks = vec![LayoutBlock {
            id: 0,
            label: "Text".into(),
            confidence: 0.9,
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        }];

        let j = build_json(&[page]);

        assert_eq!(j.pages[0].text_items[0].text, "hi");
        assert_eq!(j.pages[0].layout_blocks[0].label, "Text");
    }
}
