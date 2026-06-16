use crate::projection;
use crate::types::{LayoutBlock, ParsedPage, TextItem};
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

#[derive(Debug, Serialize)]
pub(crate) struct JsonLayoutBlock {
    pub id: usize,
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub text_items: Vec<JsonTextItem>,
}

#[derive(Debug, Serialize)]
pub(crate) struct JsonPage {
    pub page: usize,
    pub width: f32,
    pub height: f32,
    pub text: String,
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
                layout_blocks: page
                    .layout_blocks
                    .iter()
                    .map(|block| layout_block_to_json(page, block))
                    .collect(),
            })
            .collect(),
    }
}

fn layout_block_to_json(page: &ParsedPage, block: &LayoutBlock) -> JsonLayoutBlock {
    let block_items: Vec<TextItem> = page
        .text_items
        .iter()
        .filter(|item| item.layout_block_id == Some(block.id))
        .cloned()
        .collect();

    let text_items: Vec<JsonTextItem> = block_items.iter().map(text_item_to_json).collect();
    let text = projection::project_text_items_to_text(
        page.page_number,
        page.page_width,
        page.page_height,
        block_items,
    )
    .trim()
    .to_string();

    JsonLayoutBlock {
        id: block.id,
        label: block.label.clone(),
        confidence: block.confidence,
        x: block.x,
        y: block.y,
        width: block.width,
        height: block.height,
        text,
        text_items,
    }
}

fn text_item_to_json(item: &TextItem) -> JsonTextItem {
    JsonTextItem {
        text: item.text.clone(),
        x: item.x,
        y: item.y,
        width: item.width,
        height: item.height,
        font_name: item.font_name.clone(),
        font_size: item.font_size,
        confidence: item.confidence.or(Some(1.0)),
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

    fn page(items: Vec<TextItem>) -> ParsedPage {
        ParsedPage {
            page_number: 1,
            page_width: 612.0,
            page_height: 792.0,
            text: "txt".into(),
            text_items: items,
            layout_blocks: vec![],
        }
    }

    fn layout_block() -> LayoutBlock {
        LayoutBlock {
            id: 2,
            label: "table".into(),
            confidence: 0.875,
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 50.0,
        }
    }

    #[test]
    fn test_build_json_native_text_defaults_confidence_to_one() {
        let j = build_json(&[page(vec![item("hi", None)])]);
        assert_eq!(j.pages.len(), 1);
        assert_eq!(j.pages[0].page, 1);
        assert!(j.pages[0].layout_blocks.is_empty());
    }

    #[test]
    fn test_build_json_preserves_ocr_confidence() {
        let mut text_item = item("hi", Some(0.42));
        text_item.layout_block_id = Some(2);
        text_item.layout_label = Some("table".into());
        let mut page = page(vec![text_item]);
        page.layout_blocks = vec![layout_block()];

        let j = build_json(&[page]);

        assert_eq!(
            j.pages[0].layout_blocks[0].text_items[0].confidence,
            Some(0.42)
        );
    }

    #[test]
    fn test_format_json_pretty() {
        let s = format_json(&[page(vec![item("hi", None)])]).unwrap();
        assert!(s.contains("\n"));
        assert!(s.contains("\"text\": \"txt\""));
        assert!(s.contains("\"page\": 1"));
        assert!(!s.contains("\"text_items\": ["));
    }

    #[test]
    fn test_build_json_empty() {
        let j = build_json(&[]);
        assert!(j.pages.is_empty());
    }

    #[test]
    fn test_build_json_includes_layout_fields() {
        let mut text_item = item("hi", None);
        text_item.layout_block_id = Some(2);
        text_item.layout_label = Some("table".into());

        let mut page = page(vec![text_item]);
        page.layout_blocks = vec![layout_block()];

        let s = format_json(&[page]).unwrap();
        assert!(s.contains("\"layout_blocks\""));
        assert!(s.contains("\"label\": \"table\""));
        assert!(s.contains("\"text_items\""));
        assert!(s.contains("\"text\": \"hi\""));
        assert!(!s.contains("\"layout_block_id\""));
        assert!(!s.contains("\"layout_label\""));
    }

    #[test]
    fn test_build_json_nests_text_items_under_matching_layout_blocks() {
        let mut first = item("first", None);
        first.layout_block_id = Some(2);
        first.layout_label = Some("table".into());
        let mut second = item("second", Some(0.5));
        second.layout_block_id = Some(2);
        second.layout_label = Some("table".into());
        let mut unassigned = item("outside", None);
        unassigned.layout_block_id = None;

        let mut page = page(vec![first, second, unassigned]);
        page.layout_blocks = vec![layout_block()];

        let j = build_json(&[page]);

        assert_eq!(j.pages[0].layout_blocks[0].text_items.len(), 2);
        assert_eq!(j.pages[0].layout_blocks[0].text_items[0].text, "first");
        assert_eq!(
            j.pages[0].layout_blocks[0].text_items[1].confidence,
            Some(0.5)
        );
        let s = serde_json::to_string(&j.pages[0]).unwrap();
        assert!(!s.contains("\"text_items\":[{\"text\":\"outside\""));
    }

    #[test]
    fn test_build_json_projects_layout_block_text() {
        let mut first = item("hello", None);
        first.x = 10.0;
        first.y = 20.0;
        first.width = 20.0;
        first.height = 10.0;
        first.layout_block_id = Some(2);

        let mut second = item("world", None);
        second.x = 38.0;
        second.y = 20.0;
        second.width = 22.0;
        second.height = 10.0;
        second.layout_block_id = Some(2);

        let mut page = page(vec![first, second]);
        page.layout_blocks = vec![layout_block()];

        let j = build_json(&[page]);

        assert_eq!(j.pages[0].layout_blocks[0].text, "hello world");
    }
}
