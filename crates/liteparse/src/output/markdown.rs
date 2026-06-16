use crate::projection;
use crate::types::{LayoutBlock, ParsedPage, TextItem};

/// Format parsed pages as Markdown.
///
/// When layout blocks are available, their labels drive the Markdown structure.
/// Without layout blocks, fall back to the already reconstructed page text.
pub fn format_markdown(pages: &[ParsedPage]) -> String {
    pages
        .iter()
        .filter_map(format_page_markdown)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn format_page_markdown(page: &ParsedPage) -> Option<String> {
    if page.layout_blocks.is_empty() {
        let text = page.text.trim();
        return (!text.is_empty()).then(|| text.to_string());
    }

    let mut sections = Vec::new();
    for block in &page.layout_blocks {
        if let Some(section) = format_block_markdown(page, block) {
            sections.push(section);
        }
    }

    (!sections.is_empty()).then(|| sections.join("\n\n"))
}

fn format_block_markdown(page: &ParsedPage, block: &LayoutBlock) -> Option<String> {
    let block_items: Vec<TextItem> = page
        .text_items
        .iter()
        .filter(|item| item.layout_block_id == Some(block.id))
        .cloned()
        .collect();

    let text = projection::project_text_items_to_text(
        page.page_number,
        page.page_width,
        page.page_height,
        block_items,
    );
    let text = text.trim();

    match block.label.as_str() {
        "Picture" if text.is_empty() => Some("![Picture](#)".to_string()),
        _ if text.is_empty() => None,
        "Title" => Some(format!("# {}", one_line(text))),
        "SectionHeader" => Some(format!("## {}", one_line(text))),
        "ListItem" => Some(
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        "Formula" => Some(format!("$$\n{}\n$$", text)),
        "Caption" | "Footnote" => Some(format!("_{}_", text)),
        "Picture" => Some(format!("![Picture](#)\n\n{}", text)),
        _ => Some(text.to_string()),
    }
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(text: &str, x: f32, y: f32, block_id: usize) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width: 20.0,
            height: 10.0,
            layout_block_id: Some(block_id),
            ..Default::default()
        }
    }

    fn block(id: usize, label: &str, y: f32) -> LayoutBlock {
        LayoutBlock {
            id,
            label: label.into(),
            confidence: 0.9,
            x: 10.0,
            y,
            width: 100.0,
            height: 20.0,
        }
    }

    fn page(text_items: Vec<TextItem>, layout_blocks: Vec<LayoutBlock>) -> ParsedPage {
        ParsedPage {
            page_number: 1,
            page_width: 612.0,
            page_height: 792.0,
            text: "fallback text".into(),
            text_items,
            layout_blocks,
        }
    }

    #[test]
    fn formats_markdown_from_layout_labels() {
        let page = page(
            vec![
                item("Paper Title", 10.0, 10.0, 0),
                item("Intro", 10.0, 40.0, 1),
                item("first item", 10.0, 70.0, 2),
                item("x = y", 10.0, 100.0, 3),
                item("caption text", 10.0, 130.0, 4),
            ],
            vec![
                block(0, "Title", 10.0),
                block(1, "SectionHeader", 40.0),
                block(2, "ListItem", 70.0),
                block(3, "Formula", 100.0),
                block(4, "Caption", 130.0),
            ],
        );

        let markdown = format_markdown(&[page]);

        assert!(markdown.contains("# Paper Title"));
        assert!(markdown.contains("## Intro"));
        assert!(markdown.contains("- first item"));
        assert!(markdown.contains("$$\nx = y\n$$"));
        assert!(markdown.contains("_caption text_"));
    }

    #[test]
    fn falls_back_to_page_text_without_layout_blocks() {
        let markdown = format_markdown(&[page(vec![], vec![])]);

        assert_eq!(markdown, "fallback text");
    }
}
