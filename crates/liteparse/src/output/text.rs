use crate::types::ParsedPage;

/// Format parsed pages as plain text with page headers.
pub fn format_text(pages: &[ParsedPage]) -> String {
    pages
        .iter()
        .map(|page| format!("\n--- Page {} ---\n{}", page.page_number, page.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}
