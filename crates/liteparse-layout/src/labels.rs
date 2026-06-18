use serde::Serialize;
use std::str::FromStr;

/// YOLO document layout labels in model class-id order.
pub const LAYOUT_LABELS: [LayoutLabel; 11] = [
    LayoutLabel::Caption,
    LayoutLabel::Footnote,
    LayoutLabel::Formula,
    LayoutLabel::ListItem,
    LayoutLabel::PageFooter,
    LayoutLabel::PageHeader,
    LayoutLabel::Picture,
    LayoutLabel::SectionHeader,
    LayoutLabel::Table,
    LayoutLabel::Text,
    LayoutLabel::Title,
];

/// Strongly typed YOLO document layout label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum LayoutLabel {
    /// Text describing another page element.
    Caption,
    /// Footnote text near the page bottom.
    Footnote,
    /// Mathematical formula region.
    Formula,
    /// Bullet, numbered, or otherwise list-like item.
    ListItem,
    /// Repeated footer region.
    PageFooter,
    /// Repeated header region.
    PageHeader,
    /// Figure, image, chart, or other picture region.
    Picture,
    /// Section heading below the document title level.
    SectionHeader,
    /// Tabular region.
    Table,
    /// Main body text region.
    Text,
    /// Document or page title region.
    Title,
}

impl LayoutLabel {
    /// Return the number of layout classes emitted by the YOLO model.
    pub const fn class_count() -> usize {
        LAYOUT_LABELS.len()
    }

    /// Return the stable string representation used by the YOLO model.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Caption => "Caption",
            Self::Footnote => "Footnote",
            Self::Formula => "Formula",
            Self::ListItem => "ListItem",
            Self::PageFooter => "PageFooter",
            Self::PageHeader => "PageHeader",
            Self::Picture => "Picture",
            Self::SectionHeader => "SectionHeader",
            Self::Table => "Table",
            Self::Text => "Text",
            Self::Title => "Title",
        }
    }

    /// Return the deterministic debug RGBA color for this layout label.
    pub fn debug_color_rgba(self) -> [u8; 4] {
        match self {
            Self::Caption => [0x2A, 0x9D, 0x8F, 255],
            Self::Footnote => [0xF4, 0x43, 0x36, 255],
            Self::Formula => [0xAB, 0x47, 0xBC, 255],
            Self::ListItem => [0x03, 0xA9, 0xF4, 255],
            Self::PageFooter => [0x8D, 0x6E, 0x63, 255],
            Self::PageHeader => [0xFF, 0x8F, 0x00, 255],
            Self::Picture => [0x9E, 0x9E, 0x9E, 255],
            Self::SectionHeader => [0x8E, 0x24, 0xAA, 255],
            Self::Table => [0x00, 0x96, 0x88, 255],
            Self::Text => [0x43, 0xA0, 0x47, 255],
            Self::Title => [0x7C, 0x4D, 0xFF, 255],
        }
    }
}

impl TryFrom<usize> for LayoutLabel {
    type Error = LayoutLabelError;

    /// Convert a model class id into a typed layout label.
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        LAYOUT_LABELS
            .get(value)
            .copied()
            .ok_or(LayoutLabelError::UnknownClassId(value))
    }
}

impl TryFrom<&str> for LayoutLabel {
    type Error = LayoutLabelError;

    /// Convert a model label string into a typed layout label.
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

impl FromStr for LayoutLabel {
    type Err = LayoutLabelError;

    /// Parse a public PascalCase layout label into a typed layout label.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        LAYOUT_LABELS
            .iter()
            .copied()
            .find(|label| label.as_str() == value)
            .ok_or_else(|| LayoutLabelError::UnknownLabel(value.to_string()))
    }
}

impl From<LayoutLabel> for String {
    /// Convert a typed layout label into the public string representation.
    fn from(value: LayoutLabel) -> Self {
        value.as_str().to_string()
    }
}

impl std::fmt::Display for LayoutLabel {
    /// Format the label with the public YOLO string representation.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when converting an unknown value into a layout label.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayoutLabelError {
    /// Class id is outside the model label table.
    #[error("unknown YOLO layout class id: {0}")]
    UnknownClassId(usize),
    /// Label string does not match one of the public PascalCase names.
    #[error("unknown YOLO layout label: {0}")]
    UnknownLabel(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies that model class ids map to the expected typed labels.
    #[test]
    fn layout_label_converts_from_model_class_id() {
        assert_eq!(LayoutLabel::try_from(9), Ok(LayoutLabel::Text));
    }

    // Verifies that public label strings can be converted without parsing ambiguity.
    #[test]
    fn layout_label_converts_from_public_string() {
        assert_eq!(
            LayoutLabel::try_from("SectionHeader"),
            Ok(LayoutLabel::SectionHeader)
        );
    }

    // Verifies that FromStr accepts the public PascalCase label names.
    #[test]
    fn layout_label_parses_from_public_string() {
        assert_eq!("PageHeader".parse(), Ok(LayoutLabel::PageHeader));
    }

    // Verifies the debug color table for a representative class.
    #[test]
    fn layout_label_debug_color_returns_known_text_color() {
        assert_eq!(
            LayoutLabel::Text.debug_color_rgba(),
            [0x43, 0xA0, 0x47, 255]
        );
    }

    // Verifies that labels convert into their public string names.
    #[test]
    fn layout_label_converts_into_public_string() {
        assert_eq!(String::from(LayoutLabel::PageFooter), "PageFooter");
    }

    // Verifies that serde keeps the public PascalCase variant spelling.
    #[test]
    fn layout_label_serializes_as_pascal_case_variant_name() {
        assert_eq!(
            serde_json::to_string(&LayoutLabel::ListItem).unwrap(),
            "\"ListItem\""
        );
    }
}
