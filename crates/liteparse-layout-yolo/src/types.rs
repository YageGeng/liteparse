use crate::labels::LayoutLabel;
use serde::Serialize;

/// Borrowed page image input and corresponding PDF page dimensions.
#[derive(Debug, Clone, Copy)]
pub struct PageImage<'a> {
    /// Tightly packed RGB pixels in row-major order.
    pub rgb: &'a [u8],
    /// Rendered image width in pixels.
    pub width: u32,
    /// Rendered image height in pixels.
    pub height: u32,
    /// PDF page width in points.
    pub page_width: f32,
    /// PDF page height in points.
    pub page_height: f32,
    /// Render DPI used to map pixels back into page coordinates.
    pub dpi: f32,
}

/// Page-space layout detection returned by the YOLO detector.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayoutDetection {
    /// Typed layout class predicted by the YOLO model.
    pub label: LayoutLabel,
    /// Model confidence score for the predicted label.
    pub confidence: f32,
    /// Left position in PDF page coordinates.
    pub x: f32,
    /// Top position in PDF page coordinates.
    pub y: f32,
    /// Detection width in PDF page coordinates.
    pub width: f32,
    /// Detection height in PDF page coordinates.
    pub height: f32,
}
