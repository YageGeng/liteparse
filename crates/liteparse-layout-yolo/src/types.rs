use serde::Serialize;

#[derive(Debug, Clone, Copy)]
pub struct PageImage<'a> {
    pub rgb: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub dpi: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayoutDetection {
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
