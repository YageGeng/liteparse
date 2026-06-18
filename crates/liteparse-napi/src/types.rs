use napi_derive::napi;

use liteparse::config::{ImageMode, LiteParseConfig, OutputFormat};
use liteparse::layout_merge;
use liteparse::parser::ParseResult;
use liteparse::types::{GraphicPrimitive, LayoutBlock, Page, ParsedPage, Rect, TextItem};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[napi(object)]
#[derive(Clone)]
pub struct JsLiteParseConfig {
    /// OCR language code (e.g., "eng", "fra").
    pub ocr_language: Option<String>,
    /// Whether OCR is enabled.
    pub ocr_enabled: Option<bool>,
    /// HTTP OCR server URL. If set, uses HTTP OCR instead of Tesseract.
    pub ocr_server_url: Option<String>,
    /// Path to tessdata directory for Tesseract.
    pub tessdata_path: Option<String>,
    /// Maximum number of pages to parse.
    pub max_pages: Option<u32>,
    /// Specific pages to parse (e.g., "1-5,10,15-20").
    pub target_pages: Option<String>,
    /// DPI for rendering pages (used for OCR and screenshots).
    pub dpi: Option<f64>,
    /// Output format: "json", "text", or "markdown".
    pub output_format: Option<String>,
    /// Keep very small text that would normally be filtered out.
    pub preserve_very_small_text: Option<bool>,
    /// Password for encrypted/protected documents.
    pub password: Option<String>,
    /// Suppress progress output.
    pub quiet: Option<bool>,
    /// Number of concurrent OCR workers (default: CPU cores - 1).
    pub num_workers: Option<u32>,
    /// How to surface raster images in markdown output: "off", "placeholder"
    /// (default — emits `![](image_pN_K.png)` references with no bytes), or
    /// "embed" (also returns each image's PNG bytes on `images`).
    pub image_mode: Option<String>,
    /// Render hyperlink annotations as `[text](url)` in markdown output
    /// (default true). Set false for plain anchor text.
    pub extract_links: Option<bool>,
    /// Whether YOLO document layout detection is enabled.
    pub layout_enabled: Option<bool>,
    /// Minimum layout detection confidence score.
    pub layout_confidence_threshold: Option<f64>,
    /// IoU threshold used by layout detection NMS.
    pub layout_iou_threshold: Option<f64>,
    /// Square image size used for layout detection.
    pub layout_image_size: Option<u32>,
}

impl JsLiteParseConfig {
    pub fn into_rust(self) -> LiteParseConfig {
        let mut cfg = LiteParseConfig::default();
        if let Some(v) = self.ocr_language {
            cfg.ocr_language = v;
        }
        if let Some(v) = self.ocr_enabled {
            cfg.ocr_enabled = v;
        }
        if let Some(v) = self.ocr_server_url {
            cfg.ocr_server_url = Some(v);
        }
        if let Some(v) = self.tessdata_path {
            cfg.tessdata_path = Some(v);
        }
        if let Some(v) = self.max_pages {
            cfg.max_pages = v as usize;
        }
        if let Some(v) = self.target_pages {
            cfg.target_pages = Some(v);
        }
        if let Some(v) = self.dpi {
            cfg.dpi = v as f32;
        }
        if let Some(v) = self.output_format {
            cfg.output_format = match v.as_str() {
                "text" => OutputFormat::Text,
                "markdown" | "md" => OutputFormat::Markdown,
                _ => OutputFormat::Json,
            };
        }
        if let Some(v) = self.preserve_very_small_text {
            cfg.preserve_very_small_text = v;
        }
        if let Some(v) = self.password {
            cfg.password = Some(v);
        }
        if let Some(v) = self.quiet {
            cfg.quiet = v;
        }
        if let Some(v) = self.num_workers {
            cfg.num_workers = v as usize;
        }
        if let Some(v) = self.image_mode {
            cfg.image_mode = match v.as_str() {
                "off" | "none" => ImageMode::Off,
                "embed" => ImageMode::Embed,
                _ => ImageMode::Placeholder,
            };
        }
        if let Some(v) = self.extract_links {
            cfg.extract_links = v;
        }
        if let Some(v) = self.layout_enabled {
            cfg.layout_enabled = v;
        }
        if let Some(v) = self.layout_confidence_threshold {
            cfg.layout_confidence_threshold = v as f32;
        }
        if let Some(v) = self.layout_iou_threshold {
            cfg.layout_iou_threshold = v as f32;
        }
        if let Some(v) = self.layout_image_size {
            cfg.layout_image_size = v;
        }
        cfg
    }

    pub fn from_rust(cfg: &LiteParseConfig) -> Self {
        Self {
            ocr_language: Some(cfg.ocr_language.clone()),
            ocr_enabled: Some(cfg.ocr_enabled),
            ocr_server_url: cfg.ocr_server_url.clone(),
            tessdata_path: cfg.tessdata_path.clone(),
            max_pages: Some(cfg.max_pages as u32),
            target_pages: cfg.target_pages.clone(),
            dpi: Some(cfg.dpi as f64),
            output_format: Some(match cfg.output_format {
                OutputFormat::Json => "json".to_string(),
                OutputFormat::Text => "text".to_string(),
                OutputFormat::Markdown => "markdown".to_string(),
            }),
            preserve_very_small_text: Some(cfg.preserve_very_small_text),
            password: cfg.password.clone(),
            quiet: Some(cfg.quiet),
            num_workers: Some(cfg.num_workers as u32),
            image_mode: Some(match cfg.image_mode {
                ImageMode::Off => "off".to_string(),
                ImageMode::Placeholder => "placeholder".to_string(),
                ImageMode::Embed => "embed".to_string(),
            }),
            extract_links: Some(cfg.extract_links),
            layout_enabled: Some(cfg.layout_enabled),
            layout_confidence_threshold: Some(cfg.layout_confidence_threshold as f64),
            layout_iou_threshold: Some(cfg.layout_iou_threshold as f64),
            layout_image_size: Some(cfg.layout_image_size),
        }
    }
}

// ---------------------------------------------------------------------------
// TextItem
// ---------------------------------------------------------------------------

#[napi(object)]
#[derive(Clone)]
pub struct JsTextItem {
    pub text: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub font_name: Option<String>,
    pub font_size: Option<f64>,
    pub confidence: Option<f64>,
    /// Rotation in degrees (viewport space). Defaults to 0 when omitted.
    pub rotation: Option<f64>,
    pub layout_block_id: Option<u32>,
    pub layout_label: Option<String>,
}

impl JsTextItem {
    pub fn to_rust(&self) -> TextItem {
        TextItem {
            text: self.text.clone(),
            x: self.x as f32,
            y: self.y as f32,
            width: self.width as f32,
            height: self.height as f32,
            rotation: self.rotation.unwrap_or(0.0) as f32,
            font_name: self.font_name.clone(),
            font_size: self.font_size.map(|v| v as f32),
            confidence: self.confidence.map(|v| v as f32),
            layout_block_id: self.layout_block_id.map(|v| v as usize),
            layout_label: self.layout_label.clone(),
            ..Default::default()
        }
    }

    pub fn from_rust(item: &TextItem) -> Self {
        Self {
            text: item.text.clone(),
            x: item.x as f64,
            y: item.y as f64,
            width: item.width as f64,
            height: item.height as f64,
            rotation: Some(item.rotation as f64),
            font_name: item.font_name.clone(),
            font_size: item.font_size.map(|v| v as f64),
            confidence: item.confidence.map(|v| v as f64).or(Some(1.0)),
            layout_block_id: item.layout_block_id.map(|v| v as u32),
            layout_label: item.layout_label.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Graphic primitive (pre-extracted vector graphics)
// ---------------------------------------------------------------------------

/// A vector-graphic primitive supplied by an external extractor. `kind` selects
/// the variant: `"stroke"` (uses `x1/y1/x2/y2`) or `"rect"` (uses
/// `x/y/width/height`). Coordinates are viewport space (top-left origin, 72
/// DPI), matching the text items. `has_fill`/`has_stroke` carry the paint
/// intent even when no color is known, so ruled-table edge detection still
/// treats a colorless stroked rect as stroked.
#[napi(object)]
#[derive(Clone)]
pub struct JsGraphic {
    /// "stroke" or "rect". Anything else is dropped.
    pub kind: String,
    // Stroke endpoints (used when kind == "stroke").
    pub x1: Option<f64>,
    pub y1: Option<f64>,
    pub x2: Option<f64>,
    pub y2: Option<f64>,
    // Rect bbox top-left + size (used when kind == "rect").
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    /// Whether the path is filled. Drives Rect `fill` presence.
    pub has_fill: Option<bool>,
    /// Whether the path is stroked. Drives Rect `stroke` presence.
    pub has_stroke: Option<bool>,
    /// Fill color as ARGB hex (e.g. "ff000000"). May be absent even when filled.
    pub fill_color: Option<String>,
    /// Stroke color as ARGB hex. May be absent even when stroked.
    pub stroke_color: Option<String>,
    /// Stroke line width in points.
    pub line_width: Option<f64>,
}

impl JsGraphic {
    pub fn to_rust(&self) -> Option<GraphicPrimitive> {
        match self.kind.as_str() {
            "stroke" => Some(GraphicPrimitive::Stroke {
                x1: self.x1.unwrap_or(0.0) as f32,
                y1: self.y1.unwrap_or(0.0) as f32,
                x2: self.x2.unwrap_or(0.0) as f32,
                y2: self.y2.unwrap_or(0.0) as f32,
                color: self.stroke_color.clone(),
                width: self.line_width.unwrap_or(0.0) as f32,
            }),
            "rect" => Some(GraphicPrimitive::Rect {
                bbox: Rect {
                    x: self.x.unwrap_or(0.0) as f32,
                    y: self.y.unwrap_or(0.0) as f32,
                    width: self.width.unwrap_or(0.0) as f32,
                    height: self.height.unwrap_or(0.0) as f32,
                },
                fill: if self.has_fill.unwrap_or(false) {
                    Some(self.fill_color.clone().unwrap_or_default())
                } else {
                    None
                },
                stroke: if self.has_stroke.unwrap_or(false) {
                    Some(self.stroke_color.clone().unwrap_or_default())
                } else {
                    None
                },
            }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Page input (pre-extracted)
// ---------------------------------------------------------------------------

/// A page of pre-extracted text supplied by an external extractor. Coordinates
/// are viewport space (top-left origin, 72 DPI). `graphics` enables ruled-table
/// and horizontal-rule detection; struct nodes are still unsupported on this
/// path, so tagged-heading detection remains unavailable until they are added.
#[napi(object)]
#[derive(Clone)]
pub struct JsPageInput {
    pub page_number: u32,
    pub page_width: f64,
    pub page_height: f64,
    pub text_items: Vec<JsTextItem>,
    pub graphics: Option<Vec<JsGraphic>>,
}

impl JsPageInput {
    pub fn to_rust(&self) -> Page {
        Page {
            page_number: self.page_number as usize,
            page_width: self.page_width as f32,
            page_height: self.page_height as f32,
            text_items: self.text_items.iter().map(JsTextItem::to_rust).collect(),
            graphics: self
                .graphics
                .as_ref()
                .map(|gs| gs.iter().filter_map(JsGraphic::to_rust).collect())
                .unwrap_or_default(),
            struct_nodes: Vec::new(),
            image_refs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ParsedPage
// ---------------------------------------------------------------------------

#[napi(object)]
#[derive(Clone)]
pub struct JsLayoutBlock {
    pub id: u32,
    pub label: String,
    pub confidence: f64,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub text: String,
}

impl JsLayoutBlock {
    pub fn from_rust(block: &LayoutBlock, page: &ParsedPage) -> Self {
        let text = layout_merge::project_layout_text(
            page.page_number,
            page.page_width,
            page.page_height,
            page.text_items
                .iter()
                .filter(|item| item.layout_block_id == Some(block.id))
                .cloned()
                .collect(),
        )
        .trim()
        .to_string();

        Self {
            id: block.id as u32,
            label: block.label.clone(),
            confidence: block.confidence as f64,
            x: block.x as f64,
            y: block.y as f64,
            width: block.width as f64,
            height: block.height as f64,
            text,
        }
    }
}

#[napi(object)]
#[derive(Clone)]
pub struct JsParsedPage {
    pub page_num: u32,
    pub width: f64,
    pub height: f64,
    pub text: String,
    pub text_items: Vec<JsTextItem>,
    pub layout_blocks: Vec<JsLayoutBlock>,
}

impl JsParsedPage {
    pub fn from_rust(page: &ParsedPage) -> Self {
        Self {
            page_num: page.page_number as u32,
            width: page.page_width as f64,
            height: page.page_height as f64,
            text: page.text.clone(),
            text_items: page.text_items.iter().map(JsTextItem::from_rust).collect(),
            layout_blocks: page
                .layout_blocks
                .iter()
                .map(|block| JsLayoutBlock::from_rust(block, page))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// ParseResult
// ---------------------------------------------------------------------------

#[napi(object)]
#[derive(Clone)]
pub struct JsParseResult {
    pub pages: Vec<JsParsedPage>,
    pub text: String,
    pub images: Vec<JsExtractedImage>,
}

#[napi(object)]
#[derive(Clone)]
pub struct JsExtractedImage {
    pub id: String,
    pub page: u32,
    pub format: String,
    pub bytes: napi::bindgen_prelude::Buffer,
}

// ---------------------------------------------------------------------------
// ScreenshotResult
// ---------------------------------------------------------------------------

#[napi(object)]
#[derive(Clone)]
pub struct JsScreenshotResult {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub image_buffer: napi::bindgen_prelude::Buffer,
}

impl JsParseResult {
    pub fn from_rust(result: &ParseResult, _config: &LiteParseConfig) -> Self {
        Self {
            pages: result.pages.iter().map(JsParsedPage::from_rust).collect(),
            text: result.text.clone(),
            images: result
                .images
                .iter()
                .map(|img| JsExtractedImage {
                    id: img.id.clone(),
                    page: img.page,
                    format: img.format.clone(),
                    bytes: img.bytes.clone().into(),
                })
                .collect(),
        }
    }
}
