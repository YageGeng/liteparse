use crate::config::{LiteParseConfig, parse_target_pages};
#[cfg(not(target_arch = "wasm32"))]
use crate::conversion;
use crate::error::LiteParseError;
use crate::extract;
use crate::layout_annotate;
use crate::layout_merge;
#[cfg(any(
    feature = "layout-yolo",
    feature = "layout-yolo-metal",
    feature = "layout-yolo-vulkan",
    feature = "layout-yolo-webgpu"
))]
use crate::layout_yolo::{LayoutDetector, RenderedLayoutPage};
use crate::ocr::OcrEngine;
#[cfg(not(target_arch = "wasm32"))]
use crate::ocr::http_simple::HttpOcrEngine;
#[cfg(feature = "tesseract")]
use crate::ocr::tesseract::TesseractOcrEngine;
use crate::ocr_merge;
use crate::projection;
use crate::render;
use crate::types::{ParsedPage, PdfInput};
use pdfium::Library;

/// Result of parsing a document.
pub struct ParseResult {
    /// Parsed pages with projected text layout.
    pub pages: Vec<ParsedPage>,
    /// Full document text, concatenated from all pages.
    pub text: String,
}

/// Result of rendering a single page screenshot.
#[derive(Debug, Clone)]
pub struct ScreenshotResult {
    pub page_num: u32,
    pub width: u32,
    pub height: u32,
    pub image_bytes: Vec<u8>,
}

/// Main LiteParse orchestrator.
///
/// ### Thread safety
///
/// `LiteParse` is `Send + Sync` and safe to share across threads (e.g.
/// behind an `Arc`, or used concurrently from a multi-threaded `tokio`
/// runtime).
///
/// PDFium itself is **not** thread-safe, so all PDFium FFI work — document
/// loading, page rendering, text extraction — is serialized through a
/// process-global lock held by [`pdfium::Library`]. From a caller's
/// perspective, this means concurrent `parse_*` / `screenshot*` calls are
/// safe but their PDFium portions run sequentially. The OCR pass and grid
/// projection (which dominate runtime for OCR-heavy documents) run outside
/// the lock and remain fully concurrent.
pub struct LiteParse {
    config: LiteParseConfig,
    /// Optional caller-provided OCR engine. When set, this overrides the
    /// built-in selection logic (HTTP OCR / Tesseract). This is the primary
    /// mechanism for plugging an OCR engine in environments without the
    /// built-ins (e.g. WASM, where the JS side supplies a callback engine).
    ocr_engine_override: Option<std::sync::Arc<dyn OcrEngine>>,
}

impl LiteParse {
    pub fn new(config: LiteParseConfig) -> Self {
        Self {
            config,
            ocr_engine_override: None,
        }
    }

    /// Override the OCR engine. When set, the engine is used regardless of
    /// `ocr_server_url` / built-in Tesseract availability.
    pub fn with_ocr_engine(mut self, engine: std::sync::Arc<dyn OcrEngine>) -> Self {
        self.ocr_engine_override = Some(engine);
        self
    }

    /// Parse a document from a file path, returning structured results.
    ///
    /// Non-PDF files are automatically converted to PDF first (requires
    /// LibreOffice/ImageMagick on the system).
    ///
    /// Not available on `wasm32` — the browser has no filesystem. Use
    /// [`LiteParse::parse_input`] with [`PdfInput::Bytes`] instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn parse(&self, input: &str) -> Result<ParseResult, LiteParseError> {
        self.parse_input(PdfInput::Path(input.to_string())).await
    }

    /// Parse a document from either a file path or raw bytes.
    ///
    /// Use `PdfInput::Path` for files on disk or `PdfInput::Bytes` for
    /// in-memory PDF data (e.g. from a network response or Node.js Buffer).
    pub async fn parse_input(&self, input: PdfInput) -> Result<ParseResult, LiteParseError> {
        let log = |msg: &str| {
            if !self.config.quiet {
                eprintln!("{}", msg);
            }
        };

        let t0 = web_time::Instant::now();

        #[cfg(not(target_arch = "wasm32"))]
        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), false).await?;

        #[cfg(target_arch = "wasm32")]
        let validated_input = input;

        // Determine which pages to extract
        let target_pages = self
            .config
            .target_pages
            .as_ref()
            .map(|s| parse_target_pages(s))
            .transpose()
            .map_err(|e| format!("invalid --target-pages: {}", e))?;

        let password = self.config.password.as_deref();
        let page_numbers = {
            let lib = Library::init();
            let document = extract::load_document_from_input(&lib, &validated_input, password)?;
            let page_count = document.page_count() as u32;
            let mut page_numbers: Vec<u32> = (1..=page_count)
                .filter(|page_number| {
                    target_pages
                        .as_ref()
                        .is_none_or(|targets| targets.contains(page_number))
                })
                .take(self.config.max_pages)
                .collect();
            page_numbers.sort_unstable();
            page_numbers
        };

        let ocr_engine: Option<std::sync::Arc<dyn OcrEngine>> = if self.config.ocr_enabled {
            Some(self.resolve_ocr_engine()?)
        } else {
            None
        };

        #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
        let layout_detector = LayoutDetector::from_config_async(&self.config).await?;

        #[cfg(all(
            not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")),
            any(
                feature = "layout-yolo",
                feature = "layout-yolo-metal",
                feature = "layout-yolo-vulkan",
                feature = "layout-yolo-webgpu"
            )
        ))]
        let layout_detector = LayoutDetector::from_config(&self.config)?;

        #[cfg(not(any(
            feature = "layout-yolo",
            feature = "layout-yolo-metal",
            feature = "layout-yolo-vulkan",
            feature = "layout-yolo-webgpu"
        )))]
        if self.config.layout_enabled {
            return Err(LiteParseError::Config(
                "layout detection requires a YOLO layout feature".into(),
            ));
        }

        let mut parsed_pages = Vec::with_capacity(page_numbers.len());

        for page_number in page_numbers {
            let page_start = web_time::Instant::now();
            let page_index = page_number as i32 - 1;

            let (mut page, ocr_rendered, layout_rendered_for_detection) = {
                let lib = Library::init();
                let document = extract::load_document_from_input(&lib, &validated_input, password)?;
                let page = extract::extract_page_from_document(&document, page_index)?;
                let after_extract = web_time::Instant::now();
                log(&format!(
                    "[liteparse] page {} extract: {:.1}ms, items={}",
                    page_number,
                    after_extract.duration_since(page_start).as_secs_f64() * 1000.0,
                    page.text_items.len()
                ));

                let ocr_rendered = if self.config.ocr_enabled {
                    ocr_merge::render_pages_for_ocr(
                        &document,
                        std::slice::from_ref(&page),
                        self.config.dpi,
                    )?
                } else {
                    Vec::new()
                };

                #[cfg(any(
                    feature = "layout-yolo",
                    feature = "layout-yolo-metal",
                    feature = "layout-yolo-vulkan",
                    feature = "layout-yolo-webgpu"
                ))]
                let layout_rendered = if self.config.layout_enabled {
                    if let Some(rendered) = ocr_rendered.first() {
                        Some(RenderedLayoutPage::new(
                            rendered.rgb_bytes.clone(),
                            rendered.width,
                            rendered.height,
                            page.page_width,
                            page.page_height,
                        ))
                    } else {
                        let page_obj = document.page(page_index)?;
                        let bitmap = page_obj.render(self.config.dpi)?;
                        Some(RenderedLayoutPage::new(
                            bitmap.to_rgb(),
                            bitmap.width() as u32,
                            bitmap.height() as u32,
                            page.page_width,
                            page.page_height,
                        ))
                    }
                } else {
                    None
                };

                #[cfg(not(any(
                    feature = "layout-yolo",
                    feature = "layout-yolo-metal",
                    feature = "layout-yolo-vulkan",
                    feature = "layout-yolo-webgpu"
                )))]
                let layout_rendered: Option<()> = None;

                (page, ocr_rendered, layout_rendered)
            };

            let layout_started = web_time::Instant::now();
            #[cfg(not(any(
                feature = "layout-yolo",
                feature = "layout-yolo-metal",
                feature = "layout-yolo-vulkan",
                feature = "layout-yolo-webgpu"
            )))]
            let _ = layout_rendered_for_detection;

            #[cfg(all(
                not(target_arch = "wasm32"),
                any(
                    feature = "layout-yolo",
                    feature = "layout-yolo-metal",
                    feature = "layout-yolo-vulkan",
                    feature = "layout-yolo-webgpu"
                )
            ))]
            let layout_handle = if let (Some(detector), Some(rendered)) = (
                layout_detector.clone(),
                layout_rendered_for_detection.clone(),
            ) {
                let dpi = self.config.dpi;
                Some(tokio::task::spawn_blocking(move || {
                    detector.detect_rendered(rendered, dpi)
                }))
            } else {
                None
            };

            #[cfg(not(any(
                feature = "layout-yolo",
                feature = "layout-yolo-metal",
                feature = "layout-yolo-vulkan",
                feature = "layout-yolo-webgpu"
            )))]
            let layout_handle: Option<
                tokio::task::JoinHandle<Result<Vec<crate::types::LayoutBlock>, LiteParseError>>,
            > = None;

            let ocr_started = web_time::Instant::now();
            if let Some(engine) = ocr_engine.clone() {
                if ocr_rendered.is_empty() {
                    log(&format!(
                        "[liteparse] page {} ocr: 0.0ms, skipped",
                        page_number
                    ));
                } else {
                    let mut one_page = vec![page];
                    ocr_merge::ocr_and_merge_rendered(
                        &mut one_page,
                        ocr_rendered,
                        self.config.dpi,
                        engine,
                        &self.config.ocr_language,
                        self.config.num_workers,
                    )
                    .await?;
                    page = one_page.remove(0);
                    log(&format!(
                        "[liteparse] page {} ocr: {:.1}ms",
                        page_number,
                        web_time::Instant::now()
                            .duration_since(ocr_started)
                            .as_secs_f64()
                            * 1000.0
                    ));
                }
            } else {
                log(&format!(
                    "[liteparse] page {} ocr: 0.0ms, disabled",
                    page_number
                ));
            }

            #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
            let layout_blocks = if let (Some(detector), Some(rendered)) = (
                layout_detector.clone(),
                layout_rendered_for_detection.clone(),
            ) {
                match detector
                    .detect_rendered_async(rendered, self.config.dpi)
                    .await
                {
                    Ok(blocks) => {
                        log(&format!(
                            "[liteparse] page {} layout: {:.1}ms, blocks={}",
                            page_number,
                            web_time::Instant::now()
                                .duration_since(layout_started)
                                .as_secs_f64()
                                * 1000.0,
                            blocks.len()
                        ));
                        blocks
                    }
                    Err(e) => {
                        return Err(LiteParseError::Other(format!(
                            "layout detection failed on page {}: {}",
                            page_number, e
                        )));
                    }
                }
            } else {
                log(&format!(
                    "[liteparse] page {} layout: 0.0ms, {}",
                    page_number,
                    if self.config.layout_enabled {
                        "unavailable"
                    } else {
                        "disabled"
                    }
                ));
                Vec::new()
            };

            #[cfg(all(
                target_arch = "wasm32",
                not(feature = "layout-yolo-webgpu"),
                any(
                    feature = "layout-yolo",
                    feature = "layout-yolo-metal",
                    feature = "layout-yolo-vulkan"
                )
            ))]
            let layout_blocks = if let (Some(detector), Some(rendered)) = (
                layout_detector.clone(),
                layout_rendered_for_detection.clone(),
            ) {
                match detector.detect_rendered(rendered, self.config.dpi) {
                    Ok(blocks) => {
                        log(&format!(
                            "[liteparse] page {} layout: {:.1}ms, blocks={}",
                            page_number,
                            web_time::Instant::now()
                                .duration_since(layout_started)
                                .as_secs_f64()
                                * 1000.0,
                            blocks.len()
                        ));
                        blocks
                    }
                    Err(e) => {
                        return Err(LiteParseError::Other(format!(
                            "layout detection failed on page {}: {}",
                            page_number, e
                        )));
                    }
                }
            } else {
                log(&format!(
                    "[liteparse] page {} layout: 0.0ms, {}",
                    page_number,
                    if self.config.layout_enabled {
                        "unavailable"
                    } else {
                        "disabled"
                    }
                ));
                Vec::new()
            };

            #[cfg(not(all(
                target_arch = "wasm32",
                any(
                    feature = "layout-yolo",
                    feature = "layout-yolo-metal",
                    feature = "layout-yolo-vulkan",
                    feature = "layout-yolo-webgpu"
                )
            )))]
            let layout_blocks = match layout_handle {
                Some(handle) => match handle.await {
                    Ok(Ok(blocks)) => {
                        log(&format!(
                            "[liteparse] page {} layout: {:.1}ms, blocks={}",
                            page_number,
                            web_time::Instant::now()
                                .duration_since(layout_started)
                                .as_secs_f64()
                                * 1000.0,
                            blocks.len()
                        ));
                        blocks
                    }
                    Ok(Err(e)) => {
                        return Err(LiteParseError::Other(format!(
                            "layout detection failed on page {}: {}",
                            page_number, e
                        )));
                    }
                    Err(e) => {
                        return Err(LiteParseError::Other(format!(
                            "layout detection task failed on page {}: {}",
                            page_number, e
                        )));
                    }
                },
                None => {
                    log(&format!(
                        "[liteparse] page {} layout: 0.0ms, {}",
                        page_number,
                        if self.config.layout_enabled {
                            "unavailable"
                        } else {
                            "disabled"
                        }
                    ));
                    Vec::new()
                }
            };

            let mut parsed_page = projection::project_pages_to_grid(vec![page])
                .into_iter()
                .next()
                .ok_or_else(|| LiteParseError::Other("page projection produced no page".into()))?;
            parsed_page.layout_blocks = layout_blocks;
            layout_merge::assign_text_items_to_layout_blocks(
                &mut parsed_page.text_items,
                &parsed_page.layout_blocks,
            );
            layout_merge::compact_layout_blocks(
                &mut parsed_page.layout_blocks,
                &mut parsed_page.text_items,
            );
            // The initial page text is produced before layout assignment, so it
            // only knows about item-level geometry. Once layout blocks are
            // available, rebuild the plain text from those blocks so the
            // detected reading order affects user-facing text output too.
            if let Some(layout_text) = parsed_page.rebuild_text_in_layout_order() {
                parsed_page.text = layout_text;
            }
            let assigned = parsed_page
                .text_items
                .iter()
                .filter(|item| item.layout_block_id.is_some())
                .count();
            log(&format!(
                "[liteparse] page {} merge: text_items={}, assigned={}",
                page_number,
                parsed_page.text_items.len(),
                assigned
            ));
            log(&format!(
                "[liteparse] page {} total: {:.1}ms",
                page_number,
                web_time::Instant::now()
                    .duration_since(page_start)
                    .as_secs_f64()
                    * 1000.0
            ));

            parsed_pages.push(parsed_page);
        }

        let t2 = web_time::Instant::now();

        let full_text = parsed_pages
            .iter()
            .map(|p| p.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let total = t2.duration_since(t0).as_secs_f64() * 1000.0;
        log(&format!("[liteparse] total: {:.1}ms", total));

        Ok(ParseResult {
            pages: parsed_pages,
            text: full_text,
        })
    }

    fn resolve_ocr_engine(&self) -> Result<std::sync::Arc<dyn OcrEngine>, LiteParseError> {
        if let Some(e) = self.ocr_engine_override.clone() {
            return Ok(e);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref url) = self.config.ocr_server_url {
                Ok(std::sync::Arc::new(HttpOcrEngine::new(url.clone())))
            } else {
                #[cfg(feature = "tesseract")]
                {
                    Ok(std::sync::Arc::new(TesseractOcrEngine::new(
                        self.config.tessdata_path.clone(),
                    )))
                }
                #[cfg(not(feature = "tesseract"))]
                {
                    Err("OCR enabled but no --ocr-server-url provided and tesseract feature is disabled".into())
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(
                "OCR enabled but no `ocrEngine` callback was provided (WASM builds have no built-in OCR engine)"
                    .into(),
            )
        }
    }

    /// Generate screenshots of document pages as PNG bytes.
    ///
    /// Non-PDF files are automatically converted to PDF first (requires
    /// LibreOffice/ImageMagick on the system). Plain-text formats cannot be
    /// rendered and return a clear error.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn screenshot(
        &self,
        input: &str,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        self.screenshot_input(PdfInput::Path(input.to_string()), page_numbers)
            .await
    }

    /// Generate screenshots from a file path or raw bytes.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn screenshot_input(
        &self,
        input: PdfInput,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        let log = |msg: &str| {
            if !self.config.quiet {
                eprintln!("{}", msg);
            }
        };

        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), true).await?;

        if let PdfInput::Path(ref path) = validated_input
            && !conversion::is_pdf(path)
        {
            log("[liteparse] converted input to PDF for screenshot rendering");
        }

        let rendered = render::render_pages_to_png(
            &validated_input,
            page_numbers.as_deref(),
            self.config.dpi,
            self.config.password.as_deref(),
        )?;

        Ok(rendered
            .into_iter()
            .map(|page| ScreenshotResult {
                page_num: page.page_num,
                width: page.width,
                height: page.height,
                image_bytes: page.png_bytes,
            })
            .collect())
    }

    /// Generate page screenshots with detected layout blocks drawn on top.
    ///
    /// This method enables layout detection for the parse pass regardless of
    /// the parser's current `layout_enabled` setting. It returns plain page
    /// screenshots if the detector produces no blocks.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn layout_screenshot(
        &self,
        input: &str,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        self.layout_screenshot_input(PdfInput::Path(input.to_string()), page_numbers)
            .await
    }

    /// Generate annotated layout screenshots from a file path or raw PDF bytes.
    pub async fn layout_screenshot_input(
        &self,
        input: PdfInput,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        #[cfg(not(target_arch = "wasm32"))]
        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), true).await?;

        #[cfg(target_arch = "wasm32")]
        let validated_input = input;

        let effective_page_numbers = match page_numbers {
            Some(nums) => Some(nums),
            None => self
                .config
                .target_pages
                .as_ref()
                .map(|s| parse_target_pages(s))
                .transpose()
                .map_err(|e| format!("invalid target_pages: {}", e))?,
        };

        let mut parse_config = self.config.clone();
        parse_config.layout_enabled = true;
        if let Some(nums) = &effective_page_numbers {
            parse_config.target_pages = Some(
                nums.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            parse_config.max_pages = parse_config.max_pages.max(nums.len());
        }

        let parser = LiteParse {
            config: parse_config,
            ocr_engine_override: self.ocr_engine_override.clone(),
        };
        let parsed = parser.parse_input(validated_input.clone()).await?;
        let mut rendered = render::render_pages_to_rgba(
            &validated_input,
            effective_page_numbers.as_deref(),
            self.config.dpi,
            self.config.password.as_deref(),
        )?;

        rendered
            .iter_mut()
            .map(|page| {
                let parsed_page = parsed
                    .pages
                    .iter()
                    .find(|parsed_page| parsed_page.page_number as u32 == page.page_num);
                let blocks = parsed_page
                    .map(|parsed_page| parsed_page.layout_blocks.as_slice())
                    .unwrap_or(&[]);
                let (page_width, page_height) = parsed_page
                    .map(|parsed_page| (parsed_page.page_width, parsed_page.page_height))
                    .unwrap_or((page.width as f32, page.height as f32));
                let image_bytes = layout_annotate::annotate_layout_png(
                    &mut page.rgba_bytes,
                    page.width,
                    page.height,
                    page_width,
                    page_height,
                    blocks,
                )?;
                Ok(ScreenshotResult {
                    page_num: page.page_num,
                    width: page.width,
                    height: page.height,
                    image_bytes,
                })
            })
            .collect()
    }

    pub fn config(&self) -> &LiteParseConfig {
        &self.config
    }
}

impl ParsedPage {
    /// Rebuild page text by walking layout blocks in their reading order.
    ///
    /// Layout detection provides coarse document regions, which is a better
    /// ordering signal for multi-column pages than sorting all text items by
    /// page y/x. Each block is still rendered through the normal projection
    /// pipeline so local line grouping, spacing, and table-like alignment keep
    /// the existing behavior. Text that could not be assigned to any block is
    /// appended at the end instead of being dropped.
    fn rebuild_text_in_layout_order(&self) -> Option<String> {
        if self.layout_blocks.is_empty() {
            return None;
        }

        let mut sections = Vec::new();

        for block in &self.layout_blocks {
            // Re-project each block's items independently. This keeps the
            // block-level order from XY-cut while preserving the mature
            // item-level layout reconstruction inside the block.
            let block_items: Vec<_> = self
                .text_items
                .iter()
                .filter(|item| item.layout_block_id == Some(block.id))
                .cloned()
                .collect();
            let text = projection::project_text_items_to_text(
                self.page_number,
                self.page_width,
                self.page_height,
                block_items,
            );
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_string());
            }
        }

        // Layout detectors can miss headers, footers, marginalia, or low
        // confidence regions. Keep that text after the ordered blocks so the
        // parser remains lossless even when layout assignment is imperfect.
        let unassigned_items: Vec<_> = self
            .text_items
            .iter()
            .filter(|item| item.layout_block_id.is_none())
            .cloned()
            .collect();
        let unassigned_text = projection::project_text_items_to_text(
            self.page_number,
            self.page_width,
            self.page_height,
            unassigned_items,
        );
        let unassigned_trimmed = unassigned_text.trim();
        if !unassigned_trimmed.is_empty() {
            sections.push(unassigned_trimmed.to_string());
        }

        if sections.is_empty() {
            None
        } else {
            Some(sections.join("\n\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LayoutBlock, ParsedPage, TextItem};

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_new_stores_config() {
        let mut cfg = LiteParseConfig::default();
        cfg.ocr_enabled = false;
        cfg.max_pages = 7;
        let lp = LiteParse::new(cfg);
        assert!(!lp.config().ocr_enabled);
        assert_eq!(lp.config().max_pages, 7);
    }

    fn text_item(text: &str, x: f32, y: f32, layout_block_id: Option<usize>) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width: 80.0,
            height: 12.0,
            layout_block_id,
            ..Default::default()
        }
    }

    fn block(id: usize, label: &str, x: f32, y: f32, width: f32, height: f32) -> LayoutBlock {
        LayoutBlock {
            id,
            label: label.into(),
            confidence: 0.9,
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn rebuilds_page_text_in_layout_block_order() {
        let page = ParsedPage {
            page_number: 1,
            page_width: 600.0,
            page_height: 800.0,
            text: "left-1 right-1\nleft-2 right-2".into(),
            text_items: vec![
                text_item("left-1", 40.0, 40.0, Some(0)),
                text_item("right-1", 320.0, 40.0, Some(2)),
                text_item("left-2", 40.0, 100.0, Some(1)),
                text_item("right-2", 320.0, 100.0, Some(3)),
            ],
            layout_blocks: vec![
                block(0, "left-1", 40.0, 40.0, 180.0, 30.0),
                block(1, "left-2", 40.0, 100.0, 180.0, 30.0),
                block(2, "right-1", 320.0, 40.0, 180.0, 30.0),
                block(3, "right-2", 320.0, 100.0, 180.0, 30.0),
            ],
        };

        assert_eq!(
            page.rebuild_text_in_layout_order().as_deref(),
            Some("left-1\n\nleft-2\n\nright-1\n\nright-2")
        );
    }
}
