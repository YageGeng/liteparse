use crate::config::{LiteParseConfig, parse_target_pages};
#[cfg(not(target_arch = "wasm32"))]
use crate::conversion;
use crate::error::LiteParseError;
use crate::extract;
#[cfg(not(target_arch = "wasm32"))]
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
use crate::output::markdown;
use crate::projection;
#[cfg(not(target_arch = "wasm32"))]
use crate::render;
use crate::types::{ExtractedImage, LayoutBlock, OutlineTarget, Page, ParsedPage, PdfInput};
use pdfium::{Document, Library};

#[cfg(not(any(
    feature = "layout-yolo",
    feature = "layout-yolo-metal",
    feature = "layout-yolo-vulkan",
    feature = "layout-yolo-webgpu"
)))]
struct LayoutDetector;

/// Result of parsing a document.
pub struct ParseResult {
    /// Parsed pages with projected text layout.
    pub pages: Vec<ParsedPage>,
    /// Full document text, concatenated from all pages.
    pub text: String,
    /// Document outline (bookmarks) when present. Used by the markdown
    /// emitter as a high-priority heading source on untagged PDFs.
    pub outline: Vec<OutlineTarget>,
    /// Raster images extracted from the document. Empty unless the parser
    /// was configured with `ImageMode::Embed`. Each entry carries the same
    /// `id` the markdown emitter referenced in `![](image_{id}.png)`, so the
    /// caller can match them up without parsing markdown.
    pub images: Vec<ExtractedImage>,
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

        // Extract text (and pre-render OCR pages in one PDF load when OCR is on).
        // The PDFium lock is acquired for this entire critical section and
        // released before any `.await` below — OCR (network / CPU) and grid
        // projection (pure Rust) do not touch PDFium, so they can run
        // concurrently with other `LiteParse` calls.
        let password = self.config.password.as_deref();
        let render_images = matches!(self.config.image_mode, crate::config::ImageMode::Embed);
        let layout_detector = self.create_layout_detector().await?;
        let (pages, ocr_rendered, layout_rendered, outline, images) = {
            let lib = Library::init();
            let document = extract::load_document_from_input(&lib, &validated_input, password)?;
            let outline = extract::extract_outline(&document);
            let (pages, images) = extract::extract_pages_and_images(
                &document,
                target_pages.as_deref(),
                self.config.max_pages,
                render_images,
                self.config.extract_links
                    && self.config.output_format == crate::config::OutputFormat::Markdown,
            )?;
            let t_extract = web_time::Instant::now();
            log(&format!(
                "[liteparse] extract: {:.1}ms ({} pages)",
                t_extract.duration_since(t0).as_secs_f64() * 1000.0,
                pages.len()
            ));
            let rendered = if self.config.ocr_enabled {
                let r = ocr_merge::render_pages_for_ocr(&document, &pages, self.config.dpi)?;
                log(&format!(
                    "[liteparse] ocr render: {:.1}ms ({} pages)",
                    web_time::Instant::now()
                        .duration_since(t_extract)
                        .as_secs_f64()
                        * 1000.0,
                    r.len()
                ));
                r
            } else {
                Vec::new()
            };
            let layout_rendered = self.render_pages_for_layout(
                &document,
                &pages,
                &rendered,
                layout_detector.is_some(),
            )?;
            // `lib` is dropped here, releasing the PDFium lock.
            (pages, rendered, layout_rendered, outline, images)
        };
        let mut pages = pages;
        let t1 = web_time::Instant::now();

        // OCR pass
        if self.config.ocr_enabled {
            let engine: std::sync::Arc<dyn OcrEngine> = if let Some(e) =
                self.ocr_engine_override.clone()
            {
                e
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    if let Some(ref url) = self.config.ocr_server_url {
                        std::sync::Arc::new(HttpOcrEngine::with_headers(
                            url.clone(),
                            self.config.ocr_server_headers.clone(),
                        ))
                    } else {
                        #[cfg(feature = "tesseract")]
                        {
                            std::sync::Arc::new(TesseractOcrEngine::new(
                                self.config.tessdata_path.clone(),
                            ))
                        }
                        #[cfg(not(feature = "tesseract"))]
                        {
                            return Err("OCR enabled but no --ocr-server-url provided and tesseract feature is disabled".into());
                        }
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    return Err(
                        "OCR enabled but no `ocrEngine` callback was provided (WASM builds have no built-in OCR engine)".into(),
                    );
                }
            };
            ocr_merge::ocr_and_merge_rendered(
                &mut pages,
                ocr_rendered,
                self.config.dpi,
                engine,
                &self.config.ocr_language,
                self.config.num_workers,
            )
            .await?;
        }
        let t_ocr = web_time::Instant::now();
        log(&format!(
            "[liteparse] ocr: {:.1}ms",
            t_ocr.duration_since(t1).as_secs_f64() * 1000.0
        ));

        let layout_blocks = self
            .detect_layout_blocks(layout_detector, layout_rendered)
            .await?;
        let t_layout = web_time::Instant::now();
        log(&format!(
            "[liteparse] layout: {:.1}ms",
            t_layout.duration_since(t_ocr).as_secs_f64() * 1000.0
        ));

        // Grid projection
        let mut parsed_pages = projection::project_pages_to_grid(pages);
        attach_layout_blocks(&mut parsed_pages, layout_blocks);
        let t2 = web_time::Instant::now();
        log(&format!(
            "[liteparse] project: {:.1}ms",
            t2.duration_since(t_layout).as_secs_f64() * 1000.0
        ));

        let full_text = if self.config.output_format == crate::config::OutputFormat::Markdown {
            let md = markdown::format_markdown(&parsed_pages, &outline, self.config.image_mode);
            let t3 = web_time::Instant::now();
            log(&format!(
                "[liteparse] markdown: {:.1}ms",
                t3.duration_since(t2).as_secs_f64() * 1000.0
            ));
            md
        } else {
            parsed_pages
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        let total = web_time::Instant::now().duration_since(t0).as_secs_f64() * 1000.0;
        log(&format!("[liteparse] total: {:.1}ms", total));

        Ok(ParseResult {
            pages: parsed_pages,
            text: full_text,
            outline,
            images,
        })
    }

    /// Parse from pre-extracted pages, skipping PDFium text extraction.
    ///
    /// The caller supplies `Page`s already populated with text items (and,
    /// optionally, graphics / struct nodes / image refs) in viewport space
    /// (top-left origin, 72 DPI). This runs only grid projection and the
    /// configured output formatter, so it touches neither PDFium nor OCR and
    /// is fully synchronous. Used when an external extractor (e.g. with its
    /// own font-recovery pipeline) owns text extraction.
    pub fn parse_from_pages(&self, pages: Vec<Page>, outline: Vec<OutlineTarget>) -> ParseResult {
        let parsed_pages = projection::project_pages_to_grid(pages);

        let full_text = if self.config.output_format == crate::config::OutputFormat::Markdown {
            markdown::format_markdown(&parsed_pages, &outline, self.config.image_mode)
        } else {
            parsed_pages
                .iter()
                .map(|p| p.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        ParseResult {
            pages: parsed_pages,
            text: full_text,
            outline,
            images: Vec::new(),
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
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn layout_screenshot_input(
        &self,
        input: PdfInput,
        page_numbers: Option<Vec<u32>>,
    ) -> Result<Vec<ScreenshotResult>, LiteParseError> {
        let (validated_input, _guard) =
            conversion::resolve_pdf_input(input, self.config.password.as_deref(), true).await?;

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

    async fn create_layout_detector(&self) -> Result<Option<LayoutDetector>, LiteParseError> {
        #[cfg(not(any(
            feature = "layout-yolo",
            feature = "layout-yolo-metal",
            feature = "layout-yolo-vulkan",
            feature = "layout-yolo-webgpu"
        )))]
        {
            if self.config.layout_enabled {
                return Err(LiteParseError::Config(
                    "layout detection requires a YOLO layout feature".into(),
                ));
            }
            Ok(None)
        }

        #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
        {
            LayoutDetector::from_config_async(&self.config).await
        }

        #[cfg(all(
            not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")),
            any(
                feature = "layout-yolo",
                feature = "layout-yolo-metal",
                feature = "layout-yolo-vulkan",
                feature = "layout-yolo-webgpu"
            )
        ))]
        {
            LayoutDetector::from_config(&self.config)
        }
    }

    #[cfg(any(
        feature = "layout-yolo",
        feature = "layout-yolo-metal",
        feature = "layout-yolo-vulkan",
        feature = "layout-yolo-webgpu"
    ))]
    fn render_pages_for_layout(
        &self,
        document: &Document,
        pages: &[Page],
        ocr_rendered: &[ocr_merge::RenderedPage],
        enabled: bool,
    ) -> Result<Vec<(usize, RenderedLayoutPage)>, LiteParseError> {
        if !enabled {
            return Ok(Vec::new());
        }

        let mut rendered = Vec::with_capacity(pages.len());
        for (idx, page) in pages.iter().enumerate() {
            if let Some(existing) = ocr_rendered.iter().find(|r| r.idx == idx) {
                rendered.push((
                    idx,
                    RenderedLayoutPage::new(
                        existing.rgb_bytes.clone(),
                        existing.width,
                        existing.height,
                        page.page_width,
                        page.page_height,
                    ),
                ));
                continue;
            }

            let page_obj = document.page((page.page_number - 1) as i32)?;
            let bitmap = page_obj.render(self.config.dpi)?;
            rendered.push((
                idx,
                RenderedLayoutPage::new(
                    bitmap.to_rgb(),
                    bitmap.width() as u32,
                    bitmap.height() as u32,
                    page.page_width,
                    page.page_height,
                ),
            ));
        }
        Ok(rendered)
    }

    #[cfg(not(any(
        feature = "layout-yolo",
        feature = "layout-yolo-metal",
        feature = "layout-yolo-vulkan",
        feature = "layout-yolo-webgpu"
    )))]
    fn render_pages_for_layout(
        &self,
        _document: &Document,
        _pages: &[Page],
        _ocr_rendered: &[ocr_merge::RenderedPage],
        _enabled: bool,
    ) -> Result<Vec<(usize, ())>, LiteParseError> {
        Ok(Vec::new())
    }

    #[cfg(any(
        feature = "layout-yolo",
        feature = "layout-yolo-metal",
        feature = "layout-yolo-vulkan",
        feature = "layout-yolo-webgpu"
    ))]
    async fn detect_layout_blocks(
        &self,
        detector: Option<LayoutDetector>,
        rendered: Vec<(usize, RenderedLayoutPage)>,
    ) -> Result<Vec<(usize, Vec<LayoutBlock>)>, LiteParseError> {
        let Some(detector) = detector else {
            return Ok(Vec::new());
        };

        let mut by_page = Vec::with_capacity(rendered.len());
        for (idx, page) in rendered {
            #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
            let blocks = detector
                .detect_rendered_async(page, self.config.dpi)
                .await?;

            #[cfg(not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")))]
            let blocks = detector.detect_rendered(page, self.config.dpi)?;

            by_page.push((idx, blocks));
        }
        Ok(by_page)
    }

    #[cfg(not(any(
        feature = "layout-yolo",
        feature = "layout-yolo-metal",
        feature = "layout-yolo-vulkan",
        feature = "layout-yolo-webgpu"
    )))]
    async fn detect_layout_blocks(
        &self,
        _detector: Option<LayoutDetector>,
        _rendered: Vec<(usize, ())>,
    ) -> Result<Vec<(usize, Vec<LayoutBlock>)>, LiteParseError> {
        Ok(Vec::new())
    }
}

fn attach_layout_blocks(
    parsed_pages: &mut [ParsedPage],
    layout_blocks: Vec<(usize, Vec<LayoutBlock>)>,
) {
    for (idx, blocks) in layout_blocks {
        let Some(page) = parsed_pages.get_mut(idx) else {
            continue;
        };
        page.layout_blocks = blocks;
        layout_merge::assign_text_items_to_layout_blocks(&mut page.text_items, &page.layout_blocks);
        layout_merge::compact_layout_blocks(&mut page.layout_blocks, &mut page.text_items);
        if let Some(text) = rebuild_text_in_layout_order(page) {
            page.text = text;
        }
    }
}

fn rebuild_text_in_layout_order(page: &ParsedPage) -> Option<String> {
    if page.layout_blocks.is_empty() {
        return None;
    }

    let mut sections = Vec::new();
    for block in &page.layout_blocks {
        let block_items: Vec<_> = page
            .text_items
            .iter()
            .filter(|item| item.layout_block_id == Some(block.id))
            .cloned()
            .collect();
        let text = layout_merge::project_layout_text(
            page.page_number,
            page.page_width,
            page.page_height,
            block_items,
        );
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            sections.push(trimmed.to_string());
        }
    }

    let unassigned_items: Vec<_> = page
        .text_items
        .iter()
        .filter(|item| item.layout_block_id.is_none())
        .cloned()
        .collect();
    let unassigned_text = layout_merge::project_layout_text(
        page.page_number,
        page.page_width,
        page.page_height,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
