use std::pin::Pin;
use std::sync::Arc;

use crate::error::LiteParseError;
use crate::extract;
use crate::types::{LayoutBlock, ParsedPage, PdfInput};
use pdfium::Document;
use pdfium::Library;

/// Error type returned by external layout providers.
pub type LayoutProviderError = Box<dyn std::error::Error + Send + Sync>;

/// Future returned by native layout providers.
#[cfg(not(target_arch = "wasm32"))]
pub type LayoutProviderFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<LayoutBlock>, LayoutProviderError>> + Send + 'a>>;

/// Future returned by wasm layout providers.
#[cfg(target_arch = "wasm32")]
pub type LayoutProviderFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Vec<LayoutBlock>, LayoutProviderError>> + 'a>>;

/// Maximum number of rendered RGB pages kept in memory while running layout detection.
///
/// Each page image can be several megabytes at normal parsing DPI, so layout
/// detection deliberately renders and processes bounded batches instead of
/// collecting the whole document as images. Tune this constant when the
/// provider runtime has a different memory/performance profile.
pub(crate) const LAYOUT_RENDER_BATCH_SIZE: usize = 4;

/// Rendered page image passed to an external layout provider.
#[derive(Debug, Clone)]
pub struct LayoutPageImage {
    /// Index into the current parse result's page slice.
    pub page_index: usize,
    /// One-based PDF page number.
    pub page_number: usize,
    /// Tightly packed RGB pixels in row-major order.
    pub rgb_bytes: Vec<u8>,
    /// Rendered image width in pixels.
    pub width: u32,
    /// Rendered image height in pixels.
    pub height: u32,
    /// PDF page width in viewport-space points.
    pub page_width: f32,
    /// PDF page height in viewport-space points.
    pub page_height: f32,
    /// Render DPI used to produce `rgb_bytes`.
    pub dpi: f32,
}

/// External document-layout detector used to provide page-level layout hints.
#[cfg(not(target_arch = "wasm32"))]
pub trait LayoutProvider: Send + Sync {
    fn name(&self) -> &str;

    fn detect_page<'a>(&'a self, page: &'a LayoutPageImage) -> LayoutProviderFuture<'a>;
}

/// External document-layout detector used to provide page-level layout hints.
#[cfg(target_arch = "wasm32")]
pub trait LayoutProvider {
    fn name(&self) -> &str;

    fn detect_page<'a>(&'a self, page: &'a LayoutPageImage) -> LayoutProviderFuture<'a>;
}

/// Run layout detection for parsed pages using bounded render batches.
///
/// This is the high-level layout helper used by the parser. It owns the full
/// render/detect/assign loop so `parse_input` stays focused on parse stages
/// rather than PDF rendering details. The helper intentionally renders only
/// `LAYOUT_RENDER_BATCH_SIZE` pages at a time, drops the PDFium document, then
/// awaits the provider. Dropping PDFium before `.await` keeps the global PDFium
/// lock out of external provider code and caps peak RGB image memory.
pub(crate) async fn detect_layout_blocks_for_pages(
    input: &PdfInput,
    password: Option<&str>,
    pages: &mut [ParsedPage],
    dpi: f32,
    provider: Arc<dyn LayoutProvider>,
) -> Result<(), LiteParseError> {
    for (start, end) in layout_batch_ranges(pages.len(), LAYOUT_RENDER_BATCH_SIZE) {
        let rendered = render_layout_batch(input, password, pages, start, end, dpi)?;
        detect_and_assign_layout_blocks(pages, rendered, provider.clone()).await?;
    }

    Ok(())
}

/// Render one contiguous page batch into owned RGB buffers.
///
/// This helper opens the PDF document only for the duration of the render
/// batch. The caller must not await while the returned document is alive, so
/// the document is created and dropped entirely inside this synchronous helper.
/// The returned buffers are owned and safe to pass to async providers.
fn render_layout_batch(
    input: &PdfInput,
    password: Option<&str>,
    pages: &[ParsedPage],
    start: usize,
    end: usize,
    dpi: f32,
) -> Result<Vec<LayoutPageImage>, LiteParseError> {
    let lib = Library::init();
    let document = extract::load_document_from_input(&lib, input, password)?;
    render_layout_pages_from_document(&document, pages, start, end, dpi)
}

/// Render pages from an already-open PDFium document.
///
/// The function does no async work and only converts PDFium bitmaps into owned
/// RGB buffers. Keeping this separate makes the PDFium lifetime boundary
/// explicit: callers can render inside a short critical section, then drop the
/// document before invoking an external layout provider.
fn render_layout_pages_from_document(
    document: &Document,
    pages: &[ParsedPage],
    start: usize,
    end: usize,
    dpi: f32,
) -> Result<Vec<LayoutPageImage>, LiteParseError> {
    let mut rendered = Vec::with_capacity(end.saturating_sub(start));
    for (offset, page) in pages[start..end].iter().enumerate() {
        let page_index = start + offset;
        let page_obj = document.page((page.page_number - 1) as i32)?;
        let bitmap = page_obj.render(dpi)?;
        rendered.push(LayoutPageImage {
            page_index,
            page_number: page.page_number,
            rgb_bytes: bitmap.to_rgb(),
            width: bitmap.width() as u32,
            height: bitmap.height() as u32,
            page_width: page.page_width,
            page_height: page.page_height,
            dpi,
        });
    }
    Ok(rendered)
}

/// Run the provider on one rendered batch and attach returned blocks.
///
/// This helper is intentionally small and synchronous-looking from the caller's
/// perspective: it iterates rendered pages, awaits the provider for each page,
/// then writes the returned blocks into the matching `ParsedPage`. The external
/// provider owns output validity for now; this hook only transports blocks.
async fn detect_and_assign_layout_blocks(
    pages: &mut [ParsedPage],
    rendered: Vec<LayoutPageImage>,
    provider: Arc<dyn LayoutProvider>,
) -> Result<(), LiteParseError> {
    for page_image in rendered {
        let blocks = provider
            .detect_page(&page_image)
            .await
            .map_err(|error| LiteParseError::Other(error.to_string()))?;
        if let Some(page) = pages.get_mut(page_image.page_index) {
            page.layout_blocks = blocks;
        }
    }
    Ok(())
}

/// Split a page count into half-open `[start, end)` batch ranges.
///
/// The helper is pure and unit-tested so the memory cap behavior is explicit
/// without needing a multi-page PDF fixture in unit tests.
fn layout_batch_ranges(page_count: usize, batch_size: usize) -> Vec<(usize, usize)> {
    let batch_size = batch_size.max(1);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < page_count {
        let end = (start + batch_size).min(page_count);
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptyProvider;

    impl LayoutProvider for EmptyProvider {
        fn name(&self) -> &str {
            "empty"
        }

        fn detect_page<'a>(&'a self, _page: &'a LayoutPageImage) -> LayoutProviderFuture<'a> {
            Box::pin(async { Ok(Vec::new()) })
        }
    }

    #[tokio::test]
    async fn layout_provider_trait_is_object_safe() {
        let provider: Arc<dyn LayoutProvider> = Arc::new(EmptyProvider);
        let page = LayoutPageImage {
            page_index: 0,
            page_number: 1,
            rgb_bytes: vec![255, 255, 255],
            width: 1,
            height: 1,
            page_width: 1.0,
            page_height: 1.0,
            dpi: 72.0,
        };

        let blocks = provider.detect_page(&page).await.unwrap();

        assert!(blocks.is_empty());
    }

    #[test]
    fn layout_batch_ranges_caps_pages_per_batch() {
        assert_eq!(layout_batch_ranges(0, 4), Vec::<(usize, usize)>::new());
        assert_eq!(layout_batch_ranges(1, 4), vec![(0, 1)]);
        assert_eq!(layout_batch_ranges(9, 4), vec![(0, 4), (4, 8), (8, 9)]);
    }

    #[test]
    fn layout_batch_ranges_treats_zero_batch_as_one() {
        assert_eq!(layout_batch_ranges(3, 0), vec![(0, 1), (1, 2), (2, 3)]);
    }
}
