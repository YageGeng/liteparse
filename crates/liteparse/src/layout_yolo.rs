use crate::config::LiteParseConfig;
use crate::error::LiteParseError;
use crate::extract;
use crate::layout_merge;
use crate::types::{LayoutBlock, ParsedPage, PdfInput};
use liteparse_layout::{
    LayoutDetection, PageImage, YoloLayoutDetector, YoloLayoutOptions,
    order_layout_detections_xy_cut,
};
use pdfium::Library;

/// Rendered page pixels and page-space dimensions used for YOLO layout detection.
#[derive(Debug, Clone)]
pub(crate) struct RenderedLayoutPage {
    /// Tightly packed RGB pixels rendered from the PDF page.
    rgb: Vec<u8>,
    /// Rendered image width in pixels.
    width: u32,
    /// Rendered image height in pixels.
    height: u32,
    /// PDF page width in points.
    page_width: f32,
    /// PDF page height in points.
    page_height: f32,
}

impl RenderedLayoutPage {
    /// Create a rendered layout page from tightly packed RGB pixels.
    pub(crate) fn new(
        rgb: Vec<u8>,
        width: u32,
        height: u32,
        page_width: f32,
        page_height: f32,
    ) -> Self {
        Self {
            rgb,
            width,
            height,
            page_width,
            page_height,
        }
    }

    /// Borrow this rendered page as a YOLO input image.
    fn as_page_image(&self, dpi: f32) -> PageImage<'_> {
        // CONTEXT: The layout crate owns YOLO preprocessing. Core LiteParse
        // only passes a rendered page image and its page-space dimensions.
        PageImage {
            rgb: &self.rgb,
            width: self.width,
            height: self.height,
            page_width: self.page_width,
            page_height: self.page_height,
            dpi,
        }
    }
}

/// Thin core adapter around the YOLO layout detector crate.
#[derive(Debug, Clone)]
pub(crate) struct LayoutDetector {
    inner: YoloLayoutDetector,
}

impl LayoutDetector {
    #[cfg(not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")))]
    /// Create the layout detector for synchronous YOLO backends when layout is enabled.
    pub(crate) fn from_config(config: &LiteParseConfig) -> Result<Option<Self>, LiteParseError> {
        if !config.layout_enabled {
            return Ok(None);
        }

        let detector = YoloLayoutDetector::new(YoloLayoutOptions::default())
            .map_err(|e| LiteParseError::Other(e.to_string()))?;
        Ok(Some(Self { inner: detector }))
    }

    #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
    /// Create the layout detector for the asynchronous browser WebGPU backend.
    pub(crate) async fn from_config_async(
        config: &LiteParseConfig,
    ) -> Result<Option<Self>, LiteParseError> {
        if !config.layout_enabled {
            return Ok(None);
        }

        let detector = YoloLayoutDetector::new_async(YoloLayoutOptions::default())
            .await
            .map_err(|e| LiteParseError::Other(e.to_string()))?;
        Ok(Some(Self { inner: detector }))
    }

    #[cfg(not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")))]
    /// Detect layout blocks from a rendered page with a synchronous backend.
    pub(crate) fn detect_rendered(
        &self,
        rendered: RenderedLayoutPage,
        dpi: f32,
    ) -> Result<Vec<LayoutBlock>, LiteParseError> {
        let image = rendered.as_page_image(dpi);
        self.inner
            .detect_page(&image)
            .map(Self::detections_to_blocks)
            .map_err(|e| LiteParseError::Other(e.to_string()))
    }

    #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
    /// Detect layout blocks from a rendered page with async browser WebGPU readback.
    pub(crate) async fn detect_rendered_async(
        &self,
        rendered: RenderedLayoutPage,
        dpi: f32,
    ) -> Result<Vec<LayoutBlock>, LiteParseError> {
        let image = rendered.as_page_image(dpi);
        self.inner
            .detect_page_async(&image)
            .await
            .map(Self::detections_to_blocks)
            .map_err(|e| LiteParseError::Other(e.to_string()))
    }

    /// Convert YOLO detections into stable page-local layout blocks.
    fn detections_to_blocks(detections: Vec<LayoutDetection>) -> Vec<LayoutBlock> {
        // CONTEXT: Ordering belongs to the YOLO layout module so core output
        // receives already stable page-local ids without knowing YOLO details.
        let blocks = order_layout_detections_xy_cut(detections)
            .into_iter()
            .enumerate()
            .map(|(id, detection)| LayoutBlock {
                id,
                label: String::from(detection.label),
                confidence: detection.confidence,
                x: detection.x,
                y: detection.y,
                width: detection.width,
                height: detection.height,
            })
            .collect();

        blocks
    }
}

/// Detect and assign layout blocks for projected pages.
pub(crate) async fn detect_layout_blocks_for_pages(
    config: &LiteParseConfig,
    input: &PdfInput,
    password: Option<&str>,
    pages: &mut [ParsedPage],
) -> Result<(), LiteParseError> {
    #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
    let detector = LayoutDetector::from_config_async(config).await?;

    #[cfg(not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")))]
    let detector = LayoutDetector::from_config(config)?;

    let Some(detector) = detector else {
        return Ok(());
    };

    let lib = Library::init();
    let document = extract::load_document_from_input(&lib, input, password)?;

    for page in pages {
        // CONTEXT: Layout detection intentionally renders pages after text
        // projection so the invasive model path stays optional and isolated.
        let page_obj = document.page(page.page_number as i32 - 1)?;
        let bitmap = page_obj.render(config.dpi)?;
        let rendered = RenderedLayoutPage::new(
            bitmap.to_rgb(),
            bitmap.width() as u32,
            bitmap.height() as u32,
            page.page_width,
            page.page_height,
        );

        #[cfg(all(target_arch = "wasm32", feature = "layout-yolo-webgpu"))]
        let blocks = detector.detect_rendered_async(rendered, config.dpi).await?;

        #[cfg(not(all(target_arch = "wasm32", feature = "layout-yolo-webgpu")))]
        let blocks = detector.detect_rendered(rendered, config.dpi)?;

        // CONTEXT: Keep the raw detected blocks on the page, then annotate
        // text items with compact block ids for consumers that need grouping.
        page.layout_blocks = blocks;
        layout_merge::assign_text_items_to_layout_blocks(&mut page.text_items, &page.layout_blocks);
        layout_merge::compact_layout_blocks(&mut page.layout_blocks, &mut page.text_items);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use liteparse_layout::LayoutLabel;

    // Verifies that YOLO detections are converted after reading-order sorting.
    #[test]
    fn detections_to_blocks_assigns_reading_order_ids() {
        let blocks = LayoutDetector::detections_to_blocks(vec![
            LayoutDetection {
                label: LayoutLabel::Text,
                confidence: 0.8,
                x: 20.0,
                y: 20.0,
                width: 10.0,
                height: 10.0,
            },
            LayoutDetection {
                label: LayoutLabel::Title,
                confidence: 0.9,
                x: 10.0,
                y: 10.0,
                width: 10.0,
                height: 10.0,
            },
        ]);

        assert_eq!(blocks[0].label, "Title");
        assert_eq!(blocks[0].id, 0);
        assert_eq!(blocks[1].id, 1);
    }
}
