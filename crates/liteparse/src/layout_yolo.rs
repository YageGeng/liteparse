use crate::config::LiteParseConfig;
use crate::error::LiteParseError;
use crate::types::LayoutBlock;
use liteparse_layout_yolo::{LayoutDetection, PageImage, YoloLayoutDetector, YoloLayoutOptions};

/// Rendered page pixels and page-space dimensions used for YOLO layout detection.
#[derive(Debug, Clone)]
pub(crate) struct RenderedLayoutPage {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    page_width: f32,
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

        let detector = YoloLayoutDetector::new(Self::options_from_config(config))
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

        let detector = YoloLayoutDetector::new_async(Self::options_from_config(config))
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
    fn detections_to_blocks(mut detections: Vec<LayoutDetection>) -> Vec<LayoutBlock> {
        detections.sort_by(|a, b| {
            a.y.partial_cmp(&b.y)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
        });

        detections
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
            .collect()
    }

    /// Build YOLO detector options from the public LiteParse configuration.
    fn options_from_config(config: &LiteParseConfig) -> YoloLayoutOptions {
        YoloLayoutOptions {
            confidence_threshold: config.layout_confidence_threshold,
            iou_threshold: config.layout_iou_threshold,
            image_size: config.layout_image_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liteparse_layout_yolo::LayoutLabel;

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
