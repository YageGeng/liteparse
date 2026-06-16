use crate::config::LiteParseConfig;
use crate::error::LiteParseError;
use crate::layout_order;
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
    fn detections_to_blocks(detections: Vec<LayoutDetection>) -> Vec<LayoutBlock> {
        let blocks = detections
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

        Self::order_layout_blocks_xy_cut(blocks)
    }

    /// Sort layout blocks with the detector's reading-order strategy.
    ///
    /// Keep this as the layout detector boundary because YOLO detections are
    /// where region-level reading order is established. The standalone
    /// `layout_order` module keeps the XY-cut implementation independently
    /// testable.
    fn order_layout_blocks_xy_cut(blocks: Vec<LayoutBlock>) -> Vec<LayoutBlock> {
        layout_order::order_layout_blocks_xy_cut(blocks)
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

    #[test]
    fn order_layout_blocks_xy_cut_reads_columns_top_to_bottom() {
        let blocks = LayoutDetector::order_layout_blocks_xy_cut(vec![
            block(0, "left-1", 40.0, 40.0, 180.0, 40.0),
            block(1, "right-1", 320.0, 40.0, 180.0, 40.0),
            block(2, "left-2", 40.0, 120.0, 180.0, 40.0),
            block(3, "right-2", 320.0, 120.0, 180.0, 40.0),
        ]);

        let labels: Vec<&str> = blocks.iter().map(|block| block.label.as_str()).collect();
        assert_eq!(labels, vec!["left-1", "left-2", "right-1", "right-2"]);
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
}
