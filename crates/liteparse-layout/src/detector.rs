use crate::error::LayoutError;
use crate::model::EmbeddedYoloModel;
use crate::preprocess::validate_page_image;
use crate::types::{LayoutDetection, PageImage};

/// Runtime options used by the embedded YOLO layout detector.
#[derive(Debug, Clone)]
pub struct YoloLayoutOptions {
    /// Minimum class confidence required before a candidate is considered.
    pub confidence_threshold: f32,
    /// IoU threshold used by non-maximum suppression.
    pub iou_threshold: f32,
    /// Square input image size expected by the embedded model.
    pub image_size: u32,
}

impl Default for YoloLayoutOptions {
    /// Return conservative defaults for DocLayNet-style page layout detection.
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            image_size: 1280,
        }
    }
}

/// Document layout detector backed by the embedded YOLO model.
#[derive(Debug, Clone)]
pub struct YoloLayoutDetector {
    model: EmbeddedYoloModel,
    options: YoloLayoutOptions,
}

impl YoloLayoutDetector {
    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    /// Create a YOLO layout detector with a synchronously initialized backend.
    pub fn new(options: YoloLayoutOptions) -> Result<Self, LayoutError> {
        Ok(Self {
            model: EmbeddedYoloModel::new()?,
            options,
        })
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Create a YOLO layout detector with an asynchronously initialized WebGPU backend.
    pub async fn new_async(options: YoloLayoutOptions) -> Result<Self, LayoutError> {
        Ok(Self {
            model: EmbeddedYoloModel::new_async().await?,
            options,
        })
    }

    /// Detect document layout regions for one rendered page image.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when image validation, preprocessing, inference,
    /// or postprocessing fails.
    pub fn detect_page(&self, image: &PageImage<'_>) -> Result<Vec<LayoutDetection>, LayoutError> {
        validate_page_image(image)?;
        self.model.detect(
            image,
            self.options.confidence_threshold,
            self.options.iou_threshold,
            self.options.image_size,
        )
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Detect document layout regions through the asynchronous browser WebGPU path.
    ///
    /// # Errors
    ///
    /// Returns [`LayoutError`] when image validation, WebGPU inference, or
    /// postprocessing fails.
    pub async fn detect_page_async(
        &self,
        image: &PageImage<'_>,
    ) -> Result<Vec<LayoutDetection>, LayoutError> {
        validate_page_image(image)?;
        self.model
            .detect_async(
                image,
                self.options.confidence_threshold,
                self.options.iou_threshold,
                self.options.image_size,
            )
            .await
    }
}
