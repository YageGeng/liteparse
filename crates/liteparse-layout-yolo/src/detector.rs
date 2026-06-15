use crate::error::LayoutError;
use crate::model::EmbeddedYoloModel;
use crate::preprocess::validate_page_image;
use crate::types::{LayoutDetection, PageImage};

#[derive(Debug, Clone)]
pub struct YoloLayoutOptions {
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
    pub image_size: u32,
}

impl Default for YoloLayoutOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            iou_threshold: 0.45,
            image_size: 1280,
        }
    }
}

#[derive(Debug, Clone)]
pub struct YoloLayoutDetector {
    model: EmbeddedYoloModel,
    options: YoloLayoutOptions,
}

impl YoloLayoutDetector {
    pub fn new(options: YoloLayoutOptions) -> Result<Self, LayoutError> {
        Ok(Self {
            model: EmbeddedYoloModel::new()?,
            options,
        })
    }

    pub fn detect_page(&self, image: &PageImage<'_>) -> Result<Vec<LayoutDetection>, LayoutError> {
        validate_page_image(image)?;
        self.model.detect(
            image,
            self.options.confidence_threshold,
            self.options.iou_threshold,
            self.options.image_size,
        )
    }
}
