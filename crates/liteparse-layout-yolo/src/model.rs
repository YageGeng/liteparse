use crate::error::LayoutError;
use crate::postprocess::{DetectionCandidate, non_max_suppression, restore_box_to_page};
use crate::preprocess::letterbox_rgb_to_chw_f32;
use crate::types::{LayoutDetection, PageImage};
use burn::tensor::{Tensor, TensorData};
use std::sync::Arc;

mod yolo26 {
    #![allow(dead_code, unused_variables)]
    include!(concat!(env!("OUT_DIR"), "/model/yolo26n_doc_layout.rs"));
}

#[cfg(all(
    feature = "backend-ndarray",
    any(feature = "backend-metal", feature = "backend-vulkan")
))]
compile_error!("select only one YOLO layout backend feature");

#[cfg(all(feature = "backend-metal", feature = "backend-vulkan"))]
compile_error!("select only one YOLO layout backend feature");

#[cfg(not(any(
    feature = "backend-ndarray",
    feature = "backend-metal",
    feature = "backend-vulkan"
)))]
compile_error!(
    "select one YOLO layout backend feature: backend-ndarray, backend-metal, or backend-vulkan"
);

const MODEL_IMAGE_SIZE: u32 = 1280;
const LABELS: [&str; 11] = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
];

#[cfg(feature = "backend-ndarray")]
type YoloBackend = burn_ndarray::NdArray<f32>;
#[cfg(feature = "backend-ndarray")]
type YoloDevice = burn_ndarray::NdArrayDevice;

#[cfg(feature = "backend-metal")]
type YoloBackend = burn_wgpu::Metal;
#[cfg(feature = "backend-metal")]
type YoloDevice = burn_wgpu::WgpuDevice;

#[cfg(feature = "backend-vulkan")]
type YoloBackend = burn_wgpu::Vulkan;
#[cfg(feature = "backend-vulkan")]
type YoloDevice = burn_wgpu::WgpuDevice;

#[cfg(feature = "backend-ndarray")]
const BACKEND_NAME: &str = "ndarray";
#[cfg(feature = "backend-metal")]
const BACKEND_NAME: &str = "metal";
#[cfg(feature = "backend-vulkan")]
const BACKEND_NAME: &str = "vulkan";

#[derive(Debug, Clone)]
pub struct EmbeddedYoloModel {
    model: Arc<yolo26::Model<YoloBackend>>,
    device: YoloDevice,
}

impl EmbeddedYoloModel {
    pub fn new() -> Result<Self, LayoutError> {
        let device = create_device();
        let model = yolo26::Model::from_embedded(&device);
        eprintln!("[liteparse-layout-yolo] backend: {BACKEND_NAME}");

        Ok(Self {
            model: Arc::new(model),
            device,
        })
    }

    pub fn detect(
        &self,
        image: &PageImage<'_>,
        confidence_threshold: f32,
        iou_threshold: f32,
        image_size: u32,
    ) -> Result<Vec<LayoutDetection>, LayoutError> {
        if image_size != MODEL_IMAGE_SIZE {
            return Err(LayoutError::UnsupportedImageSize {
                expected: MODEL_IMAGE_SIZE,
                actual: image_size,
            });
        }

        let (input, letterbox) = letterbox_rgb_to_chw_f32(image, MODEL_IMAGE_SIZE)?;
        let tensor = Tensor::<YoloBackend, 4>::from_data(
            TensorData::new(
                input,
                [1, 3, MODEL_IMAGE_SIZE as usize, MODEL_IMAGE_SIZE as usize],
            ),
            &self.device,
        );
        let output = self.model.forward(tensor).into_data();
        let shape = output.shape.clone();
        let values = output
            .to_vec::<f32>()
            .map_err(|error| LayoutError::InvalidModelOutput(error.to_string()))?;

        if shape.as_slice() != [1, 300, 6] {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected [1, 300, 6], got {shape:?}"
            )));
        }

        let mut candidates = Vec::new();
        for row in values.chunks_exact(6) {
            let confidence = row[4];
            if confidence < confidence_threshold {
                continue;
            }

            let class_id = row[5].round() as usize;
            let Some(label) = LABELS.get(class_id) else {
                continue;
            };
            let x = row[0];
            let y = row[1];
            let width = (row[2] - row[0]).max(0.0);
            let height = (row[3] - row[1]).max(0.0);
            if width <= 0.0 || height <= 0.0 {
                continue;
            }

            let (page_x, page_y, page_width, page_height) = restore_box_to_page(
                x,
                y,
                width,
                height,
                &letterbox,
                image.page_width,
                image.page_height,
            );
            if page_width <= 0.0 || page_height <= 0.0 {
                continue;
            }

            candidates.push(DetectionCandidate {
                label: (*label).to_owned(),
                confidence,
                x: page_x,
                y: page_y,
                width: page_width,
                height: page_height,
            });
        }

        Ok(non_max_suppression(candidates, iou_threshold))
    }
}

#[cfg(feature = "backend-ndarray")]
fn create_device() -> YoloDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

#[cfg(feature = "backend-metal")]
fn create_device() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());
    device
}

#[cfg(feature = "backend-vulkan")]
fn create_device() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());
    device
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_compiled_backend_name() {
        assert!(matches!(BACKEND_NAME, "ndarray" | "metal" | "vulkan"));
    }
}
