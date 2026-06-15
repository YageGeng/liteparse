use crate::error::LayoutError;
use crate::labels::LayoutLabel;
use crate::postprocess::{DetectionCandidate, non_max_suppression, restore_box_to_page};
use crate::preprocess::{Letterbox, letterbox_rgb_to_chw_f32};
use crate::types::{LayoutDetection, PageImage};
use burn::tensor::{Tensor, TensorData};
use std::sync::Arc;

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
thread_local! {
    static WEBGPU_MODEL_CACHE: std::cell::RefCell<Option<EmbeddedYoloModel>> =
        std::cell::RefCell::new(None);
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
static WEBGPU_MODEL_INIT_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

mod yolo26 {
    #![allow(clippy::type_complexity, dead_code, unused_variables)]
    include!(concat!(env!("OUT_DIR"), "/model/yolo26n_doc_layout.rs"));
}

#[cfg(not(any(
    feature = "backend-ndarray",
    feature = "backend-metal",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
)))]
compile_error!(
    "select one YOLO layout backend feature: backend-ndarray, backend-metal, backend-vulkan, or backend-webgpu"
);

const MODEL_IMAGE_SIZE: u32 = 1280;

// CONTEXT: `cargo clippy --all-features` enables every backend feature at
// once, so backend cfgs use a deterministic priority instead of conflicting.
#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
type YoloBackend = burn_ndarray::NdArray<f32>;
#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
type YoloDevice = burn_ndarray::NdArrayDevice;

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
type YoloBackend = burn_wgpu::Metal;
#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
type YoloDevice = burn_wgpu::WgpuDevice;

#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
type YoloBackend = burn_wgpu::Vulkan;
#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
type YoloDevice = burn_wgpu::WgpuDevice;
#[cfg(feature = "backend-webgpu")]
type YoloBackend = burn_wgpu::WebGpu;
#[cfg(feature = "backend-webgpu")]
type YoloDevice = burn_wgpu::WgpuDevice;

#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
const BACKEND_NAME: &str = "ndarray";
#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
const BACKEND_NAME: &str = "metal";
#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
const BACKEND_NAME: &str = "vulkan";
#[cfg(feature = "backend-webgpu")]
const BACKEND_NAME: &str = "webgpu";

#[derive(Debug, Clone)]
pub struct EmbeddedYoloModel {
    model: Arc<yolo26::Model<YoloBackend>>,
    device: YoloDevice,
}

impl EmbeddedYoloModel {
    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    /// Create an embedded YOLO model with a synchronously initialized backend.
    pub fn new() -> Result<Self, LayoutError> {
        let device = create_device();
        let model = yolo26::Model::from_embedded(&device);
        eprintln!("[liteparse-layout-yolo] backend: {BACKEND_NAME}");

        Ok(Self {
            model: Arc::new(model),
            device,
        })
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Create an embedded YOLO model with the browser WebGPU backend.
    pub async fn new_async() -> Result<Self, LayoutError> {
        if let Some(model) = Self::cached_webgpu_model() {
            eprintln!("[liteparse-layout-yolo] backend: {BACKEND_NAME} (cached)");
            return Ok(model);
        }

        // CONTEXT: CubeCL registers one browser WebGPU server per device.
        // Serializing the first initialization prevents concurrent parses from
        // racing into duplicate server registration before the cache is filled.
        let _init_guard = WEBGPU_MODEL_INIT_LOCK.lock().await;
        if let Some(model) = Self::cached_webgpu_model() {
            eprintln!("[liteparse-layout-yolo] backend: {BACKEND_NAME} (cached)");
            return Ok(model);
        }

        let device = create_device_async().await;
        let model = yolo26::Model::from_embedded(&device);
        eprintln!("[liteparse-layout-yolo] backend: {BACKEND_NAME}");

        let model = Self {
            model: Arc::new(model),
            device,
        };
        Self::set_cached_webgpu_model(&model);

        Ok(model)
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Return the cached browser WebGPU model if this tab has already initialized one.
    fn cached_webgpu_model() -> Option<Self> {
        WEBGPU_MODEL_CACHE.with(|cache| cache.borrow().clone())
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Store the initialized browser WebGPU model for later parses in the same tab.
    fn set_cached_webgpu_model(model: &Self) {
        WEBGPU_MODEL_CACHE.with(|cache| {
            *cache.borrow_mut() = Some(model.clone());
        });
    }

    #[cfg(all(test, target_family = "wasm", feature = "backend-webgpu"))]
    /// Clear the cached browser WebGPU model for cache behavior tests.
    fn clear_cached_webgpu_model_for_test() {
        WEBGPU_MODEL_CACHE.with(|cache| {
            *cache.borrow_mut() = None;
        });
    }

    #[cfg(all(test, target_family = "wasm", feature = "backend-webgpu"))]
    /// Return whether a browser WebGPU model is cached for cache behavior tests.
    fn has_cached_webgpu_model_for_test() -> bool {
        WEBGPU_MODEL_CACHE.with(|cache| cache.borrow().is_some())
    }

    /// Detect layout candidates from one page image using the synchronous backend path.
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
        let candidates = Self::decode_processed_candidates(
            shape.as_slice(),
            &values,
            confidence_threshold,
            image,
            &letterbox,
        )?;

        Ok(non_max_suppression(candidates, iou_threshold))
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Detect layout candidates from one page image using browser WebGPU readback.
    pub async fn detect_async(
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
        let output = self
            .model
            .forward(tensor)
            .into_data_async()
            .await
            .map_err(|error| LayoutError::InvalidModelOutput(error.to_string()))?;
        let shape = output.shape.clone();
        let values = output
            .to_vec::<f32>()
            .map_err(|error| LayoutError::InvalidModelOutput(error.to_string()))?;
        let candidates = Self::decode_raw_webgpu_candidates(
            shape.as_slice(),
            &values,
            confidence_threshold,
            image,
            &letterbox,
        )?;

        Ok(non_max_suppression(candidates, iou_threshold))
    }

    /// Decode the generated model's processed `[x1, y1, x2, y2, score, class]` rows.
    fn decode_processed_candidates(
        shape: &[usize],
        values: &[f32],
        confidence_threshold: f32,
        image: &PageImage<'_>,
        letterbox: &Letterbox,
    ) -> Result<Vec<DetectionCandidate>, LayoutError> {
        if shape != [1, 300, 6] {
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
            let Ok(label) = LayoutLabel::try_from(class_id) else {
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
                letterbox,
                image.page_width,
                image.page_height,
            );
            if page_width <= 0.0 || page_height <= 0.0 {
                continue;
            }

            candidates.push(DetectionCandidate {
                label,
                confidence,
                x: page_x,
                y: page_y,
                width: page_width,
                height: page_height,
            });
        }

        Ok(candidates)
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    /// Decode raw `[x1, y1, x2, y2, class_scores...]` rows before generated TopK.
    fn decode_raw_webgpu_candidates(
        shape: &[usize],
        values: &[f32],
        confidence_threshold: f32,
        image: &PageImage<'_>,
        letterbox: &Letterbox,
    ) -> Result<Vec<DetectionCandidate>, LayoutError> {
        let row_width = LayoutLabel::class_count() + 4;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != row_width {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected [1, N, 15], got {shape:?}"
            )));
        }

        let row_count = shape[1];
        if values.len() != row_count * row_width {
            return Err(LayoutError::InvalidModelOutput(format!(
                "expected {} raw values, got {}",
                row_count * row_width,
                values.len()
            )));
        }

        let mut ranked = Vec::new();
        for row in values.chunks_exact(row_width) {
            let max_score = row[4..].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max_score >= confidence_threshold {
                ranked.push((max_score, row));
            }
        }
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates = Vec::new();
        for (_, row) in ranked.into_iter().take(300) {
            for (class_id, confidence) in row[4..].iter().copied().enumerate() {
                if confidence < confidence_threshold {
                    continue;
                }
                let Ok(label) = LayoutLabel::try_from(class_id) else {
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
                    letterbox,
                    image.page_width,
                    image.page_height,
                );
                if page_width <= 0.0 || page_height <= 0.0 {
                    continue;
                }

                candidates.push(DetectionCandidate {
                    label,
                    confidence,
                    x: page_x,
                    y: page_y,
                    width: page_width,
                    height: page_height,
                });
            }
        }

        Ok(candidates)
    }
}

#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
fn create_device() -> YoloDevice {
    burn_ndarray::NdArrayDevice::Cpu
}

#[cfg(all(
    feature = "backend-metal",
    not(any(feature = "backend-vulkan", feature = "backend-webgpu"))
))]
fn create_device() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Metal>(&device, Default::default());
    device
}

#[cfg(all(feature = "backend-vulkan", not(feature = "backend-webgpu")))]
fn create_device() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::Vulkan>(&device, Default::default());
    device
}

#[cfg(all(not(target_family = "wasm"), feature = "backend-webgpu"))]
/// Create a WebGPU device for native targets that can initialize synchronously.
fn create_device() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup::<burn_wgpu::graphics::WebGpu>(&device, Default::default());
    device
}

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
/// Create a browser WebGPU device through the async adapter request path.
async fn create_device_async() -> YoloDevice {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    burn_wgpu::init_setup_async::<burn_wgpu::graphics::WebGpu>(&device, Default::default()).await;
    device
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_compiled_backend_name() {
        assert!(matches!(
            BACKEND_NAME,
            "ndarray" | "metal" | "vulkan" | "webgpu"
        ));
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    #[test]
    fn webgpu_model_cache_starts_empty_for_tests() {
        EmbeddedYoloModel::clear_cached_webgpu_model_for_test();

        assert!(!EmbeddedYoloModel::has_cached_webgpu_model_for_test());
    }
}
