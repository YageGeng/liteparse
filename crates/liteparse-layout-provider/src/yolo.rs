use liteparse::layout::{
    LayoutPageImage, LayoutProvider, LayoutProviderError, LayoutProviderFuture,
};
use liteparse::types::LayoutBlock;
use liteparse_layout::{
    LayoutDetection, PageImage, YoloLayoutDetector, YoloLayoutOptions,
    order_layout_detections_xy_cut,
};

#[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
#[wasm_bindgen::prelude::wasm_bindgen]
unsafe extern "C" {
    #[wasm_bindgen::prelude::wasm_bindgen(js_namespace = console, js_name = info)]
    fn console_info(message: &str);
}

/// LiteParse layout provider backed by the embedded YOLO document layout model.
pub struct YoloLayoutProvider {
    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    detector: YoloLayoutDetector,
    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    options: YoloLayoutOptions,
    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    detector: std::cell::RefCell<Option<YoloLayoutDetector>>,
}

impl YoloLayoutProvider {
    /// Create a provider for native or synchronously initialized backends.
    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    pub fn new(options: YoloLayoutOptions) -> Result<Self, liteparse_layout::LayoutError> {
        Ok(Self {
            detector: YoloLayoutDetector::new(options)?,
        })
    }

    /// Create a browser WebGPU provider that initializes the model on first use.
    ///
    /// WebGPU adapter/device creation is asynchronous in browsers, while the
    /// wasm `LiteParse` constructor is synchronous. Deferring initialization to
    /// `detect_page` keeps the JS constructor unchanged and still allows the
    /// provider to cache the initialized model for subsequent pages.
    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    pub fn lazy(options: YoloLayoutOptions) -> Self {
        console_info("[liteparse-layout-provider] configured backend: webgpu");
        Self {
            options,
            detector: std::cell::RefCell::new(None),
        }
    }
}

impl LayoutProvider for YoloLayoutProvider {
    fn name(&self) -> &str {
        "yolo-layout"
    }

    #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
    fn detect_page<'a>(&'a self, page: &'a LayoutPageImage) -> LayoutProviderFuture<'a> {
        Box::pin(async move {
            let image = page_image(page);
            self.detector
                .detect_page(&image)
                .map(detections_to_blocks)
                .map_err(provider_error)
        })
    }

    #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
    fn detect_page<'a>(&'a self, page: &'a LayoutPageImage) -> LayoutProviderFuture<'a> {
        Box::pin(async move {
            let cached_detector = { self.detector.borrow().clone() };
            let detector = match cached_detector {
                Some(detector) => {
                    console_info("[liteparse-layout-provider] backend: webgpu (cached)");
                    detector
                }
                None => {
                    let detector = YoloLayoutDetector::new_async(self.options.clone())
                        .await
                        .map_err(provider_error)?;

                    let mut cached_detector = self.detector.borrow_mut();
                    if cached_detector.is_none() {
                        console_info("[liteparse-layout-provider] backend: webgpu initialized");
                        *cached_detector = Some(detector.clone());
                    }
                    detector
                }
            };

            let image = page_image(page);
            detector
                .detect_page_async(&image)
                .await
                .map(detections_to_blocks)
                .map_err(provider_error)
        })
    }
}

/// Borrow a rendered LiteParse page image as the YOLO crate's detector input.
fn page_image(page: &LayoutPageImage) -> PageImage<'_> {
    PageImage {
        rgb: &page.rgb_bytes,
        width: page.width,
        height: page.height,
        page_width: page.page_width,
        page_height: page.page_height,
        dpi: page.dpi,
    }
}

/// Convert ordered YOLO detections into LiteParse page-local layout blocks.
fn detections_to_blocks(detections: Vec<LayoutDetection>) -> Vec<LayoutBlock> {
    order_layout_detections_xy_cut(detections)
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

/// Box YOLO detector errors into the generic provider error type.
fn provider_error(error: liteparse_layout::LayoutError) -> LayoutProviderError {
    Box::new(error)
}
