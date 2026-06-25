//! Adapter crate that exposes the Burn/ONNX YOLO detector as a LiteParse layout provider.
//!
//! The core `liteparse` crate only knows about the generic `LayoutProvider`
//! trait. This crate owns the coupling between that trait and the YOLO detector
//! crate so model, Burn, and WebGPU dependencies stay outside the parser core.

#[cfg(any(
    feature = "backend-metal",
    feature = "backend-ndarray",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
))]
mod yolo;

#[cfg(any(
    feature = "backend-metal",
    feature = "backend-ndarray",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
))]
pub use yolo::YoloLayoutProvider;

#[cfg(any(
    feature = "backend-metal",
    feature = "backend-ndarray",
    feature = "backend-vulkan",
    feature = "backend-webgpu"
))]
pub use liteparse_layout::YoloLayoutOptions;
