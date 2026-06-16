//! LiteParse — open-source PDF parsing with spatial text extraction, OCR, and bounding boxes.
//!
//! This crate is the core Rust library. Language bindings for Node.js, Python,
//! and WebAssembly re-export the same types with language-idiomatic wrappers.
//!

// ── Public API re-exports ──────────────────────────────────────────────
pub use config::{LiteParseConfig, OutputFormat};
pub use error::LiteParseError;
pub use parser::{LiteParse, ParseResult, ScreenshotResult};
pub use search::{SearchOptions, search_items};
pub use types::{ParsedPage, TextItem};

// ── Modules with user-facing types (visible in docs) ───────────────────
pub mod config;
pub mod error;
#[doc(hidden)]
pub mod layout_annotate;
pub mod layout_merge;
#[doc(hidden)]
pub mod layout_order;
pub mod parser;
pub mod search;
pub mod types;

// ── Internal modules (available for binding crates, hidden from docs) ──
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod conversion;
#[doc(hidden)]
pub mod extract;
#[cfg(any(
    feature = "layout-yolo",
    feature = "layout-yolo-metal",
    feature = "layout-yolo-vulkan",
    feature = "layout-yolo-webgpu"
))]
#[doc(hidden)]
mod layout_yolo;
#[doc(hidden)]
pub mod ocr;
#[doc(hidden)]
pub mod ocr_merge;
#[doc(hidden)]
pub mod output;
#[doc(hidden)]
pub mod projection;
#[doc(hidden)]
pub mod render;
