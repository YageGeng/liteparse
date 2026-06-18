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
pub use types::{LayoutBlock, ParsedPage, TextItem};

// ── Modules with user-facing types (visible in docs) ───────────────────
pub mod config;
pub mod error;
pub mod parser;
pub mod search;
pub mod types;

// ── Internal modules (available for binding crates, hidden from docs) ──
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod conversion;
#[doc(hidden)]
pub mod extract;
#[doc(hidden)]
pub mod figure_cluster;
#[doc(hidden)]
pub mod font_cmap;
#[doc(hidden)]
pub mod glyph_names;
#[cfg(any(
    feature = "layout-yolo",
    feature = "layout-yolo-metal",
    feature = "layout-yolo-vulkan",
    feature = "layout-yolo-webgpu"
))]
#[doc(hidden)]
/// Assigns layout blocks back onto text items without exposing YOLO internals.
pub(crate) mod layout_merge;
#[cfg(any(
    feature = "layout-yolo",
    feature = "layout-yolo-metal",
    feature = "layout-yolo-vulkan",
    feature = "layout-yolo-webgpu"
))]
#[doc(hidden)]
/// Isolates YOLO rendering and inference behind the generic layout flag.
pub(crate) mod layout_yolo;
#[doc(hidden)]
pub mod markdown_layout;
#[doc(hidden)]
pub mod ocr;
#[doc(hidden)]
pub mod ocr_merge;
#[doc(hidden)]
pub mod output;
#[doc(hidden)]
pub mod projection;
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub mod render;
