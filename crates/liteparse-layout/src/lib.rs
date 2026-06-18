//! YOLO document layout detection support for LiteParse.

/// High-level detector API and runtime options.
pub mod detector;
/// Error types returned by layout detection stages.
pub mod error;
/// Typed model labels and label conversion helpers.
pub mod labels;
/// Embedded Burn model wrapper and backend-specific inference code.
pub mod model;
/// Reading-order sorting for detected layout regions.
pub mod order;
/// Model-output decoding and non-max suppression.
pub mod postprocess;
/// Page-image validation and YOLO input preprocessing.
pub mod preprocess;
/// Shared input and output data types.
pub mod types;

pub use detector::{YoloLayoutDetector, YoloLayoutOptions};
pub use error::LayoutError;
pub use labels::{LAYOUT_LABELS, LayoutLabel, LayoutLabelError};
pub use order::order_layout_detections_xy_cut;
pub use postprocess::{DetectionCandidate, non_max_suppression, restore_box_to_page};
pub use preprocess::Letterbox;
pub use types::{LayoutDetection, PageImage};
