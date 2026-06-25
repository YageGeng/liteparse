#[path = "../build_support.rs"]
mod build_support;

use std::path::Path;

#[test]
fn webgpu_topk_patch_fails_when_marker_is_missing() {
    let error = build_support::patch_generated_model(
        "use burn_store::ModuleSnapshot;\nfn forward() {}\n".to_string(),
        Path::new("models/yolo26_doc_layout.onnx"),
        true,
    )
    .unwrap_err();

    assert!(error.contains("TopK patch marker"));
}
