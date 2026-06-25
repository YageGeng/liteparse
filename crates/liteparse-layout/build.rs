use burn_onnx::{LoadStrategy, ModelGen};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod build_support;
use build_support::patch_generated_model;

const ONNX_MODEL: &str = "../../models/yolo26_doc_layout.onnx";
/// Relative output directory used by Burn ONNX code generation.
const GENERATED_DIR: &str = "model";
/// Stable file name included by `model.rs` after build-time patching.
const PATCHED_MODEL_FILE: &str = "yolo26_doc_layout.rs";

/// Generate the embedded Burn model from the exported ONNX file.
fn main() {
    println!("cargo:rerun-if-changed={ONNX_MODEL}");

    let onnx_path = PathBuf::from(ONNX_MODEL);
    if !onnx_path.exists() {
        panic!(
            "YOLO layout ONNX model not found at {}; run `uv run python scripts/export-yolo-layout-onnx.py --variant n` first",
            onnx_path.display()
        );
    }

    generate_burn_model(&onnx_path);
}

/// Run Burn ONNX codegen, then patch generated source for LiteParse backends.
fn generate_burn_model(onnx_path: &Path) {
    let onnx_path = onnx_path
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize ONNX path {}: {error}", onnx_path.display()));

    ModelGen::new()
        .input(onnx_path.to_str().expect("ONNX path should be valid UTF-8"))
        .out_dir(GENERATED_DIR)
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();

    // CONTEXT: Burn codegen is close to usable but still needs small import
    // and dtype patches for the backend combinations this crate supports.
    let generated_path = generated_model_path(&onnx_path);
    let source = fs::read_to_string(&generated_path).unwrap_or_else(|error| {
        panic!(
            "read generated burn model {}: {error}",
            generated_path.display()
        )
    });
    let patched_path = patched_model_path();
    let require_webgpu_topk_patch = env::var_os("CARGO_FEATURE_BACKEND_WEBGPU").is_some();
    let patched = patch_generated_model(source, &onnx_path, require_webgpu_topk_patch)
        .unwrap_or_else(|error| panic!("patch generated burn model: {error}"));
    fs::write(&patched_path, patched)
        .unwrap_or_else(|error| panic!("write patched model {}: {error}", patched_path.display()));
}

/// Return the Burn-generated model source path for the selected ONNX file.
fn generated_model_path(onnx_path: &Path) -> PathBuf {
    let generated_file = onnx_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| format!("{stem}.rs"))
        .expect("ONNX model path should have a UTF-8 file stem");

    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join(GENERATED_DIR)
        .join(generated_file)
}

/// Return the stable patched model path included by the runtime module.
fn patched_model_path() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join(GENERATED_DIR)
        .join(PATCHED_MODEL_FILE)
}
