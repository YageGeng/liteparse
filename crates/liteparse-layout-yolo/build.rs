use burn_onnx::{LoadStrategy, ModelGen};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const ONNX_MODEL: &str = "../../models/yolo26n_doc_layout.onnx";
const GENERATED_DIR: &str = "model";
const GENERATED_MODEL_FILE: &str = "yolo26n_doc_layout.rs";

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

fn generate_burn_model(onnx_path: &Path) {
    let onnx_path = onnx_path
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize ONNX path {}: {error}", onnx_path.display()));

    ModelGen::new()
        .input(onnx_path.to_str().expect("ONNX path should be valid UTF-8"))
        .out_dir(GENERATED_DIR)
        .load_strategy(LoadStrategy::Embedded)
        .run_from_script();

    let generated_path = generated_model_path();
    let source = fs::read_to_string(&generated_path).unwrap_or_else(|error| {
        panic!(
            "read generated burn model {}: {error}",
            generated_path.display()
        )
    });
    fs::write(&generated_path, patch_generated_model(source))
        .expect("write patched generated model source");
}

fn generated_model_path() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join(GENERATED_DIR)
        .join(GENERATED_MODEL_FILE)
}

fn patch_generated_model(mut source: String) -> String {
    source = replace_generated_header(source);

    for (from, to) in [
        (
            "use burn::nn::PaddingConfig2d;",
            "use burn_nn::PaddingConfig2d;",
        ),
        ("use burn::nn::conv::Conv2d;", "use burn_nn::conv::Conv2d;"),
        (
            "use burn::nn::conv::Conv2dConfig;",
            "use burn_nn::conv::Conv2dConfig;",
        ),
        (
            "use burn::nn::pool::MaxPool2d;",
            "use burn_nn::pool::MaxPool2d;",
        ),
        (
            "use burn::nn::pool::MaxPool2dConfig;",
            "use burn_nn::pool::MaxPool2dConfig;",
        ),
        ("burn::nn::interpolate::", "burn_nn::interpolate::"),
        (
            "__topk_indices_raw.cast(burn::tensor::DType::I64)",
            "__topk_indices_raw.cast(yolo_index_dtype())",
        ),
    ] {
        source = source.replace(from, to);
    }

    source = insert_index_helpers(source);
    replace_class_count_initializer(source)
}

fn replace_generated_header(source: String) -> String {
    const HEADER: &str =
        "// Generated from ONNX \"../../models/yolo26n_doc_layout.onnx\" by burn-onnx";

    let Some(line_end) = source.find('\n') else {
        return source;
    };
    if !source.starts_with("// Generated from ONNX ") {
        return source;
    }

    format!("{HEADER}{}", &source[line_end..])
}

fn insert_index_helpers(source: String) -> String {
    const MARKER: &str = "use burn_store::ModuleSnapshot;\n";
    if source.contains("fn yolo_index_dtype()") {
        return source;
    }

    let helpers = r#"
#[cfg(feature = "backend-ndarray")]
fn yolo_index_dtype() -> burn::tensor::DType {
    burn::tensor::DType::I64
}

#[cfg(any(feature = "backend-metal", feature = "backend-vulkan"))]
fn yolo_index_dtype() -> burn::tensor::DType {
    burn::tensor::DType::I32
}

#[cfg(feature = "backend-ndarray")]
fn class_count_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::from([11i64]),
        (device, burn::tensor::DType::I64),
    )
}

#[cfg(any(feature = "backend-metal", feature = "backend-vulkan"))]
fn class_count_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::from([11i32]),
        (device, burn::tensor::DType::I32),
    )
}
"#;

    source.replacen(MARKER, &format!("{MARKER}{helpers}"), 1)
}

fn replace_class_count_initializer(mut source: String) -> String {
    let Some(constant_pos) = source.find("let constant227:") else {
        return source;
    };
    let Some(relative_start) = source[constant_pos..].find("move |device, _require_grad|") else {
        return source;
    };
    let start = constant_pos + relative_start;
    let Some(relative_end) = source[start..].find("\n            device.clone(),") else {
        return source;
    };
    let end = start + relative_end;
    source.replace_range(
        start..end,
        "move |device, _require_grad| class_count_tensor::<B>(device),",
    );
    source
}
