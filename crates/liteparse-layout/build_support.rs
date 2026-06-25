use std::path::Path;

/// Apply deterministic source patches to the Burn-generated model module.
pub fn patch_generated_model(
    source: String,
    onnx_path: &Path,
    require_webgpu_topk_patch: bool,
) -> Result<String, String> {
    let mut source = replace_generated_header(source, onnx_path);

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
    source = replace_class_count_initializer(source);
    replace_wasm_webgpu_topk_postprocess(source, require_webgpu_topk_patch)
}

/// Replace the absolute generated header with a stable selected-model header.
fn replace_generated_header(source: String, onnx_path: &Path) -> String {
    let Some(line_end) = source.find('\n') else {
        return source;
    };
    if !source.starts_with("// Generated from ONNX ") {
        return source;
    }

    let header = format!(
        "// Generated from ONNX \"{}\" by burn-onnx",
        onnx_path.display()
    );
    format!("{header}{}", &source[line_end..])
}

/// Inject backend-specific helpers used by generated tensor indexing code.
fn insert_index_helpers(source: String) -> String {
    const MARKER: &str = "use burn_store::ModuleSnapshot;\n";
    if source.contains("fn yolo_index_dtype()") {
        return source;
    }

    let helpers = r#"
#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
/// Return the integer dtype required by ndarray tensor indexing.
fn yolo_index_dtype() -> burn::tensor::DType {
    burn::tensor::DType::I64
}

#[cfg(any(feature = "backend-metal", feature = "backend-vulkan", feature = "backend-webgpu"))]
/// Return the integer dtype required by WGPU tensor indexing.
fn yolo_index_dtype() -> burn::tensor::DType {
    burn::tensor::DType::I32
}

#[cfg(all(
    feature = "backend-ndarray",
    not(any(
        feature = "backend-metal",
        feature = "backend-vulkan",
        feature = "backend-webgpu"
    ))
))]
/// Create the class-count tensor using the ndarray integer dtype.
fn class_count_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::from([11i64]),
        (device, burn::tensor::DType::I64),
    )
}

#[cfg(any(feature = "backend-metal", feature = "backend-vulkan", feature = "backend-webgpu"))]
/// Create the class-count tensor using the WGPU integer dtype.
fn class_count_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_data(
        burn::tensor::TensorData::from([11i32]),
        (device, burn::tensor::DType::I32),
    )
}
"#;

    source.replacen(MARKER, &format!("{MARKER}{helpers}"), 1)
}

/// Replace a generated class-count closure with a backend-specific tensor.
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

/// Return raw candidates before Burn's TopK nodes on browser WebGPU builds.
fn replace_wasm_webgpu_topk_postprocess(
    mut source: String,
    require_webgpu_topk_patch: bool,
) -> Result<String, String> {
    let Some(start) = source
        .find("        let split_tensors = transpose5_out1.split_with_sizes([4, 11].into(), 2);")
    else {
        if require_webgpu_topk_patch {
            return Err("Burn WebGPU TopK patch marker was not found in generated model".into());
        }
        return Ok(source);
    };
    let Some(relative_end) = source[start..].find("\n    }\n}\n\n#[derive(Module, Debug)]") else {
        if require_webgpu_topk_patch {
            return Err(
                "Burn WebGPU TopK patch end marker was not found in generated model".into(),
            );
        }
        return Ok(source);
    };
    let end = start + relative_end;
    let original = source[start..end].to_owned();
    // CONTEXT: The native generated path returns processed top-k rows. Browser
    // WebGPU cannot execute that generated TopK reliably yet, so model.rs
    // decodes raw rows for that one target instead.
    let replacement = format!(
        r#"        #[cfg(all(target_family = "wasm", feature = "backend-webgpu"))]
        {{
            // CONTEXT: Burn WebGPU cannot execute generated TopK on wasm yet.
            transpose5_out1
        }}
        #[cfg(not(all(target_family = "wasm", feature = "backend-webgpu")))]
        {{
{original}        }}
"#
    );
    source.replace_range(start..end, &replacement);
    Ok(source)
}
