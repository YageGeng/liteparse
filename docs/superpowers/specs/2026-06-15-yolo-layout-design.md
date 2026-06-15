# YOLO Layout Detection Design

## Goal

Add document layout detection to LiteParse using a YOLOv26 DocLayNet model while preserving the existing `parse()` API shape: callers still receive a complete `ParseResult`, but the internal pipeline processes extraction, rendering, layout detection, OCR, and merging at page granularity.

## Requirements

- Add a new sub-crate for YOLO layout detection.
- Use the existing LiteParse page rendering path for layout input images.
- Convert the Hugging Face `.pt` YOLOv26 document layout model to ONNX with a Python script.
- Use `burn-onnx` to generate Rust model code from the ONNX graph and compile that generated model into the layout crate.
- Do not add a runtime `layout_model_path` setting.
- Run layout detection by page rather than waiting for full-document layout detection before merging.
- Keep `LiteParse::parse()` returning a full `ParseResult`.
- Expose page-level layout blocks and text-item-level layout assignment in Rust JSON, Node.js, and Python outputs.
- Print per-page progress logs when `quiet=false`.

## Model Source

The target Hugging Face repository is `GengYage/yolo26-document-layout`. As of June 15, 2026, the repository exposes three Ultralytics `.pt` weights: `yolo26n_doc_layout.pt`, `yolo26s_doc_layout.pt`, and `yolo26m_doc_layout.pt`. The model card states the model was trained on DocLayNet v1.2 at `1280x1280` resolution and predicts 11 labels:

- `text`
- `title`
- `section_header`
- `table`
- `picture`
- `caption`
- `list_item`
- `formula`
- `page_header`
- `page_footer`
- `footnote`

The default embedded model should use the nano weight (`yolo26n_doc_layout.pt`) because it is the recommended speed/accuracy tradeoff in the model card.

## Crate Layout

Create a new workspace member:

```text
crates/liteparse-layout-yolo/
├── Cargo.toml
├── build.rs
└── src/
    ├── lib.rs
    ├── detector.rs
    ├── error.rs
    ├── model.rs
    ├── postprocess.rs
    ├── preprocess.rs
    └── types.rs
```

Responsibilities:

- `types.rs`: public layout types, labels, image metadata, detection options.
- `preprocess.rs`: RGB image to YOLO tensor input, including letterbox resize to `layout_image_size`.
- `model.rs`: wrapper around the generated Burn model module.
- `postprocess.rs`: YOLO output decoding, confidence filtering, NMS, and coordinate restoration.
- `detector.rs`: `YoloLayoutDetector` API used by `liteparse`.
- `error.rs`: typed errors for model generation, preprocessing, inference, and postprocessing.
- `build.rs`: invoke `burn_onnx::ModelGen` against the checked-in/generated ONNX artifact and include generated Rust model code in the crate build.

The layout crate should not depend on `liteparse`. It should accept plain RGB image bytes plus metadata and return layout detections in page-space coordinates.

## Model Conversion Assets

Add:

```text
scripts/export-yolo-layout-onnx.py
```

The script should:

1. Download `yolo26n_doc_layout.pt` from `GengYage/yolo26-document-layout` using `huggingface_hub`.
2. Load it with `ultralytics.YOLO`.
3. Export ONNX with `imgsz=1280`.
4. Save the result under the layout crate model asset directory.

The script is an offline developer/build-prep tool. Runtime parsing must not download model files.

## Core LiteParse Changes

Add configuration fields to `LiteParseConfig`:

```rust
pub layout_enabled: bool,
pub layout_confidence_threshold: f32,
pub layout_iou_threshold: f32,
pub layout_image_size: u32,
```

Defaults:

```rust
layout_enabled: false,
layout_confidence_threshold: 0.25,
layout_iou_threshold: 0.45,
layout_image_size: 1280,
```

Do not add `layout_model_path`.

Add result fields:

```rust
pub struct TextItem {
    pub layout_block_id: Option<usize>,
    pub layout_label: Option<String>,
}

pub struct ParsedPage {
    pub layout_blocks: Vec<LayoutBlock>,
}
```

`LayoutBlock` should serialize with stable snake_case Rust field names and language-wrapper-friendly camelCase equivalents in Node.

## Page-Level Pipeline

`parse()` should still return a full `ParseResult`, but the implementation should be organized around pages:

1. Resolve input and target pages.
2. Open the document under the PDFium global lock.
3. For each target page:
   - Extract native text for that page.
   - Render the page once to RGB using the existing rendering/PDFium bitmap path.
   - Decide whether OCR is needed for that page.
   - Run OCR and layout detection for that page concurrently when both are enabled.
   - Merge OCR results into that page's `text_items`.
   - Project that page's text to grid text.
   - Attach layout blocks to the parsed page.
   - Assign each text item to the best matching layout block.
4. Concatenate parsed page text into the document-level `ParseResult.text`.

PDFium access remains serialized. CPU or async work that does not touch PDFium should run outside the PDFium lock when possible. Page-level implementation can still collect all page results before returning, preserving the public API.

## Layout/Text Merge

The merge algorithm should be deterministic:

1. Sort layout blocks by page reading order: top-to-bottom, then left-to-right.
2. Assign stable `id` values starting at `0` within each page after sorting.
3. For each text item, compute overlap ratio against all layout blocks.
4. Prefer the block with the largest intersection over text-item area.
5. Assign when overlap is at least `0.5`, or when the text item center point lies inside the block.
6. Leave `layout_block_id` and `layout_label` as `None` if no block matches.

This avoids forcing decorative or marginal text into weak layout detections.

## Logging

When `quiet=false`, print page-level logs:

```text
[liteparse] page 3 extract: 12.4ms, items=87
[liteparse] page 3 layout: 44.1ms, blocks=9
[liteparse] page 3 ocr: 0.0ms, skipped
[liteparse] page 3 merge: text_items=91, assigned=88
[liteparse] page 3 total: 63.2ms
```

If layout is disabled, log `layout: 0.0ms, disabled`. If layout inference fails on a page, return an error rather than silently dropping layout results when `layout_enabled=true`.

## Language Bindings

Rust JSON output:

- Add `layout_blocks` to each JSON page.
- Add `layout_block_id` and `layout_label` to each JSON text item, skipped when absent.

Node.js:

- Add config fields in `LiteParseConfig` and napi config conversion.
- Add `LayoutBlock` TypeScript interface.
- Add `layoutBlocks` on `ParsedPage`.
- Add `layoutBlockId` and `layoutLabel` on `TextItem`.

Python:

- Add config keyword arguments.
- Add `LayoutBlock` dataclass.
- Add `layout_blocks` on `ParsedPage`.
- Add `layout_block_id` and `layout_label` on `TextItem`.

WASM:

- Keep layout disabled by default.
- Do not expose embedded Burn model inference in WASM in this change.
- Preserve compatibility for existing WASM parsing.

## Testing Strategy

Unit tests:

- Layout label serialization.
- Letterbox metadata and coordinate restoration.
- NMS keeps the highest-confidence overlapping detection.
- Text-to-layout assignment chooses the largest overlap and leaves unmatched items unassigned.
- Config defaults and serde roundtrip.
- JSON formatter includes layout fields only when present.

Integration tests:

- Parse with `layout_enabled=false` to confirm existing behavior remains compatible.
- Parse a small PDF with a fake or fixture layout detector if the implementation introduces an injectable detector trait.
- Verify Node and Python wrappers preserve layout fields in converted results.

Model inference tests should not require downloading Hugging Face assets during normal test runs. Full model export/inference can be covered by an ignored test or documented manual command.

## Non-Goals

- No streaming public API in this change.
- No runtime model download.
- No `layout_model_path` configuration.
- No HTTP layout server.
- No WASM layout inference.
- No changes to the existing `parse()` return timing contract beyond internal page-level work organization.
