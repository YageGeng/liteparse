# YOLO Layout Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project constraint: do not create commits unless the user explicitly allows them.

**Goal:** Add page-level YOLO document layout detection to LiteParse using an embedded Burn-generated model while keeping `parse()` returning a complete `ParseResult`.

**Architecture:** Add a standalone `liteparse-layout-yolo` crate for layout types, preprocessing, postprocessing, generated Burn model wrapping, and detection. Extend the core `liteparse` crate with layout config, page/text output fields, page-level merge logic, and parser orchestration that can process OCR and layout per page before assembling the full result.

**Tech Stack:** Rust 2024, Cargo workspace, Burn/Burn ONNX code generation, PDFium rendering already present in `crates/pdfium`, Python Ultralytics export script, napi-rs, PyO3.

---

## File Structure

- Create `crates/liteparse-layout-yolo/Cargo.toml`: layout crate manifest.
- Create `crates/liteparse-layout-yolo/build.rs`: optional Burn ONNX model generation hook.
- Create `crates/liteparse-layout-yolo/src/lib.rs`: public exports.
- Create `crates/liteparse-layout-yolo/src/types.rs`: layout data structures and labels.
- Create `crates/liteparse-layout-yolo/src/preprocess.rs`: letterbox resize metadata and RGB tensor preparation.
- Create `crates/liteparse-layout-yolo/src/postprocess.rs`: box math, confidence filtering, NMS, coordinate restoration.
- Create `crates/liteparse-layout-yolo/src/detector.rs`: detector options and page detection entry point.
- Create `crates/liteparse-layout-yolo/src/error.rs`: layout-specific error type.
- Create `crates/liteparse-layout-yolo/src/model.rs`: generated Burn model wrapper and fallback compile-time boundary.
- Create `scripts/export-yolo-layout-onnx.py`: offline `.pt` to ONNX export tool.
- Modify `Cargo.toml`: add workspace member.
- Modify `crates/liteparse/Cargo.toml`: optional dependency on layout crate behind a `layout-yolo` feature.
- Modify `crates/liteparse/src/types.rs`: add `LayoutBlock`, layout fields on `TextItem` and `ParsedPage`.
- Modify `crates/liteparse/src/config.rs`: add layout config defaults and serde coverage.
- Modify `crates/liteparse/src/parser.rs`: add page-level layout orchestration and logging.
- Modify `crates/liteparse/src/extract.rs`: expose page-level extraction helper for parser use.
- Modify `crates/liteparse/src/ocr_merge.rs`: expose page-level OCR render/merge helper or adapt existing batch helpers safely.
- Modify `crates/liteparse/src/output/json.rs`: serialize layout fields.
- Modify `crates/liteparse/src/main.rs`: add CLI layout flags.
- Modify `crates/liteparse-napi/src/types.rs` and `crates/liteparse-napi/src/lib.rs`: expose layout config/result fields.
- Modify `packages/node/src/lib.ts` and `packages/node/src/native.ts`: update TypeScript public types.
- Modify `crates/liteparse-python/src/lib.rs`, `packages/python/liteparse/types.py`, and `packages/python/liteparse/parser.py`: expose Python layout config/result fields.
- Modify `crates/liteparse-wasm/src/lib.rs`: preserve compatibility with layout disabled.

---

### Task 1: Add Core Layout Types and Merge Semantics

**Files:**
- Modify: `crates/liteparse/src/types.rs`
- Create: `crates/liteparse/src/layout_merge.rs`
- Modify: `crates/liteparse/src/lib.rs`

- [ ] **Step 1: Write failing tests for layout assignment**

Add unit tests in `crates/liteparse/src/layout_merge.rs` covering:

```rust
#[test]
fn assigns_text_item_to_block_with_largest_overlap() {
    let mut items = vec![TextItem {
        text: "Revenue".into(),
        x: 12.0,
        y: 12.0,
        width: 40.0,
        height: 10.0,
        ..Default::default()
    }];
    let blocks = vec![
        LayoutBlock { id: 0, label: "table".into(), confidence: 0.9, x: 0.0, y: 0.0, width: 20.0, height: 20.0 },
        LayoutBlock { id: 1, label: "text".into(), confidence: 0.8, x: 10.0, y: 10.0, width: 60.0, height: 20.0 },
    ];

    assign_text_items_to_layout_blocks(&mut items, &blocks);

    assert_eq!(items[0].layout_block_id, Some(1));
    assert_eq!(items[0].layout_label.as_deref(), Some("text"));
}
```

Also add tests for center-point assignment and unmatched items.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p liteparse layout_merge --no-default-features`

Expected: FAIL because `layout_merge`, `LayoutBlock`, and text item layout fields do not exist yet.

- [ ] **Step 3: Implement minimal layout types and merge function**

Add `LayoutBlock` to `types.rs`:

```rust
#[derive(Debug, Clone, Default, Serialize)]
pub struct LayoutBlock {
    pub id: usize,
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
```

Add `layout_block_id` and `layout_label` to `TextItem`, skipped when `None`. Add `layout_blocks: Vec<LayoutBlock>` to `ParsedPage`. Implement `layout_merge::assign_text_items_to_layout_blocks`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p liteparse layout_merge --no-default-features`

Expected: PASS.

- [ ] **Step 5: Inspect diff**

Run: `git diff -- crates/liteparse/src/types.rs crates/liteparse/src/layout_merge.rs crates/liteparse/src/lib.rs`

Expected: only layout type and merge additions.

---

### Task 2: Add Layout Configuration and JSON Serialization

**Files:**
- Modify: `crates/liteparse/src/config.rs`
- Modify: `crates/liteparse/src/output/json.rs`

- [ ] **Step 1: Write failing tests**

Add config tests asserting defaults:

```rust
let c = LiteParseConfig::default();
assert!(!c.layout_enabled);
assert_eq!(c.layout_confidence_threshold, 0.25);
assert_eq!(c.layout_iou_threshold, 0.45);
assert_eq!(c.layout_image_size, 1280);
```

Add JSON formatter test with a page containing one `LayoutBlock` and one assigned `TextItem`, asserting `layout_blocks`, `layout_block_id`, and `layout_label` are present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p liteparse config output::json --no-default-features`

Expected: FAIL because config and JSON fields do not exist yet.

- [ ] **Step 3: Implement config and JSON fields**

Add fields and defaults in `LiteParseConfig`. Extend `JsonTextItem` and `JsonPage` with layout fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p liteparse config output::json --no-default-features`

Expected: PASS.

---

### Task 3: Create `liteparse-layout-yolo` Crate with Testable Pre/Postprocessing

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/liteparse-layout-yolo/Cargo.toml`
- Create: `crates/liteparse-layout-yolo/src/lib.rs`
- Create: `crates/liteparse-layout-yolo/src/types.rs`
- Create: `crates/liteparse-layout-yolo/src/preprocess.rs`
- Create: `crates/liteparse-layout-yolo/src/postprocess.rs`
- Create: `crates/liteparse-layout-yolo/src/error.rs`

- [ ] **Step 1: Write failing crate tests**

Add tests for:

- Letterbox from `1000x500` to `1280x1280` yields scale `1.28`, resized `1280x640`, vertical pad `320`.
- Restoring model-space box coordinates removes pad and divides by scale.
- NMS keeps the highest-confidence overlapping detection.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p liteparse-layout-yolo`

Expected: FAIL until the crate is added and code is implemented.

- [ ] **Step 3: Implement crate scaffold and math**

Implement public types:

```rust
pub struct PageImage<'a> {
    pub rgb: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub page_width: f32,
    pub page_height: f32,
    pub dpi: f32,
}

pub struct LayoutDetection {
    pub label: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
```

Implement letterbox metadata, coordinate restoration, IoU, and NMS without invoking model inference yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p liteparse-layout-yolo`

Expected: PASS for math-only tests.

---

### Task 4: Add Offline YOLO to ONNX Export Script and Burn Generation Boundary

**Files:**
- Create: `scripts/export-yolo-layout-onnx.py`
- Create: `crates/liteparse-layout-yolo/build.rs`
- Create: `crates/liteparse-layout-yolo/src/model.rs`
- Modify: `crates/liteparse-layout-yolo/src/lib.rs`

- [ ] **Step 1: Write script validation test or dry-run check**

Run: `python3 -m py_compile scripts/export-yolo-layout-onnx.py`

Expected initially: FAIL because the script does not exist.

- [ ] **Step 2: Implement export script**

The script should accept `--variant n|s|m`, `--output-dir`, and `--imgsz`, defaulting to `n`, `crates/liteparse-layout-yolo/model`, and `1280`. It should call `hf_hub_download` and `YOLO(...).export(format="onnx", imgsz=args.imgsz)`.

- [ ] **Step 3: Implement Burn generation boundary**

Add `build.rs` that looks for `model/yolo26n_doc_layout.onnx`. If absent, print a Cargo warning and compile a model module that returns a clear `LayoutError::ModelUnavailable` when `layout_enabled=true`. If present, use `burn_onnx::ModelGen` to generate Rust code into `OUT_DIR`.

- [ ] **Step 4: Run verification**

Run: `python3 -m py_compile scripts/export-yolo-layout-onnx.py`

Run: `cargo test -p liteparse-layout-yolo`

Expected: PASS without requiring model download.

---

### Task 5: Integrate Layout Config and CLI Flags

**Files:**
- Modify: `crates/liteparse/Cargo.toml`
- Modify: `crates/liteparse/src/main.rs`
- Modify: `crates/liteparse/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add tests for CLI/config parsing helpers where practical, or config-level tests if CLI parsing is not unit-testable.

- [ ] **Step 2: Implement feature and flags**

Add a `layout-yolo` feature to `liteparse` depending on `liteparse-layout-yolo`. Add CLI flags:

```text
--layout
--layout-confidence-threshold
--layout-iou-threshold
--layout-image-size
```

Use `layout_enabled=false` unless `--layout` is set.

- [ ] **Step 3: Run verification**

Run: `cargo test -p liteparse --no-default-features`

Expected: PASS.

---

### Task 6: Add Page-Level Parser Orchestration and Logging

**Files:**
- Modify: `crates/liteparse/src/extract.rs`
- Modify: `crates/liteparse/src/ocr_merge.rs`
- Modify: `crates/liteparse/src/parser.rs`

- [ ] **Step 1: Write failing tests**

Add a parser unit test using synthetic `Page`, `LayoutBlock`, and `TextItem` merge helper to verify page-level assignment survives projection. If direct parser integration is too PDFium-heavy for a unit test, keep this as a focused helper test.

- [ ] **Step 2: Extract page-level helpers**

Expose a crate-private helper in `extract.rs` to extract one page from an already-open document. Add an owned rendered RGB page type shared by OCR/layout scheduling.

- [ ] **Step 3: Implement page-level flow**

In `parse_input`, keep public return unchanged but process each target page with page-local timing. Render page RGB once when OCR or layout needs an image. Run OCR and layout work for the page outside PDFium when possible. Merge OCR first, project the single page, then attach layout blocks and assign text items.

- [ ] **Step 4: Add logs**

Print extract, layout, OCR, merge, and total per-page logs when `quiet=false`.

- [ ] **Step 5: Run verification**

Run: `cargo test -p liteparse --no-default-features`

Expected: PASS.

---

### Task 7: Update Node.js, Python, and WASM Wrappers

**Files:**
- Modify: `crates/liteparse-napi/src/types.rs`
- Modify: `packages/node/src/lib.ts`
- Modify: `packages/node/src/native.ts`
- Modify: `packages/node/native.d.ts`
- Modify: `crates/liteparse-python/src/lib.rs`
- Modify: `packages/python/liteparse/types.py`
- Modify: `packages/python/liteparse/parser.py`
- Modify: `crates/liteparse-wasm/src/lib.rs`

- [ ] **Step 1: Write failing wrapper tests or type checks**

Run existing package tests/type checks:

```text
npm test --workspace packages/node
uv run pytest packages/python/tests/test_parse_e2e.py
```

Expected: existing commands may need adjustment based on package scripts; failures should indicate missing fields once tests are added.

- [ ] **Step 2: Implement wrapper field mapping**

Add layout config fields and result fields in napi/PyO3/native TypeScript/Python wrappers. Keep WASM layout disabled and compatible.

- [ ] **Step 3: Run verification**

Run:

```text
cargo test -p liteparse-napi --no-default-features
cargo test -p liteparse-python --no-default-features
cargo test -p liteparse-wasm --no-default-features
```

Expected: PASS or documented dependency-related skips.

---

### Task 8: Final Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run Rust tests**

Run:

```text
cargo test -p liteparse-layout-yolo
cargo test -p liteparse --no-default-features
```

Expected: PASS.

- [ ] **Step 2: Run formatting**

Run:

```text
cargo fmt --all --check
```

Expected: PASS. If it fails due formatting, run `cargo fmt --all`, then rerun check.

- [ ] **Step 3: Review diff**

Run:

```text
git status --short
git diff --stat
```

Expected: only layout-related files changed. No commits are created.
