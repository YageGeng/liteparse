#!/usr/bin/env node
/**
 * Build the LiteParse Wasm package with the linker flags needed by the
 * statically linked PDFium archive.
 */

import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const target = process.argv[2] ?? "web";
const feature = process.argv[3];
const wasmPackArgs = [
  "build",
  "../../crates/liteparse-wasm",
  "--release",
  "--target",
  target,
  "--out-dir",
  "../../packages/wasm/pkg",
  "--out-name",
  "liteparse_wasm",
];

if (feature) {
  wasmPackArgs.push("--", "--features", feature);
}

// PDFium's Wasm archive imports the __c_longjmp exception tag. The generated
// JS glue provides that tag after patching, so rust-lld must keep it undefined.
const wasmTargetRustflags = [
  process.env.CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS,
  "-C link-arg=--allow-undefined",
]
  .filter(Boolean)
  .join(" ");

const build = spawnSync("wasm-pack", wasmPackArgs, {
  cwd: join(__dirname, ".."),
  env: {
    ...process.env,
    CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS: wasmTargetRustflags,
  },
  stdio: "inherit",
});

if (build.status !== 0) {
  process.exit(build.status ?? 1);
}

const patch = spawnSync("node", ["scripts/patch-wasi-imports.js"], {
  cwd: join(__dirname, ".."),
  stdio: "inherit",
});

process.exit(patch.status ?? 1);
