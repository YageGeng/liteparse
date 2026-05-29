#!/bin/sh
set -eux

# tesseract-rs's build.rs hard-codes -DCMAKE_CXX_COMPILER=clang++ and -stdlib=libc++,
# so we need real clang + libc++ in the image (gcc/g++ from build-base is not enough).
# Alpine's libc++ links against libc++abi and libunwind, hence those *-dev packages.
# Static libs (openssl-libs-static, zlib-static) are required because musl rust defaults
# to crt-static for build scripts.
apk add --no-cache \
  build-base cmake git curl pkgconf perl \
  clang libc++-dev libc++abi libunwind-dev \
  tesseract-ocr-dev leptonica-dev \
  openssl-dev openssl-libs-static zlib-static

curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable -t x86_64-unknown-linux-musl
. /root/.cargo/env

export RUSTFLAGS="-C target-feature=-crt-static"
npx napi build --cargo-cwd ../../crates/liteparse-napi --platform --release --js false --dts native.d.ts --target x86_64-unknown-linux-musl .
