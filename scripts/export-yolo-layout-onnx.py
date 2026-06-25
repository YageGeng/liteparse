#!/usr/bin/env python3
"""Export the YOLOv26 document layout model from Hugging Face to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO


MODEL_FILES = {
    "n": "yolo26n_doc_layout.pt",
    "s": "yolo26s_doc_layout.pt",
    "m": "yolo26m_doc_layout.pt",
}

ONNX_MODEL_NAME = "yolo26_doc_layout.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a YOLOv26 DocLayNet .pt weight and export it to ONNX.",
    )
    parser.add_argument("--variant", choices=sorted(MODEL_FILES), default="n")
    parser.add_argument(
        "--repo-id",
        default="GengYage/yolo26-document-layout",
        help="Hugging Face model repository id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    filename = MODEL_FILES[args.variant]
    weight_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=filename,
        repo_type="model",
        local_dir=args.output_dir,
    )

    model = YOLO(weight_path)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=True,
    )

    exported_path = Path(exported)
    target_path = args.output_dir / ONNX_MODEL_NAME
    if exported_path.resolve() != target_path.resolve():
        exported_path.replace(target_path)

    print(target_path)


if __name__ == "__main__":
    main()
