#!/usr/bin/env python3
"""Export and inspect the real Qwen2.5 sub-4-bit validation model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path


REPO_ID = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILENAME = "Qwen2.5-0.5B-Instruct-IQ4_XS.gguf"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CUSTOM_DOMAIN = "com.github.onnxruntime.genai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobius-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.mobius_root / "src"))

    import onnx
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    import mobius.integrations.gguf._builder as gguf_builder
    from mobius.integrations.gguf._tensor_mapping import map_gguf_to_hf_names
    from mobius.integrations.onnx_genai import write_inference_metadata

    logging.basicConfig(level=logging.INFO)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    download_dir = args.artifact_root / "download"
    model_dir = args.artifact_root / "model"
    download_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    gguf_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=GGUF_FILENAME,
            local_dir=download_dir,
        )
    )

    # PR #406 currently chooses the lone Q8_0 embedding as the shared module
    # scaffold for mixed native-IQ files, then rejects Q5_1 fallback projections.
    # Native projections do not consume this setting, so use the intended
    # temporary 4-bit/block-32 scaffold for the non-native fallback modules.
    original_detect = gguf_builder._detect_quant_params

    def detect_with_native_scaffold(gguf_model, gguf_arch: str):
        counts = Counter()
        for name, _raw, qtype, _shape in gguf_model.tensor_items_raw():
            hf_name = map_gguf_to_hf_names(name, gguf_arch)
            if hf_name is not None and hf_name.endswith(".weight"):
                counts[qtype] += 1
        if any(gguf_builder._native_block_format(qtype) for qtype in counts):
            return 4, 32, True
        return original_detect(gguf_model, gguf_arch)

    gguf_builder._detect_quant_params = detect_with_native_scaffold
    package = gguf_builder.build_from_gguf(
        gguf_path,
        keep_quantized=True,
        execution_provider="cpu",
    )
    package.save(str(model_dir), external_data="onnx")
    write_inference_metadata(package, str(model_dir), max_sequence_length=512)
    AutoTokenizer.from_pretrained(TOKENIZER_ID).save_pretrained(model_dir)

    model_path = model_dir / "model.onnx"
    model = onnx.load(model_path, load_external_data=False)

    # PR #406 emits the custom-domain nodes but omits the required opset import.
    if not any(item.domain == CUSTOM_DOMAIN for item in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid(CUSTOM_DOMAIN, 1))
        onnx.save_model(model, model_path)

    formats: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.domain != CUSTOM_DOMAIN or node.op_type != "BlockQuantizedMatMul":
            continue
        attrs = {
            attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        format_name = attrs["format"]
        if isinstance(format_name, bytes):
            format_name = format_name.decode()
        formats[format_name] += 1
        assert attrs["block_layout_version"] == 1
        assert attrs["K"] > 0 and attrs["N"] > 0
        if len(samples) < 3:
            samples.append(
                {
                    "name": node.name,
                    "format": format_name,
                    "K": attrs["K"],
                    "N": attrs["N"],
                }
            )

    assert formats["iq4_nl"] > 0
    assert formats["iq4_xs"] > 0
    report = {
        "gguf": str(gguf_path),
        "model_dir": str(model_dir),
        "custom_domain": CUSTOM_DOMAIN,
        "block_quantized_matmul_count": sum(formats.values()),
        "formats": dict(sorted(formats.items())),
        "samples": samples,
        "opset_imports": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
    }
    (args.artifact_root / "graph-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
