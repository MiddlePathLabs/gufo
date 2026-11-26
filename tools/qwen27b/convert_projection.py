#!/usr/bin/env python3
"""Convert per-module rank-1 directions (safetensors) into a Qwen27B
projection GGUF for `gufo serve llm --projection-vector`.

The input holds one F32 `[hidden]` unit direction per residual writer, keyed
by the Hugging Face module name (`model.language_model.layers.N.
{linear_attn.out_proj,self_attn.o_proj,mlp.down_proj}`), plus an optional
`__coefs__` F32 tensor whose order is the `coef_order` JSON list in the
safetensors metadata. Each output tensor is `coef * direction`: Gufo reads
its norm back as the writer's scale, so lambda 1 reproduces the source
ablation profile per module.

    python3 tools/qwen27b/convert_projection.py refusal_dirs_qwen38.safetensors \
        qwen27b-projection.gguf
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

_MODULE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(linear_attn\.out_proj|self_attn\.o_proj|"
    r"mlp\.down_proj)$")
_SITE = {"linear_attn.out_proj": "mixer", "self_attn.o_proj": "mixer",
         "mlp.down_proj": "ffn"}
_GGUF_MAGIC = 0x46554747
_GGUF_VERSION = 3
_TYPE_STRING = 8
_TYPE_UINT32 = 4
_GGML_F32 = 0
_ALIGN = 32


def read_safetensors(path: Path):
    raw = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8:8 + header_len])
    base = 8 + header_len
    meta = header.pop("__metadata__", {}) or {}
    tensors = {}
    for name, info in header.items():
        if info["dtype"] != "F32":
            raise SystemExit(f"{name}: expected F32, got {info['dtype']}")
        start, end = info["data_offsets"]
        tensors[name] = np.frombuffer(raw[base + start:base + end],
                                      dtype="<f4").reshape(info["shape"])
    return tensors, meta


def gguf_string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, metadata: dict[str, object],
               tensors: list[tuple[str, np.ndarray]]) -> None:
    out = bytearray()
    out += struct.pack("<IIQQ", _GGUF_MAGIC, _GGUF_VERSION, len(tensors),
                       len(metadata))
    for key, value in metadata.items():
        out += gguf_string(key)
        if isinstance(value, str):
            out += struct.pack("<I", _TYPE_STRING) + gguf_string(value)
        else:
            out += struct.pack("<II", _TYPE_UINT32, int(value))
    offset = 0
    for name, array in tensors:
        out += gguf_string(name)
        out += struct.pack("<I", array.ndim)
        for dim in reversed(array.shape):
            out += struct.pack("<Q", dim)
        out += struct.pack("<IQ", _GGML_F32, offset)
        offset += (array.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    out += b"\0" * (-len(out) % _ALIGN)
    for _, array in tensors:
        data = array.astype("<f4").tobytes()
        out += data + b"\0" * (-len(data) % _ALIGN)
    path.write_bytes(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-hint", default="qwen35",
                        help="general.architecture the file is meant for")
    parser.add_argument("--no-coefs", action="store_true",
                        help="ignore __coefs__ and write unit directions")
    args = parser.parse_args()

    tensors, meta = read_safetensors(args.input)
    coefs = {}
    if "__coefs__" in tensors and not args.no_coefs:
        order = json.loads(meta.get("coef_order", "[]"))
        values = tensors["__coefs__"].reshape(-1)
        if len(order) != len(values):
            raise SystemExit("coef_order does not match __coefs__")
        coefs = dict(zip(order, (float(v) for v in values)))

    out: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for name in sorted(tensors):
        if name == "__coefs__":
            continue
        match = _MODULE.search(name)
        if match is None:
            print(f"skipping {name}: not a residual writer", file=sys.stderr)
            continue
        layer, module = int(match.group(1)), match.group(2)
        gguf_name = f"projection.{_SITE[module]}.{layer}"
        if gguf_name in seen:
            raise SystemExit(f"{gguf_name} claimed twice (by {name})")
        seen.add(gguf_name)
        direction = tensors[name].reshape(-1).astype(np.float64)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            raise SystemExit(f"{name}: zero or non-finite direction")
        scale = coefs.get(name, 1.0)
        out.append((gguf_name, (direction / norm * scale).astype(np.float32)))

    if not out:
        raise SystemExit("no residual-writer directions found")
    out.sort(key=lambda item: (int(item[0].rsplit(".", 1)[1]), item[0]))
    metadata: dict[str, object] = {
        "general.architecture": "projection",
        "projection.model_hint": args.model_hint,
        "projection.source_base": str(meta.get("source_base", "")),
        "projection.source_ablation": str(meta.get("source_abl", "")),
    }
    write_gguf(args.output, metadata, out)
    print(f"wrote {len(out)} directions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
