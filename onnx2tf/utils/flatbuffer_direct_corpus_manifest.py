from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import onnx


SCHEMA_VERSION = 1
TIERS = (
    (0, 1, 49),
    (1, 50, 199),
    (2, 200, 499),
    (3, 500, 999),
    (4, 1000, 1999),
    (5, 2000, None),
)


def node_count_tier(node_count: int) -> int:
    count = int(node_count)
    for tier, minimum, maximum in TIERS:
        if count >= minimum and (maximum is None or count <= maximum):
            return int(tier)
    return 0


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _opsets(model: onnx.ModelProto) -> Dict[str, int]:
    return {
        str(item.domain or "ai.onnx"): int(item.version)
        for item in model.opset_import
    }


def inspect_model(path: Path, *, root: Path, hash_contents: bool = True) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path) if hash_contents else None,
        "status": "pending",
        "expected_classification": None,
        "failure_signature": None,
    }
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as error:
        base.update(
            {
                "status": "invalid_onnx",
                "load_error_type": type(error).__name__,
                "load_error": str(error).splitlines()[0][:500],
                "node_count": None,
                "tier": None,
                "opsets": {},
                "op_counts": {},
            }
        )
        return base

    op_counts = collections.Counter(str(node.op_type) for node in model.graph.node)
    node_count = int(len(model.graph.node))
    base.update(
        {
            "status": "loadable",
            "node_count": node_count,
            "tier": node_count_tier(node_count),
            "opsets": _opsets(model),
            "op_counts": dict(sorted(op_counts.items())),
            "input_names": [str(value.name) for value in model.graph.input],
            "output_names": [str(value.name) for value in model.graph.output],
        }
    )
    return base


def build_corpus_manifest(
    *,
    root_dir: str,
    recursive: bool = False,
    hash_contents: bool = True,
) -> Dict[str, Any]:
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus root does not exist: {root}")
    paths: Iterable[Path] = root.rglob("*.onnx") if recursive else root.glob("*.onnx")
    models = [
        inspect_model(path, root=root, hash_contents=hash_contents)
        for path in sorted(paths)
    ]
    tier_counts = {
        str(tier): sum(model.get("tier") == tier for model in models)
        for tier, _, _ in TIERS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "discovery": "recursive" if recursive else "root_only",
        "sequential_inference_only": True,
        "model_count": len(models),
        "loadable_count": sum(model["status"] == "loadable" for model in models),
        "invalid_count": sum(model["status"] == "invalid_onnx" for model in models),
        "tier_counts": tier_counts,
        "models": models,
    }


def write_corpus_manifest(path: str, manifest: Dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic local flatbuffer_direct ONNX corpus manifest."
    )
    parser.add_argument("--root_dir", default=os.getcwd())
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--skip_sha256",
        action="store_true",
        help="Skip content hashes for a quick inventory pass.",
    )
    args = parser.parse_args()
    manifest = build_corpus_manifest(
        root_dir=args.root_dir,
        recursive=bool(args.recursive),
        hash_contents=not bool(args.skip_sha256),
    )
    write_corpus_manifest(args.output, manifest)
    print(
        f"Wrote {args.output}: models={manifest['model_count']} "
        f"loadable={manifest['loadable_count']} invalid={manifest['invalid_count']}"
    )


if __name__ == "__main__":
    main()
