"""Export embedding model and reranker to ONNX for optimized CPU inference.

Exports:
  1. nomic-embed-text-v1.5 → data/models/nomic-onnx/
  2. ms-marco-MiniLM-L-6-v2 → data/models/reranker-onnx/

Usage:
    python -m src.onnx_export [--embed-only | --reranker-only]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


MODELS_DIR = Path("data/models")


def export_embedding_model(output_dir: Path = MODELS_DIR / "nomic-onnx"):
    """Export nomic-embed-text-v1.5 to ONNX."""
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "nomic-ai/nomic-embed-text-v1.5"

    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Get the underlying transformer model
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    # Set to eval mode
    transformer.eval()

    # Create dummy input
    dummy_text = "search_document: This is a test book about Python programming"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        transformer,
        (input_ids, attention_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Save tokenizer alongside
    tokenizer.save_pretrained(str(output_dir))

    # Verify with onnxruntime
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)

    # Compare with PyTorch output
    with torch.no_grad():
        pt_output = transformer(input_ids, attention_mask)
        if hasattr(pt_output, "last_hidden_state"):
            pt_np = pt_output.last_hidden_state.numpy()
        else:
            pt_np = pt_output[0].numpy()

    diff = np.abs(pt_np - ort_outputs[0]).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")

    # File size
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"ONNX model size: {size_mb:.1f} MB")
    print(f"Saved to: {output_dir}")
    return output_dir


def export_reranker(output_dir: Path = MODELS_DIR / "reranker-onnx"):
    """Export cross-encoder/ms-marco-MiniLM-L-6-v2 to ONNX."""
    from transformers import AutoModelForSequenceClassification

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Create dummy input (query + document pair)
    dummy_query = "romance set in Scotland"
    dummy_doc = "A story of love in the Scottish Highlands"
    inputs = tokenizer(
        dummy_query,
        dummy_doc,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids")

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    if token_type_ids is not None:
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    else:
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))

    # Verify with onnxruntime
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    if token_type_ids is not None:
        ort_inputs["token_type_ids"] = token_type_ids.numpy()

    ort_outputs = session.run(None, ort_inputs)

    with torch.no_grad():
        pt_output = model(input_ids, attention_mask, token_type_ids=token_type_ids)
        pt_np = pt_output.logits.numpy()

    diff = np.abs(pt_np - ort_outputs[0]).max()
    print(f"Max diff PyTorch vs ONNX: {diff:.6f}")

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"ONNX model size: {size_mb:.1f} MB")
    print(f"Saved to: {output_dir}")
    return output_dir


def benchmark(model_type: str = "reranker"):
    """Benchmark PyTorch vs ONNX inference speed."""
    import onnxruntime as ort

    if model_type == "reranker":
        from transformers import AutoModelForSequenceClassification

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        onnx_dir = MODELS_DIR / "reranker-onnx"
        tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

        # Generate test pairs (simulating 25 candidates)
        query = "romance set in Scotland"
        docs = [f"This is test document number {i} about various topics" for i in range(25)]
        pairs = [(query, doc) for doc in docs]

        # Tokenize
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # PyTorch benchmark
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        # Warmup
        with torch.no_grad():
            _ = model(**encoded)

        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = model(**encoded)
        pt_time = (time.time() - start) / 10 * 1000

        # ONNX benchmark
        session = ort.InferenceSession(str(onnx_dir / "model.onnx"))
        ort_inputs = {k: v.numpy() for k, v in encoded.items()}

        # Warmup
        _ = session.run(None, ort_inputs)

        start = time.time()
        for _ in range(10):
            _ = session.run(None, ort_inputs)
        onnx_time = (time.time() - start) / 10 * 1000

        print(f"\n{'='*50}")
        print(f"Reranker Benchmark (25 candidates)")
        print(f"{'='*50}")
        print(f"PyTorch:  {pt_time:.1f} ms")
        print(f"ONNX:     {onnx_time:.1f} ms")
        print(f"Speedup:  {pt_time/onnx_time:.2f}x")

    elif model_type == "embed":
        from sentence_transformers import SentenceTransformer

        onnx_dir = MODELS_DIR / "nomic-onnx"
        tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

        # Test texts
        texts = [f"search_query: test query about topic {i}" for i in range(10)]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # PyTorch benchmark
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

        start = time.time()
        for _ in range(5):
            _ = model.encode(texts, normalize_embeddings=False)
        pt_time = (time.time() - start) / 5 * 1000

        # ONNX benchmark
        session = ort.InferenceSession(str(onnx_dir / "model.onnx"))
        ort_inputs = {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
        }

        # Warmup
        _ = session.run(None, ort_inputs)

        start = time.time()
        for _ in range(5):
            _ = session.run(None, ort_inputs)
        onnx_time = (time.time() - start) / 5 * 1000

        print(f"\n{'='*50}")
        print(f"Embedding Benchmark (10 texts)")
        print(f"{'='*50}")
        print(f"PyTorch (sentence-transformers):  {pt_time:.1f} ms")
        print(f"ONNX (raw inference only):        {onnx_time:.1f} ms")
        print(f"Speedup:  {pt_time/onnx_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--embed-only", action="store_true", help="Only export embedding model")
    parser.add_argument("--reranker-only", action="store_true", help="Only export reranker")
    parser.add_argument("--benchmark", choices=["embed", "reranker"], help="Run benchmark")
    args = parser.parse_args()

    if args.benchmark:
        benchmark(args.benchmark)
    elif args.embed_only:
        export_embedding_model()
    elif args.reranker_only:
        export_reranker()
    else:
        export_reranker()
        print("\n" + "=" * 50 + "\n")
        export_embedding_model()
