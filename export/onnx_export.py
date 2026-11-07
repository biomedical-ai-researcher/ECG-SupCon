"""
ONNX Export Script — EXP006
Exports the frozen 1D-MobileViT checkpoint to a single self-contained
ONNX file (opset 18) suitable for ONNX Runtime Android deployment.

Usage:
    python onnx_export.py \
        --checkpoint path/to/EXP006_v1_FROZEN_best_macro_f1.pt \
        --output     EXP006_embedded.onnx

Requirements:
    pip install torch onnx onnxruntime numpy
"""

import argparse
import hashlib
import numpy as np
import torch
import onnx
import onnxruntime as ort

from mobilevit1d import MobileViT1d, MobileViT1dInference


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export(checkpoint_path: str, output_path: str):
    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone = MobileViT1d(num_classes=4, proj_dim=128)
    backbone.load_state_dict(ckpt["model_state"])
    backbone.eval()

    model = MobileViT1dInference(backbone)
    model.eval()

    # ── Export ───────────────────────────────────────────────────────
    dummy = torch.zeros(1, 1, 260)
    print("Exporting to ONNX (opset 18)…")
    torch.onnx.export(
        model, dummy,
        output_path,
        opset_version=18,
        input_names=["ecg_input"],
        output_names=["logits"],
        dynamic_axes={
            "ecg_input": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    # ── Inline external data (single-file) ───────────────────────────
    onnx_model = onnx.load(output_path)
    onnx.save_model(
        onnx_model, output_path,
        save_as_external_data=False,
    )
    print(f"Saved: {output_path}  ({_human(output_path)})")
    print(f"SHA-256: {sha256(output_path)}")

    # ── Verify with ONNX Runtime ─────────────────────────────────────
    print("\nVerifying with ONNX Runtime…")
    sess = ort.InferenceSession(
        output_path,
        providers=["CPUExecutionProvider"],
    )
    dummy_np = np.zeros((1, 1, 260), dtype=np.float32)
    out = sess.run(["logits"], {"ecg_input": dummy_np})
    print(f"Output shape: {out[0].shape}  ✓")

    # ── Latency benchmark ────────────────────────────────────────────
    print("\nLatency benchmark (1 000 inferences, single-threaded CPU)…")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_st = ort.InferenceSession(
        output_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    import time
    for _ in range(10):          # warm-up
        sess_st.run(["logits"], {"ecg_input": dummy_np})
    t0 = time.perf_counter()
    for _ in range(1000):
        sess_st.run(["logits"], {"ecg_input": dummy_np})
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Mean latency: {elapsed / 1000:.3f} ms/inference")
    print("\nExport complete ✓")


def _human(path: str) -> str:
    import os
    size = os.path.getsize(path)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="EXP006_v1_FROZEN_best_macro_f1.pt",
        help="Path to the frozen PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        default="EXP006_embedded.onnx",
        help="Output ONNX file path",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)
