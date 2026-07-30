"""
Benchmark Torch (RTMPose / MMPose) vs ONNX Runtime inference.

Prints FPS, mean / median latency (ms), and device/provider info. Requires optional deps:

* Torch path: ``mmpose``, ``torch``, CUDA for GPU (install CUDA-enabled PyTorch).
* ONNX path: ``onnxruntime-gpu`` for NVIDIA inference on CUDA EP.

Example::

    PYTHONPATH=src python -m fightsafe_ai.pose.bench_torch_onnx \\
        --frames-dir ./frames --onnx-model ./rtmpose.onnx \\
        --device cuda:0 --fp16

"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fightsafe_ai.pose.backends.device_runtime import configure_cuda_inference
from fightsafe_ai.pose.backends.onnx_estimator import OnnxPoseEstimator
from fightsafe_ai.pose.backends.rtmpose_backend import RTMPoseBackend
from fightsafe_ai.utils.sorting import natural_sort_paths


def _print_torch_cuda_banner(device: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            print("torch: CUDA not available (torch.cuda.is_available() is False).")
            return
        configure_cuda_inference()
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        print(f"torch: CUDA device {idx} — {name}")
        print(f"torch: TF32 matmul={getattr(torch.backends.cuda.matmul, 'allow_tf32', 'n/a')}")
    except ImportError:
        print("torch: not installed.")


def _print_onnx_banner(est: OnnxPoseEstimator) -> None:
    sess = est._session
    if sess is None:
        print("onnxruntime: no session (missing model or onnxruntime).")
        return
    print(f"onnxruntime: providers={sess.get_providers()}")
    try:
        import onnxruntime as ort

        print(f"onnxruntime: package version={getattr(ort, '__version__', '?')}")
    except ImportError:
        pass


def _load_images(frames_dir: Path, patterns: tuple[str, ...], limit: int) -> list[np.ndarray]:
    frames_dir = frames_dir.expanduser().resolve()
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(frames_dir.glob(pat))
    paths = natural_sort_paths([p for p in paths if p.is_file()])[:limit]
    out: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            out.append(im)
    return out


def _bench_torch(
    frames: list[np.ndarray],
    *,
    pose2d: str,
    device: str,
    fp16: bool,
    warmup: int,
) -> tuple[float, float, float]:
    configure_cuda_inference()
    est = RTMPoseBackend(pose2d=pose2d, device=device, use_fp16=fp16)
    if est._inferencer is None:
        print("Torch RTMPose unavailable (mmpose/torch missing or init failed).", file=sys.stderr)
        return 0.0, 0.0, 0.0

    for _ in range(min(warmup, len(frames))):
        est.estimate_frame(frames[_ % len(frames)])

    latencies: list[float] = []
    t0 = time.perf_counter()
    for im in frames:
        t1 = time.perf_counter()
        est.estimate_frame(im)
        latencies.append((time.perf_counter() - t1) * 1000.0)
    wall = time.perf_counter() - t0
    mean_lat = float(statistics.mean(latencies)) if latencies else 0.0
    med_lat = float(statistics.median(latencies)) if latencies else 0.0
    fps = len(frames) / wall if wall > 0 else 0.0
    return fps, mean_lat, med_lat


def _bench_onnx(
    frames: list[np.ndarray],
    *,
    onnx_path: Path,
    fp16: bool,
    cuda_device_id: int,
    warmup: int,
) -> tuple[float, float, float]:
    est = OnnxPoseEstimator(
        model_path=onnx_path,
        use_fp16=fp16,
        cuda_device_id=cuda_device_id,
        prefer_cuda=True,
    )
    _print_onnx_banner(est)
    if est._session is None:
        print(f"ONNX session missing for {onnx_path}", file=sys.stderr)
        return 0.0, 0.0, 0.0

    for _ in range(min(warmup, len(frames))):
        est.predict(frames[_ % len(frames)])

    latencies: list[float] = []
    t0 = time.perf_counter()
    for im in frames:
        t1 = time.perf_counter()
        est.predict(im)
        latencies.append((time.perf_counter() - t1) * 1000.0)
    wall = time.perf_counter() - t0
    mean_lat = float(statistics.mean(latencies)) if latencies else 0.0
    med_lat = float(statistics.median(latencies)) if latencies else 0.0
    fps = len(frames) / wall if wall > 0 else 0.0
    return fps, mean_lat, med_lat


def run_benchmark(
    *,
    frames_dir: Path,
    globs: tuple[str, ...],
    limit: int,
    warmup: int,
    pose2d: str,
    device: str,
    fp16: bool,
    onnx_model: Path,
    onnx_device_id: int,
) -> dict[str, Any]:
    frames = _load_images(frames_dir, globs, limit)
    if not frames:
        return {"error": "no_images", "frames": 0}

    _print_torch_cuda_banner(device)
    tfps, tmean, tmed = _bench_torch(
        frames,
        pose2d=pose2d,
        device=device,
        fp16=fp16,
        warmup=warmup,
    )
    print()
    ofps, omean, omed = _bench_onnx(
        frames,
        onnx_path=onnx_model.expanduser().resolve(),
        fp16=fp16,
        cuda_device_id=onnx_device_id,
        warmup=warmup,
    )

    out = {
        "frames": len(frames),
        "warmup": warmup,
        "fp16": fp16,
        "torch": {"fps": tfps, "latency_mean_ms": tmean, "latency_median_ms": tmed},
        "onnx": {"fps": ofps, "latency_mean_ms": omean, "latency_median_ms": omed},
    }
    print_summary(out)
    return out


def print_summary(result: dict[str, Any]) -> None:
    if result.get("error") == "no_images":
        return
    print()
    print("=== Summary ===")
    print(f"frames={result['frames']} warmup={result['warmup']} fp16={result['fp16']}")
    t, o = result["torch"], result["onnx"]
    print(
        f"torch   FPS={t['fps']:.2f}  latency_mean_ms={t['latency_mean_ms']:.3f}  "
        f"latency_median_ms={t['latency_median_ms']:.3f}"
    )
    print(
        f"onnx    FPS={o['fps']:.2f}  latency_mean_ms={o['latency_mean_ms']:.3f}  "
        f"latency_median_ms={o['latency_median_ms']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RTMPose Torch vs ONNX Runtime.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of frame images.")
    parser.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob pattern (repeatable). Default: *.jpg *.jpeg *.png",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max frames to load.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend.")
    parser.add_argument(
        "--pose2d",
        default="rtmpose-m_8xb256-210e_coco-256x192",
        help="MMPose pose2d model id for Torch.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device (cuda:0, cpu, …).")
    parser.add_argument("--fp16", action="store_true", help="Torch AMP FP16 + ONNX float16 inputs.")
    parser.add_argument("--onnx-model", type=Path, required=True, help="Path to ONNX model file.")
    parser.add_argument("--onnx-device-id", type=int, default=0, help="CUDA device id for ORT.")
    args = parser.parse_args()

    globs = tuple(args.glob) if args.glob else ("*.jpg", "*.jpeg", "*.png")
    result = run_benchmark(
        frames_dir=args.frames_dir,
        globs=globs,
        limit=args.limit,
        warmup=args.warmup,
        pose2d=args.pose2d,
        device=args.device,
        fp16=args.fp16,
        onnx_model=args.onnx_model,
        onnx_device_id=args.onnx_device_id,
    )
    if result.get("error") == "no_images":
        print("No images loaded.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
