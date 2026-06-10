"""
Export an MMPose RTMPose config + checkpoint to ONNX for ``OnnxPoseEstimator`` / benchmarking.

Requires a working OpenMMLab stack (``mmengine``, ``mmpose``, ``torch``, ``onnx``).
Full top-down pose models often need custom inputs (``DataSample``); this script tries a
straight ``torch.onnx.export`` on tensor input — if it fails, use `MMDeploy <https://mmdeploy.readthedocs.io/>`__
or MMPose deployment docs for your checkpoint.

Example::

    python -m fightsafe_ai.pose.export_rtmpose_onnx \\
        --config path/to/rtmpose-m_xxx.py \\
        --checkpoint path/to.pth \\
        -o rtmpose.onnx --height 256 --width 192

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_rtmpose_to_onnx(
    config: Path,
    checkpoint: Path,
    output: Path,
    *,
    height: int = 256,
    width: int = 192,
    opset: int = 17,
    device: str = "cpu",
) -> int:
    cfg_path = config.expanduser().resolve()
    ckpt = checkpoint.expanduser().resolve()
    out = output.expanduser().resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from mmengine.config import Config
        from mmengine.runner import load_checkpoint
    except ImportError as exc:
        print("Requires torch and mmengine: pip install torch mmengine", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    try:
        from mmpose.registry import MODELS
        from mmpose.utils import register_all_modules
    except ImportError as exc:
        print("Requires mmpose: pip install mmpose", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(str(cfg_path))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(ckpt), map_location=device)
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    dummy = torch.randn(1, 3, height, width, device=dev)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["output"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except Exception as exc:
        print(
            "torch.onnx.export failed (many MMPose models need wrapped inputs / MMDeploy). "
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 3

    print(f"Wrote ONNX: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MMPose RTMPose weights to ONNX.")
    parser.add_argument("--config", type=Path, required=True, help="MMEngine config .py path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model .pth checkpoint.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .onnx path.")
    parser.add_argument("--height", type=int, default=256, help="Dummy input height (model input).")
    parser.add_argument("--width", type=int, default=192, help="Dummy input width.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to load weights before export (cpu avoids GPU OOM during export).",
    )
    args = parser.parse_args()
    return export_rtmpose_to_onnx(
        args.config,
        args.checkpoint,
        args.output,
        height=args.height,
        width=args.width,
        opset=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
