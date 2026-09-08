# -*- coding:utf-8 -*-
"""自動判斷平台與 backend 驅動選擇器"""

import platform
import sys
from typing import Tuple, Any


def detect_best_backend(user_device: Any = None) -> Tuple[str, Any]:
    """檢測目前硬體與安裝環境，傳回最佳推理 backend ("mlx", "torch_mps", "torch_cuda", "torch_cpu")"""
    # 若使用者顯式指定 mlx
    if user_device == "mlx":
        return "mlx", None

    # 自動判斷是否為 Apple Silicon (macOS + ARM64)
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    if is_apple_silicon and (user_device is None or user_device in ["auto", "mps"]):
        # 檢查是否安裝了 MLX
        try:
            import mlx.core as mx
            import mlx.nn as nn

            return "mlx", "mlx"
        except ImportError:
            # 若未安裝 MLX，退回至 PyTorch MPS
            try:
                import torch

                if torch.backends.mps.is_available():
                    return "torch", torch.device("mps")
            except ImportError:
                pass

    # 非 Mac 或無 MLX 時，使用標準 PyTorch 設備判斷
    import torch

    if user_device is None or user_device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    elif isinstance(user_device, int):
        dev = torch.device(f"cuda:{user_device}" if user_device >= 0 else "cpu")
    elif isinstance(user_device, str):
        dev = torch.device(user_device)
    else:
        dev = user_device

    backend_type = (
        "torch_cuda"
        if dev.type == "cuda"
        else "torch_mps" if dev.type == "mps" else "torch_cpu"
    )
    return "torch", dev