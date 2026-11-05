"""Per-frame embeddings from a frozen image encoder (CPU-friendly).

The default encoder is an ImageNet ResNet-18 applied to the 320x140 transcodes.
Each frame gets the concatenation of global-average and global-max pooled
layer4 features (1024-d): average pooling summarises the scene, max pooling
keeps small salient changes such as an opened flap.

These features feed the CPU baselines. The world-model monitors in
``armwm.worldmodels`` use their own video encoders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FrameEncoder:
    def __init__(self, arch: str = "resnet18", device: str = "cpu", threads: int | None = None):
        import torch
        import torchvision

        if threads:
            torch.set_num_threads(threads)
        weights = torchvision.models.get_model_weights(arch).DEFAULT
        net = torchvision.models.get_model(arch, weights=weights)
        self.body = torch.nn.Sequential(*list(net.children())[:-2]).eval().to(device)
        self.device = device
        self.torch = torch

    def __call__(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        """``(T, H, W, 3)`` uint8 -> ``(T, 1024)`` float16."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), batch):
                x = (frames[i : i + batch].astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
                f = self.body(x)
                out.append(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1).cpu().numpy())
        return np.concatenate(out).astype(np.float16)


def save_features(path: str | Path, feats: dict[str, np.ndarray], stride: int) -> None:
    np.savez_compressed(path, stride=stride, **{f"f_{k}": v for k, v in feats.items()})


def load_features(path: str | Path) -> tuple[dict[str, np.ndarray], int]:
    z = np.load(path)
    return {k[2:]: z[k] for k in z.files if k.startswith("f_")}, int(z["stride"])
