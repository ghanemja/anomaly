"""V-JEPA 2 as an action-free latent world model for online failure monitoring.

For each window ending at frame t we:

1. encode the *context* frames alone (so no token has attended to the future;
   Hugging Face's ``VJEPA2Model.forward`` encodes the whole clip before masking,
   which would leak the target frames into the context);
2. ask the pretrained predictor for the latent tokens of the next
   ``target`` frames;
3. encode the full window to get the actual tokens of those frames;
4. score the L1 gap between predicted and layer-normed actual tokens (V-JEPA 2's
   own training loss).

The score is assigned to frame t, the last frame of the window, so it is
causal. ``mean`` averages the gap over all patches; ``max`` takes the worst
patch, which suits small local events such as a flap opening. The per-patch
error map comes back too, for localization figures.

Costs: one ViT-L pass over ``context`` frames plus one over the full window.
About 10 s per window on a 4-core CPU, a few windows per second on one GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


@dataclass
class VJEPA2Config:
    model: str = DEFAULT_MODEL
    frame_stride: int = 2  # 30 fps source -> 15 fps model input
    context: int = 8  # model frames of context (0.53 s at stride 2)
    target: int = 4  # model frames predicted (0.27 s)
    hop: int = 2  # model frames between windows (one tubelet)
    device: str = "cpu"
    dtype: str = "float32"  # use "bfloat16" on GPU


class VJEPA2Monitor:
    def __init__(self, cfg: VJEPA2Config = VJEPA2Config()):
        import torch
        from transformers import AutoModel

        self.cfg, self.torch = cfg, torch
        self.model = AutoModel.from_pretrained(cfg.model, dtype=getattr(torch, cfg.dtype)).to(cfg.device).eval()
        mc = self.model.config
        self.size, self.tubelet = mc.crop_size, mc.tubelet_size
        self.grid = mc.crop_size // mc.patch_size
        assert cfg.context % self.tubelet == 0 and cfg.target % self.tubelet == 0

    def _prep(self, frames: np.ndarray):
        """uint8 (T, H, W, 3) -> float (1, T, 3, S, S), squashed to the model's square input."""
        torch = self.torch
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True, align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN)[None, :, None, None]
        std = torch.tensor(IMAGENET_STD)[None, :, None, None]
        x = (x - mean) / std
        return x[None].to(self.cfg.device, dtype=getattr(torch, self.cfg.dtype))

    def window_error(self, window: np.ndarray) -> np.ndarray:
        """Per-patch L1 error, shape ``(target // tubelet, grid, grid)``."""
        torch, cfg = self.torch, self.cfg
        x = self._prep(window)
        per_t = self.grid * self.grid
        n_ctx = cfg.context // self.tubelet * per_t
        n_all = (cfg.context + cfg.target) // self.tubelet * per_t
        with torch.inference_mode():
            ctx = self.model.encoder(pixel_values_videos=x[:, : cfg.context]).last_hidden_state
            full = self.model.encoder(pixel_values_videos=x).last_hidden_state
            cm = [torch.arange(n_ctx, device=x.device)[None]]
            tm = [torch.arange(n_ctx, n_all, device=x.device)[None]]
            pred = self.model.predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
            # V-JEPA 2 trains the predictor against layer_norm(target tokens), without affine
            tgt = torch.nn.functional.layer_norm(full[:, n_ctx:n_all].float(), (full.size(-1),))
            err = (pred.float() - tgt).abs().mean(-1)
        return err.reshape(-1, self.grid, self.grid).cpu().numpy()

    def score_clip(self, frames: np.ndarray, keep_maps: bool = False) -> dict[str, np.ndarray]:
        """Per-source-frame scores for one clip of source-rate frames.

        Returns ``mean`` and ``max`` arrays of length ``len(frames)``, NaN where
        no window ends (warm-up, and frames between hops), plus ``maps`` with
        one error map per window if ``keep_maps``.
        """
        cfg = self.cfg
        x = frames[:: cfg.frame_stride]
        W = cfg.context + cfg.target
        T = len(frames)
        out = {"mean": np.full(T, np.nan), "max": np.full(T, np.nan)}
        maps = []
        for end in range(W, len(x) + 1, cfg.hop):
            e = self.window_error(x[end - W : end])
            t = (end - 1) * cfg.frame_stride
            out["mean"][t] = e.mean()
            out["max"][t] = e.max()
            if keep_maps:
                maps.append((t, e))
        if keep_maps:
            out["maps"] = maps
        return out


def hold_last(s: np.ndarray) -> np.ndarray:
    """Carry each window score forward to the next window (still causal).

    Lets sparse world-model scores share the per-frame evaluation code.
    """
    out = s.copy()
    last = np.nan
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    return out
