"""Stage A/B: frozen pretrained embeddings of the 6,400 original slices (no labels involved).

For each backbone: embedding = mean of the image and its horizontal flip (flip TTA), computed
from the shared preprocessing cache (224 px). Optional 2.5D mode: the RGB channels are the
neighbouring slices (k-1, k, k+1) of the same patient, ordered by slice position (flagged:
uses slice order). Rows follow cv_core.slices() order. Output: output/cv/emb/<name>.npy

Usage: python src/imgProcessing/embeddings.py [--backbones all] [--mode 2d|25d|both]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402
from build_cache import open_cache  # noqa: E402

EMB_DIR = cv.CV_DIR / "emb"
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def load_backbone(name: str):
    """Returns (model producing (B, D) features, input size, (mean, std))."""
    from torchvision import models as tv
    if name == "resnet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity(); return m, 224, IMNET
    if name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier = torch.nn.Identity(); return m, 224, IMNET
    if name == "efficientnet_b3":
        m = tv.efficientnet_b3(weights=tv.EfficientNet_B3_Weights.IMAGENET1K_V1); m.classifier = torch.nn.Identity(); return m, 300, IMNET
    if name == "convnext_tiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = torch.nn.Identity(); return m, 224, IMNET
    if name in ("dinov2_vits14", "dinov2_vitb14"):
        m = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True); return m, 224, IMNET
    if name == "biomedclip":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

        class Enc(torch.nn.Module):
            def __init__(self, mm):
                super().__init__(); self.mm = mm

            def forward(self, x):
                return self.mm.encode_image(x)
        return Enc(model), 224, CLIP
    raise ValueError(name)


ALL = ["resnet50", "efficientnet_b0", "efficientnet_b3", "convnext_tiny", "dinov2_vits14", "dinov2_vitb14", "biomedclip"]


def neighbour_index(df) -> np.ndarray:
    """(N, 3) row indices of slice k-1, k, k+1 of the same patient (edges replicated)."""
    nb = np.zeros((len(df), 3), dtype=np.int64)
    for _, part in df.groupby("group"):
        order = part.sort_values("slice_pos").index.to_numpy()
        for j, i in enumerate(order):
            nb[i] = [order[max(j - 1, 0)], i, order[min(j + 1, len(order) - 1)]]
    return nb


@torch.no_grad()
def extract(name: str, mode: str, df, arr, dev, batch: int = 32) -> np.ndarray:
    model, size, (mean, std) = load_backbone(name)
    model.eval().to(dev)
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    rows = df["cache_row"].to_numpy()
    nb = neighbour_index(df) if mode == "25d" else None
    out = []
    for i in range(0, len(df), batch):
        idx = np.arange(i, min(i + batch, len(df)))
        if mode == "2d":
            x = torch.from_numpy(np.stack([arr[r] for r in rows[idx]])).to(dev).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            x = torch.from_numpy(np.stack([np.stack([arr[rows[j]] for j in nb[k]]) for k in idx])).to(dev).float().div(255)
        if size != 224:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        x = (x - mean_t) / std_t
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            f = model(x).float() + model(torch.flip(x, dims=[3])).float()
        out.append((f / 2).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbones", default="all")
    ap.add_argument("--mode", default="2d", choices=["2d", "25d", "both"])
    a = ap.parse_args()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    df = cv.slices()
    arr, _ = open_cache(224)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = ALL if a.backbones == "all" else a.backbones.split(",")
    modes = ["2d", "25d"] if a.mode == "both" else [a.mode]
    status_p = EMB_DIR / "status.json"
    status = json.loads(status_p.read_text()) if status_p.exists() else {}
    for mode in modes:
        for n in names:
            key = n if mode == "2d" else f"{n}_25d"
            p = EMB_DIR / f"{key}.npy"
            if p.exists():
                print(f"[emb] skip {key} (exists)", flush=True)
                continue
            t0 = time.time()
            try:
                e = extract(n, mode, df, arr, dev)
                np.save(p, e)
                status[key] = {"ok": True, "dim": int(e.shape[1]), "seconds": round(time.time() - t0, 1)}
                print(f"[emb] {key}: {e.shape} in {time.time() - t0:.0f}s", flush=True)
            except Exception as err:
                status[key] = {"ok": False, "error": f"{type(err).__name__}: {str(err)[:300]}"}
                print(f"[emb] SKIP {key}: {status[key]['error']}", flush=True)
            status_p.write_text(json.dumps(status, indent=2))
    # RadImageNet: no official ungated programmatic download for these frameworks -> skipped (logged)
    status.setdefault("radimagenet", {"ok": False, "error": "skipped: no official ungated torch/timm weight download"})
    status_p.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
