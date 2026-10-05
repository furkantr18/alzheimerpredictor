"""Grad-CAM for the deep MRI models ("what does the model look at").

* gradcam(model, arch, x)          -> (H, W) heatmap in [0, 1] for the predicted (or given) class
* overlay_png(gray_u8, heatmap)    -> PNG bytes (jet overlay), used by the API /predict-mri/explain
* attention_stats(gray_u8, heat)   -> share of the CAM mass inside the brain mask, in the outer
                                      ring of the brain (cortex/skull-strip edge) and in the background
* main(): gallery of correct and wrong predictions per class on the VALIDATION split
  (not test) + aggregate attention statistics.

Usage: python src/imgProcessing/mri_gradcam.py --run <dl run name> [--per-class 4]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402


def target_layer(model, arch: str):
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    raise ValueError(arch)


def gradcam(model, arch: str, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """x: (1, 3, H, W) normalised. Returns (heatmap HxW in [0,1], class used, softmax probs)."""
    acts, grads = {}, {}
    layer = target_layer(model, arch)
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x).float()
            probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            model.zero_grad(set_to_none=True)
            logits[0, c].backward()
        a, g = acts["a"].detach().float(), grads["g"].detach().float()
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / cam.max()).cpu().numpy() if float(cam.max()) > 0 else cam.cpu().numpy()
    finally:
        h1.remove(); h2.remove()
    return cam, c, probs


def brain_masks(gray_u8: np.ndarray, thr: int = 10) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    mask = (gray_u8 > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # fill holes (ventricles are dark but inside the brain)
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8), (0, 0), 1)
    mask = mask | (1 - flood)
    inner = cv2.erode(mask, np.ones((15, 15), np.uint8))
    return mask.astype(bool), inner.astype(bool)


def attention_stats(gray_u8: np.ndarray, cam: np.ndarray) -> dict:
    mask, inner = brain_masks(gray_u8)
    tot = float(cam.sum()) + 1e-9
    return {"in_brain": float(cam[mask].sum() / tot), "brain_edge_ring": float(cam[mask & ~inner].sum() / tot),
            "background": float(cam[~mask].sum() / tot), "brain_area_share": float(mask.mean()),
            "edge_ring_area_share": float((mask & ~inner).mean())}


def overlay_png(gray_u8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> bytes:
    import cv2
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(heat, alpha, base, 1 - alpha, 0)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def main() -> None:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from build_cache import open_cache
    from mri_data import load_split, save_json
    from mri_dl_trainer import build_model
    from mri_preprocess import to_tensor

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--stats-n", type=int, default=600, help="validation images for aggregate attention stats")
    args = ap.parse_args()

    run_dir = C.MODELS_DIR / "dl_runs" / args.run
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["arch"], len(meta["class_names"]), meta["config"]["dropout"], pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.to(dev).eval()
    arr, _ = open_cache(224)
    df = load_split(meta["config"]["split"])
    va = df[df["split"] == "val"].reset_index(drop=True)

    def run_one(r):
        g = np.asarray(arr[int(r.cache_row)])
        x = to_tensor(g).unsqueeze(0).to(dev)
        if meta["image_size"] != 224:
            x = F.interpolate(x, size=(meta["image_size"],) * 2, mode="bilinear", align_corners=False, antialias=True)
            g = np.asarray(F.interpolate(torch.from_numpy(g)[None, None].float(), size=(meta["image_size"],) * 2, mode="bilinear")[0, 0].clamp(0, 255).byte())
        cam, c, p = gradcam(model, meta["arch"], x)
        return g, cam, c, p

    rng = np.random.default_rng(0)
    sample = va.iloc[rng.permutation(len(va))[: args.stats_n]]
    stats_rows = []
    for r in sample.itertuples():
        g, cam, c, p = run_one(r)
        s = attention_stats(g, cam)
        s.update(true=r.class_name, pred=meta["class_names"][c], correct=meta["class_names"][c] == r.class_name)
        stats_rows.append(s)
    st = pd.DataFrame(stats_rows)
    agg = {"n": int(len(st)),
           "mean_in_brain": float(st["in_brain"].mean()), "mean_background": float(st["background"].mean()),
           "mean_brain_edge_ring": float(st["brain_edge_ring"].mean()), "mean_edge_ring_area_share": float(st["edge_ring_area_share"].mean()),
           "share_images_with_>20pct_background": float((st["background"] > 0.2).mean()),
           "by_true_class": st.groupby("true")[["in_brain", "brain_edge_ring", "background"]].mean().round(3).to_dict(orient="index"),
           "by_correct": st.groupby("correct")[["in_brain", "brain_edge_ring", "background"]].mean().round(3).to_dict(orient="index")}
    out_dir = C.REPORTS_DIR / "gradcam" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(agg, out_dir / "attention_stats.json")
    st.to_csv(out_dir / "attention_per_image.csv", index=False)

    # gallery: per class, up to N correct and N wrong
    st_full = sample.reset_index(drop=True).join(st[["pred", "correct"]])
    classes = [c for c in meta["class_names"] if c in set(st_full["class_name"])]
    fig, axes = plt.subplots(len(classes) * 2, args.per_class, figsize=(args.per_class * 2.2, len(classes) * 2 * 2.3))
    for i, cls in enumerate(classes):
        for j, ok in enumerate([True, False]):
            pick = st_full[(st_full["class_name"] == cls) & (st_full["correct"] == ok)].head(args.per_class)
            for k in range(args.per_class):
                ax = axes[2 * i + j, k]; ax.axis("off")
                if k < len(pick):
                    r = pick.iloc[k]
                    g, cam, c, p = run_one(r)
                    ax.imshow(g, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.4)
                    ax.set_title(f"true {cls[:8]}\npred {meta['class_names'][c][:8]} {p[c]:.2f}", fontsize=6,
                                 color="green" if ok else "red")
            axes[2 * i + j, 0].text(-0.1, 0.5, f"{cls}\n{'correct' if ok else 'wrong'}", transform=axes[2 * i + j, 0].transAxes,
                                    ha="right", va="center", fontsize=7)
    fig.suptitle(f"Grad-CAM, {args.run}, validation originals", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "gallery.png", dpi=110); plt.close(fig)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
