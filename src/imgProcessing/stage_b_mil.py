"""Stage B/D: patient-level Multiple Instance Learning on frozen slice embeddings, nested CV.

Bag = all slices of one patient group (32 or 64), padded with a mask. Pooling: gated attention
(Ilse et al. 2018), mean or max. Grid (16 per pooling, pre-declared): hidden {64,128} x dropout
{0.25,0.5} x slice-dropout {0,0.3} x weight decay {1e-4,1e-2}. Inner grouped 3-fold CV (PCA fit on
inner-train only) picks the config by inner patient macro-F1 and gives the epoch count (median of
inner best epochs, early stopping on inner-val loss, patience 20, <= 200 epochs). Final model: refit
on the outer-training part with 3 seeds, probabilities averaged. Optional: CORAL ordinal head,
temperature scaling fit on inner out-of-fold logits (for ECE). Attention weights saved.

Usage: python src/imgProcessing/stage_b_mil.py --emb dinov2_vitb14 --pool attn --task T3 [--loss ce|coral] [--calibrate]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_core as cv  # noqa: E402

GRID = [dict(hidden=h, dropout=d, slice_drop=s, wd=w) for h, d, s, w in itertools.product((64, 128), (0.25, 0.5), (0.0, 0.3), (1e-4, 1e-2))]
MAX_EPOCHS, PATIENCE, LR = 200, 20, 1e-3


class MIL(nn.Module):
    def __init__(self, d_in, hidden, n_out, dropout, pool):
        super().__init__()
        self.pool = pool
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        if pool == "attn":
            self.V = nn.Linear(hidden, hidden // 2)
            self.U = nn.Linear(hidden, hidden // 2)
            self.w = nn.Linear(hidden // 2, 1)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x, mask):  # x (B, N, D), mask (B, N) bool
        h = self.enc(x)
        if self.pool == "attn":
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
            a = a.masked_fill(~mask, -1e4)
            att = torch.softmax(a, 1)
            z = (att.unsqueeze(-1) * h).sum(1)
        elif self.pool == "mean":
            att = mask.float() / mask.float().sum(1, keepdim=True)
            z = (att.unsqueeze(-1) * h).sum(1)
        else:
            z = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            att = None
        return self.out(z), att


def coral_probs(logits):  # logits (B, K-1): P(y > k)
    s = torch.sigmoid(logits)
    K = s.shape[1] + 1
    p = torch.zeros(s.shape[0], K, device=s.device)
    p[:, 0] = 1 - s[:, 0]
    for k in range(1, K - 1):
        p[:, k] = s[:, k - 1] - s[:, k]
    p[:, K - 1] = s[:, K - 2]
    return p.clamp_min(1e-6) / p.clamp_min(1e-6).sum(1, keepdim=True)


def to_probs(logits, loss, T=1.0):
    return coral_probs(logits / T) if loss == "coral" else torch.softmax(logits / T, 1)


def make_bags(Z, groups, slice_pos, ids):
    """Z (N, D) -> (B, Nmax, D) tensor + mask, ordered like ids; slices sorted by position."""
    idx_of = {}
    for i, g in enumerate(groups):
        idx_of.setdefault(g, []).append(i)
    nmax = max(len(idx_of[g]) for g in ids)
    X = np.zeros((len(ids), nmax, Z.shape[1]), np.float32)
    M = np.zeros((len(ids), nmax), bool)
    order = []
    for b, g in enumerate(ids):
        ii = sorted(idx_of[g], key=lambda i: slice_pos[i])
        X[b, :len(ii)] = Z[ii]
        M[b, :len(ii)] = True
        order.append(ii)
    return X, M, order


def bag_loss(logits, yb, w, loss):
    if loss == "coral":
        K = logits.shape[1] + 1
        tgt = (yb.unsqueeze(1) > torch.arange(K - 1, device=yb.device).unsqueeze(0)).float()
        per = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none").sum(1)
        return (per * w[yb]).sum() / w[yb].sum()
    return F.cross_entropy(logits, yb, weight=w)


def train_mil(Xtr, Mtr, ytr, Xva, Mva, yva, cfg, n_classes, pool, loss, seed, dev, epochs=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n_out = n_classes - 1 if loss == "coral" else n_classes
    model = MIL(Xtr.shape[2], cfg["hidden"], n_out, cfg["dropout"], pool).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cfg["wd"])
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=dev)
    xt, mt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Mtr).to(dev), torch.from_numpy(ytr).long().to(dev)
    if Xva is not None:
        xv, mv, yv = torch.from_numpy(Xva).to(dev), torch.from_numpy(Mva).to(dev), torch.from_numpy(yva).long().to(dev)
    best, best_state, best_ep, bad, hist = float("inf"), None, 0, 0, []
    n_ep = epochs or MAX_EPOCHS
    for ep in range(1, n_ep + 1):
        model.train()
        m = mt
        if cfg["slice_drop"] > 0:
            m = mt & (torch.rand(mt.shape, device=dev) > cfg["slice_drop"])
            m[:, 0] |= ~m.any(1)  # keep at least one slice
        lg, _ = model(xt, m)
        l = bag_loss(lg, yt, w, loss)
        opt.zero_grad(); l.backward(); opt.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                lv, _ = model(xv, mv)
                vl = float(bag_loss(lv, yv, w, loss))
            hist.append({"epoch": ep, "train_loss": float(l), "val_loss": vl})
            if vl < best - 1e-4:
                best, best_ep, bad = vl, ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep or n_ep, hist


@torch.no_grad()
def infer(model, X, M, loss, dev, T=1.0):
    model.eval()
    lg, att = model(torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev))
    return to_probs(lg, loss, T).cpu().numpy(), lg.cpu(), (None if att is None else att.cpu().numpy())


def fit_temperature(logits, y, loss):
    best_T, best_nll = 1.0, float("inf")
    yt = torch.from_numpy(y).long()
    for T in np.exp(np.linspace(np.log(0.25), np.log(8), 60)):
        p = to_probs(logits, loss, float(T))
        nll = float(F.nll_loss(torch.log(p), yt))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def pca_fit(X, seed):
    sc = StandardScaler().fit(X)
    p = PCA(n_components=min(256, X.shape[1]), svd_solver="randomized", random_state=seed).fit(sc.transform(X))
    return lambda A: p.transform(sc.transform(A)).astype(np.float32)


def run(emb, pool, task, loss="ce", calibrate=False, fold_name="folds", tag=""):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = cv.slices()
    X = np.load(cv.CV_DIR / "emb" / f"{emb}.npy")
    y_s = cv.task_labels(df["stage4"].to_numpy(), task)
    nc = len(cv.TASKS[task]["names"])
    groups, pos = df["group"].to_numpy(), df["slice_pos"].to_numpy()
    gy = {g: y_s[i] for i, g in enumerate(groups)}
    name = f"B_mil_{pool}_{emb}" + (f"_{loss}" if loss != "ce" else "") + tag
    oof, att_store, details = {}, {}, []
    for r, k, tr, te in cv.outer_splits(df, cv.folds(fold_name)):
        t0 = time.time()
        seed = 1000 * r + k
        inner = list(cv.inner_folds(df, tr, n=3, seed=seed))
        # inner CV: every config on every inner fold
        scores = np.zeros(len(GRID)); best_eps = [[] for _ in GRID]
        inner_logits = [dict() for _ in GRID]
        for itr, iva in inner:
            f = pca_fit(X[itr], seed)
            ids_tr, ids_va = np.unique(groups[itr]), np.unique(groups[iva])
            Xtr, Mtr, _ = make_bags(f(X), groups, pos, ids_tr)
            Xva, Mva, _ = make_bags(f(X), groups, pos, ids_va)
            ytr = np.array([gy[g] for g in ids_tr]); yva = np.array([gy[g] for g in ids_va])
            for ci, cfg in enumerate(GRID):
                model, bep, _ = train_mil(Xtr, Mtr, ytr, Xva, Mva, yva, cfg, nc, pool, loss, seed, dev)
                P, lg, _ = infer(model, Xva, Mva, loss, dev)
                best_eps[ci].append(bep)
                for g, row in zip(ids_va, lg.numpy()):
                    inner_logits[ci][g] = row
        for ci in range(len(GRID)):
            gs = list(inner_logits[ci]); lg = torch.tensor(np.stack([inner_logits[ci][g] for g in gs]))
            scores[ci] = cv.patient_metrics(np.array([gy[g] for g in gs]), to_probs(lg, loss).numpy(), nc)["macro_f1"]
        cbest = int(np.argmax(scores)); cfg = GRID[cbest]
        n_ep = int(np.median(best_eps[cbest]))
        T = 1.0
        if calibrate:
            gs = list(inner_logits[cbest])
            T = fit_temperature(torch.tensor(np.stack([inner_logits[cbest][g] for g in gs])), np.array([gy[g] for g in gs]), loss)
        # final: refit on outer-train with 3 seeds, fixed epoch count
        f = pca_fit(X[tr], seed)
        ids_tr, ids_te = np.unique(groups[tr]), np.unique(groups[te])
        Xtr, Mtr, _ = make_bags(f(X), groups, pos, ids_tr)
        Xte, Mte, order = make_bags(f(X), groups, pos, ids_te)
        ytr = np.array([gy[g] for g in ids_tr]); yte = np.array([gy[g] for g in ids_te])
        Ps, Ptrs, atts = [], [], []
        for s in range(3):
            model, _, _ = train_mil(Xtr, Mtr, ytr, None, None, None, cfg, nc, pool, loss, seed * 10 + s, dev, epochs=n_ep)
            P, _, att = infer(model, Xte, Mte, loss, dev, T)
            Ptr, _, _ = infer(model, Xtr, Mtr, loss, dev, T)
            Ps.append(P); Ptrs.append(Ptr)
            if att is not None:
                atts.append(att)
        P = np.mean(Ps, 0); Ptr = np.mean(Ptrs, 0)
        if atts:
            A = np.mean(atts, 0)
            for b, g in enumerate(ids_te):
                att_store.setdefault(r, {})[g] = (pos[order[b]], A[b, :len(order[b])])
        oof.setdefault(r, [[], [], []])
        oof[r][0].append(ids_te); oof[r][1].append(yte); oof[r][2].append(P)
        m = cv.patient_metrics(yte, P, nc)
        trm = cv.patient_metrics(ytr, Ptr, nc)["macro_f1"]
        details.append({"r": r, "k": k, "cfg": cfg, "epochs": n_ep, "T": T, "inner_score": float(scores[cbest]), "train_mF1": trm,
                        "ece": cv.ece(yte, P)})
        cv.log_run(stage="B" if not tag else "sens", method=name, config={**cfg, "epochs": n_ep, "T": round(T, 3), "train_patient_mF1": round(trm, 4)},
                   task=task, repeat=r, outer_fold=k, seed=seed, inner_score=round(float(scores[cbest]), 4),
                   outer_macro_f1=round(m["macro_f1"], 4), outer_balanced_acc=round(m["balanced_acc"], 4),
                   seconds=round(time.time() - t0, 1), exploratory=False, note=f"{len(GRID)} configs x 3 inner folds")
        print(f"[mil] {name} {task} r{r} k{k}: cfg {cfg} ep {n_ep} inner {scores[cbest]:.3f} outer {m['macro_f1']:.3f} train {trm:.3f} ({time.time() - t0:.0f}s)", flush=True)
    oof = {r: (np.concatenate(v[0]), np.concatenate(v[1]), np.vstack(v[2])) for r, v in oof.items()}
    extra = {}
    if att_store:  # attention by slice position, for the paper
        rows = [(r, g, int(p), float(a)) for r, d in att_store.items() for g, (ps, aa) in d.items() for p, a in zip(ps, aa)]
        import pandas as pd
        pd.DataFrame(rows, columns=["repeat", "group", "slice_pos", "attention"]).to_csv(cv.CV_DIR / "results" / f"{name}__{task}__attention.csv", index=False)
    cv.save_oof(name, task, oof, extra)
    summ = cv.summarize(oof, nc)
    summ["details"] = details
    summ["ece_mean"] = float(np.mean([d["ece"] for d in details]))
    summ["train_patient_macro_f1_mean"] = float(np.mean([d["train_mF1"] for d in details]))
    (cv.CV_DIR / "results").mkdir(parents=True, exist_ok=True)
    (cv.CV_DIR / "results" / f"{name}__{task}.json").write_text(json.dumps(summ, indent=2, default=str))
    print(f"[mil] {name} {task}: patient macro-F1 {summ['macro_f1']['mean']:.3f} {summ['macro_f1']['ci95']} bal-acc {summ['balanced_acc']['mean']:.3f} "
          f"QWK {summ['qwk']['mean']:.3f} AUC {summ['roc_auc']['mean']:.3f} ECE {summ['ece_mean']:.3f} train {summ['train_patient_macro_f1_mean']:.3f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--pool", default="attn", choices=["attn", "mean", "max"])
    ap.add_argument("--task", default="T3")
    ap.add_argument("--loss", default="ce", choices=["ce", "coral"])
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    run(a.emb, a.pool, a.task, a.loss, a.calibrate)


if __name__ == "__main__":
    main()
