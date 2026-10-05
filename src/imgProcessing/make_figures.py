"""Paper figure: test macro-F1 under leaky vs honest evaluation (from final_results.json).

Usage: python src/imgProcessing/make_figures.py   -> output/reports/fig_leakage_vs_honest.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mri_config as C  # noqa: E402

LEAKY, HONEST, INK, MUTED = "#eb6834", "#2a78d6", "#1f1f1e", "#6b6a63"


def main() -> None:
    res = json.loads((C.REPORTS_DIR / "final_results.json").read_text(encoding="utf-8"))
    R, S = res["results"], res["deep_seed_summary"]
    rows = [  # label, value, std, leaky?
        ("ResNet-18, naive split (copies + same patients)", R["deep/naive_resnet18_s42"]["macro_f1"], 0, True),
        ("PCA+LogReg, naive split", R["leakage/naive"]["macro_f1"], 0, True),
        ("PCA+LogReg, image-level split (same patients)", R["leakage/image_level_originals"]["macro_f1"], 0, True),
        ("EfficientNet-B0, patient split (3 seeds)", S["efficientnet_b0"]["macro_f1"]["mean"], S["efficientnet_b0"]["macro_f1"]["std"], False),
        ("ResNet-18, patient split (3 seeds)", S["resnet18"]["macro_f1"]["mean"], S["resnet18"]["macro_f1"]["std"], False),
        ("ResNet-50, patient split (3 seeds)", S["resnet50"]["macro_f1"]["mean"], S["resnet50"]["macro_f1"]["std"], False),
        ("SVM (best classical), patient split", R["classical/SVM_RBF"]["macro_f1"], 0, False),
        ("PCA+LogReg, patient split", R["leakage/subject_group"]["macro_f1"], 0, False),
        ("Majority class, patient split", R["baseline/majority_split_subject"]["macro_f1"], 0, False),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v, sd, leaky) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=LEAKY if leaky else HONEST, xerr=sd if sd else None,
                error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        ax.text(v + (sd or 0) + 0.012, yi, f"{v:.2f}" + (f" ± {sd:.2f}" if sd else ""), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Test macro-F1 (4 classes)", color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color="#e4e3dc", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LEAKY, label="leaky evaluation"), Patch(color=HONEST, label="patient-level (honest)")],
              loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Same data, different splits: leakage inflates MRI results", fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    out = C.REPORTS_DIR / "fig_leakage_vs_honest.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
