import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = "report_figures"
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# ============================================================
# FIGURE 1: Segmentation performance
# Actual recorded validation and official test metrics
# ============================================================

datasets = ["Validation", "Official Test"]

f1 = [0.923766, 0.804232]
iou = [0.858333, 0.672565]

x = np.arange(len(datasets))
width = 0.32

fig, ax = plt.subplots(figsize=(8.5, 5.2))

bars1 = ax.bar(
    x - width / 2,
    f1,
    width,
    label="Micro F1-score"
)

bars2 = ax.bar(
    x + width / 2,
    iou,
    width,
    label="Intersection over Union"
)

ax.set_title(
    "TRINETRA Segmentation Performance",
    fontweight="bold",
    pad=16
)

ax.set_ylabel("Metric Score")
ax.set_xlabel("Evaluation Split")
ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

ax.bar_label(
    bars1,
    labels=[f"{v:.3f}" for v in f1],
    padding=4,
    fontsize=10
)

ax.bar_label(
    bars2,
    labels=[f"{v:.3f}" for v in iou],
    padding=4,
    fontsize=10
)

ax.legend(loc="upper right", frameon=False)

fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT, "fig1_segmentation_performance.png")
)

plt.close(fig)


# ============================================================
# FIGURE 2: Temporal event confirmation analysis
# Actual selected ABODA validation windows
# ============================================================

videos = ["Video 1", "Video 10", "Video 11"]

confirmation_time = [61.995, 58.025, 46.000]
confidence = [0.9914, 0.9915, 0.9578]

review_required = [False, False, True]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for i in range(len(videos)):
    ax.scatter(
        confirmation_time[i],
        confidence[i],
        s=130,
        marker="o" if not review_required[i] else "X",
        label=(
            videos[i] + (
                " (review required)"
                if review_required[i]
                else " (no review flag)"
            )
        )
    )

    ax.annotate(
        videos[i],
        (confirmation_time[i], confidence[i]),
        xytext=(8, 9),
        textcoords="offset points",
        fontsize=10
    )

ax.set_title(
    "Temporal Event Confirmation Analysis",
    fontweight="bold",
    pad=16
)

ax.set_xlabel("Confirmation Time (seconds)")
ax.set_ylabel("Heuristic Confidence Score")

ax.set_xlim(40, 68)
ax.set_ylim(0.90, 1.025)

ax.set_xticks(np.arange(40, 69, 4))
ax.set_yticks(np.arange(0.90, 1.001, 0.02))

ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="lower right", frameon=False)

ax.text(
    0.02,
    0.03,
    "Confidence is a heuristic score, not a calibrated probability.",
    transform=ax.transAxes,
    fontsize=9
)

fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT, "fig2_temporal_event_analysis.png")
)

plt.close(fig)

print("\nGraph generation completed.")
print("Output directory:", os.path.abspath(OUTPUT))
print("\nGenerated files:")
for filename in os.listdir(OUTPUT):
    print(" -", filename)
