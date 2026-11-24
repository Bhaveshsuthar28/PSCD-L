import csv
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "outputs/index/vl_cmu_test.json"

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/analysis/geometric_alignment_audit"
)

SAMPLE_IDS = [
    "085_1_00_0",
    "145_1_02_0",
    "139_1_05_0",
    "045_1_00_0",
]

ORB_FEATURES = 5000
RATIO_TEST = 0.75
RANSAC_THRESHOLD = 3.0


# =========================================================
# IMAGE LOADING
# =========================================================

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))



def resolve_path(record, key):
    # First, try the original path stored in the dataset index.
    original_path = Path(record[key])

    if original_path.is_file():
        return original_path

    # Fallback: resolve the dataset from the current project root.
    sample_id = record["sample_id"]

    fallback_path = (
        PROJECT_ROOT
        / "Dataset"
        / "VL-CMU-CD-binary255"
        / "test"
        / key.replace("_path", "")
        / f"{sample_id}.png"
    )

    if fallback_path.is_file():
        print(f"Using fallback path: {fallback_path}")
        return fallback_path

    raise FileNotFoundError(
        f"Could not find {key} for {sample_id}.\n"
        f"Original: {original_path}\n"
        f"Fallback: {fallback_path}"
    )



# =========================================================
# ALIGNMENT
# =========================================================

def estimate_alignment(t0_rgb, t1_rgb):

    t0_gray = cv2.cvtColor(t0_rgb, cv2.COLOR_RGB2GRAY)
    t1_gray = cv2.cvtColor(t1_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

    kp0, des0 = orb.detectAndCompute(t0_gray, None)
    kp1, des1 = orb.detectAndCompute(t1_gray, None)

    result = {
        "keypoints_t0": len(kp0),
        "keypoints_t1": len(kp1),
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "median_reprojection_error": None,
        "homography": None,
        "aligned_t1": None,
        "matches_image": None,
        "status": "FAILED",
    }

    if des0 is None or des1 is None or len(kp0) < 4 or len(kp1) < 4:
        result["status"] = "INSUFFICIENT_FEATURES"
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pairs = matcher.knnMatch(des1, des0, k=2)

    good = [
        first for first, second in pairs
        if first.distance < RATIO_TEST * second.distance
    ]

    result["good_matches"] = len(good)

    if len(good) < 4:
        result["status"] = "INSUFFICIENT_MATCHES"
        return result

    # Query points come from T1; train points come from T0.
    # Therefore H maps T1 coordinates into T0 coordinates.
    src = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst = np.float32(
        [kp0[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
    )

    if H is None or inlier_mask is None:
        result["status"] = "HOMOGRAPHY_FAILED"
        return result

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())

    projected = cv2.perspectiveTransform(src, H)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst.reshape(-1, 2),
        axis=1,
    )

    median_error = float(np.median(errors[inliers]))

    height, width = t0_rgb.shape[:2]

    aligned_t1 = cv2.warpPerspective(
        t1_rgb,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    matches_image = cv2.drawMatches(
        t1_rgb,
        kp1,
        t0_rgb,
        kp0,
        good,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=inliers.astype(np.uint8).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    result.update({
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(good),
        "median_reprojection_error": median_error,
        "homography": H,
        "aligned_t1": aligned_t1,
        "matches_image": matches_image,
        "status": "SUCCESS",
    })

    return result


# =========================================================
# DIAGNOSTIC METRICS
# =========================================================

def mean_absolute_difference(image_a, image_b):

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)

    return float(
        np.mean(
            np.abs(
                gray_a.astype(np.float32) -
                gray_b.astype(np.float32)
            )
        )
    )


def save_visual_report(sample_id, t0, t1, result, output_dir):

    if result["status"] != "SUCCESS":
        return

    aligned = result["aligned_t1"]

    diff_before = cv2.absdiff(t0, t1)
    diff_after = cv2.absdiff(t0, aligned)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    panels = [
        (t0, "T0 - Reference", None),
        (t1, "T1 - Original", None),
        (aligned, "T1 - Homography aligned", None),
        (diff_before, "Absolute difference BEFORE", None),
        (diff_after, "Absolute difference AFTER", None),
        (result["matches_image"], "ORB inlier matches", None),
    ]

    for ax, (image, title, cmap) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"{sample_id} | "
        f"Inliers: {result['inliers']}/{result['good_matches']} | "
        f"Inlier ratio: {result['inlier_ratio']:.3f} | "
        f"Median reprojection error: "
        f"{result['median_reprojection_error']:.2f}px",
        fontsize=13,
        fontweight="bold",
    )

    fig.tight_layout()

    output_path = output_dir / f"{sample_id}_alignment.png"

    fig.savefig(
        output_path,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = json.loads(
        INDEX_PATH.read_text(encoding="utf-8")
    )

    record_map = {
        record["sample_id"]: record
        for record in records
    }

    rows = []

    print("=" * 75)
    print("TRINETRA GEOMETRIC ALIGNMENT AUDIT")
    print("=" * 75)

    for sample_id in SAMPLE_IDS:

        print(f"\nProcessing {sample_id}")

        if sample_id not in record_map:
            raise KeyError(f"Sample not found: {sample_id}")

        record = record_map[sample_id]

        t0 = load_rgb(resolve_path(record, "t0_path"))
        t1 = load_rgb(resolve_path(record, "t1_path"))

        if t0.shape != t1.shape:
            raise ValueError(
                f"Image shape mismatch for {sample_id}: "
                f"{t0.shape} vs {t1.shape}"
            )

        result = estimate_alignment(t0, t1)

        before_diff = mean_absolute_difference(t0, t1)

        after_diff = None

        if result["status"] == "SUCCESS":
            after_diff = mean_absolute_difference(
                t0,
                result["aligned_t1"],
            )

            save_visual_report(
                sample_id,
                t0,
                t1,
                result,
                OUTPUT_DIR,
            )

        H = result["homography"]

        rows.append({
            "sample_id": sample_id,
            "status": result["status"],
            "keypoints_t0": result["keypoints_t0"],
            "keypoints_t1": result["keypoints_t1"],
            "good_matches": result["good_matches"],
            "inliers": result["inliers"],
            "inlier_ratio": result["inlier_ratio"],
            "median_reprojection_error":
                result["median_reprojection_error"],
            "mean_abs_diff_before": before_diff,
            "mean_abs_diff_after": after_diff,
            "homography": (
                json.dumps(H.tolist()) if H is not None else ""
            ),
        })

        print("Status:", result["status"])
        print("Keypoints T0:", result["keypoints_t0"])
        print("Keypoints T1:", result["keypoints_t1"])
        print("Good matches:", result["good_matches"])
        print("Inliers:", result["inliers"])
        print("Inlier ratio:", round(result["inlier_ratio"], 4))

        if after_diff is not None:
            print(
                "Median reprojection error:",
                round(result["median_reprojection_error"], 3),
            )
            print("Mean absolute difference before:", round(before_diff, 3))
            print("Mean absolute difference after:", round(after_diff, 3))

        else:
            print("Alignment could not be estimated reliably.")

    csv_path = OUTPUT_DIR / "alignment_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 75)
    print("ALIGNMENT AUDIT COMPLETED")
    print("=" * 75)
    print("Metrics:", csv_path)
    print("Visual reports:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
