import cv2
import csv
import json
import argparse
import numpy as np

from pathlib import Path


def extract_components(reference, frame, threshold=25, min_area=100):
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(reference_gray, frame_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    components = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Reject components touching the image boundary.
        # These are frequently partial objects entering or leaving the frame.
        image_h, image_w = frame.shape[:2]
        margin = 2

        if (
            x <= margin
            or y <= margin
            or x + w >= image_w - margin
            or y + h >= image_h - margin
        ):
            continue

        components.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "center": [int(x + w / 2), int(y + h / 2)],
            "area": float(area)
        })

    return mask, components


def associate_tracks(tracks, detections, frame_index, max_distance=45):
    assigned = set()

    for track in tracks:
        best_index = None
        best_distance = float("inf")

        tx, ty = track["center"]

        for i, detection in enumerate(detections):
            if i in assigned:
                continue

            dx, dy = detection["center"]
            distance = float(np.hypot(tx - dx, ty - dy))

            if distance < best_distance:
                best_distance = distance
                best_index = i

        if best_index is not None and best_distance <= max_distance:
            detection = detections[best_index]
            assigned.add(best_index)

            track["center"] = detection["center"]
            track["bbox"] = detection["bbox"]
            track["area"] = detection["area"]
            track["last_seen"] = frame_index
            track["observations"] += 1
            track["history"].append(detection["center"])

    for i, detection in enumerate(detections):
        if i in assigned:
            continue

        tracks.append({
            "track_id": len(tracks) + 1,
            "center": detection["center"],
            "bbox": detection["bbox"],
            "area": detection["area"],
            "first_seen": frame_index,
            "last_seen": frame_index,
            "observations": 1,
            "history": [detection["center"]]
        })


def run(video_path, reference_seconds, start_seconds, end_seconds,
        sample_interval, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Invalid video FPS")

    reference_frame_index = int(reference_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_index)
    ok, reference = cap.read()

    if not ok:
        raise RuntimeError("Cannot read reference frame")

    tracks = []
    timeline = []

    sample_seconds = np.arange(
        start_seconds,
        end_seconds + 0.001,
        sample_interval
    )

    for seconds in sample_seconds:
        frame_index = min(int(seconds * fps), total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()

        if not ok:
            continue

        mask, detections = extract_components(reference, frame)

        associate_tracks(
            tracks, detections,
            frame_index=int(seconds),
            max_distance=45
        )

        timeline.append({
            "time_seconds": float(seconds),
            "detections": len(detections),
            "components": detections
        })

    cap.release()

    results = []

    for track in tracks:
        duration = track["last_seen"] - track["first_seen"]

        points = np.array(track["history"], dtype=np.float32)
        movement = float(
            np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))
        )

        results.append({
            "track_id": track["track_id"],
            "first_seen_seconds": track["first_seen"],
            "last_seen_seconds": track["last_seen"],
            "observations": track["observations"],
            "duration_seconds": duration,
            "mean_center_deviation_pixels": round(movement, 2),
            "bbox": track["bbox"],
            "prototype_status": (
                "PERSISTENT_CANDIDATE"
                if duration >= 10 and movement <= 15
                else "UNCERTAIN"
            )
        })

    with open(output_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "timeline.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_seconds",
            "detections"
        ])

        for item in timeline:
            writer.writerow([
                item["time_seconds"],
                item["detections"]
            ])

    with open(output_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(results)

    print("\nTEMPORAL TRACKING RESULTS")
    print("=" * 65)

    for result in results:
        print(
            f"Track {result['track_id']:>3} | "
            f"{result['first_seen_seconds']:>3}s - "
            f"{result['last_seen_seconds']:>3}s | "
            f"Duration: {result['duration_seconds']:>3}s | "
            f"Movement: {result['mean_center_deviation_pixels']:>6.2f}px | "
            f"{result['prototype_status']}"
        )

    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", default="Dataset/ABODA/video10.avi")
    parser.add_argument("--reference", type=float, default=30)
    parser.add_argument("--start", type=float, default=35)
    parser.add_argument("--end", type=float, default=75)
    parser.add_argument("--interval", type=float, default=1)

    parser.add_argument(
        "--output",
        default="outputs/temporal_audit/video10_tracking"
    )

    args = parser.parse_args()

    run(
        Path(args.video),
        args.reference,
        args.start,
        args.end,
        args.interval,
        Path(args.output)
    )
