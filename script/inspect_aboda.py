from pathlib import Path

import cv2


ROOT = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "ABODA"
)


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        print(f"{path.name:<12} OPEN FAIL")
        return False

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    success, frame = capture.read()

    capture.release()

    readable = success and frame is not None

    print(
        f"{path.name:<12}"
        f" frames={frame_count:<7,}"
        f" fps={fps:<7.2f}"
        f" size={width}x{height}"
        f" duration={duration:<8.2f}s"
        f" first_frame={'OK' if readable else 'FAIL'}"
    )

    return readable


def main():
    videos = sorted(ROOT.glob("*.avi"))

    print("ABODA VIDEO AUDIT")
    print("=" * 100)

    print(f"Videos: {len(videos)}")

    results = []

    for video in videos:
        results.append(inspect_video(video))

    print("\n" + "=" * 100)
    print(f"Readable: {sum(results)}/{len(results)}")

    if all(results):
        print("Overall: PASS")
    else:
        print("Overall: FAIL")


if __name__ == "__main__":
    main()