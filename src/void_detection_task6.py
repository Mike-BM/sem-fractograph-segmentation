
from pathlib import Path
import csv
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = Path(
    os.environ.get(
        "SEM_IMAGE_PATH",
        str(ROOT / "data" / "raw" / "Brittle" / "100Cr6_AR_05_11.png"),
    )
)

OUT = Path(
    os.environ.get(
        "SEM_OUTPUT_DIR",
        str(ROOT / "data" / "processed" / "void_detection"),
    )
)
LOW_INTENSITY_MAX = 127
HISTOGRAM_WINDOW = 5
LOCAL_WINDOW = 31
MIN_COMPONENT_AREA = 5
EXTENSION_MARGIN = 15


def smooth_histogram(hist, window=5):
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(hist.astype(float), kernel, mode="same")


def first_low_intensity_peak(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    smooth = smooth_histogram(hist, HISTOGRAM_WINDOW)

    # First local maximum in the low-intensity range.
    peaks = [
        i for i in range(1, LOW_INTENSITY_MAX)
        if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
    ]

    if not peaks:
        raise RuntimeError(
            "No low-intensity histogram peak found. "
            "Inspect intensity_histogram.png and review the search range."
        )

    return peaks[0], hist, smooth


def get_components(mask):
    return cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )


def detect_tips(gray, threshold):
    return np.where(gray <= threshold, 255, 0).astype(np.uint8)


def extend_tips(gray, tip_mask, tip_threshold):
    height, width = gray.shape
    result = np.zeros_like(tip_mask)
    count, labels, stats, _ = get_components(tip_mask)
    rows = []

    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_id, cv2.CC_STAT_AREA])

        if area < MIN_COMPONENT_AREA:
            rows.append([component_id, area, "", "", "rejected_small_component"])
            continue

        # Local neighborhood around this tip component.
        pad = LOCAL_WINDOW // 2 + EXTENSION_MARGIN
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)

        local_gray = gray[y1:y2, x1:x2]
        local_component = labels[y1:y2, x1:x2] == component_id

        # Estimate local background statistics from the neighborhood.
        local_mean = float(np.mean(local_gray))
        local_std = float(np.std(local_gray))
        extension_threshold = max(
            local_mean - 0.5 * local_std,
            float(tip_threshold)
        )

        # Restrict extension to the component's local neighborhood.
        # Only pixels below the local threshold are added.
        eligible = local_gray <= extension_threshold
        candidate = result[y1:y2, x1:x2]
        candidate[eligible] = 255

        rows.append([
            component_id, area, round(local_mean, 3),
            round(extension_threshold, 3), "processed"
        ])

    # Remove tiny components after extension.
    count2, labels2, stats2, _ = get_components(result)
    cleaned = np.zeros_like(result)
    for cid in range(1, count2):
        if stats2[cid, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            cleaned[labels2 == cid] = 255

    return cleaned, rows


def save_histogram(hist, smooth, threshold):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(256), hist, label="Grayscale histogram")
    plt.plot(np.arange(256), smooth, label="Smoothed histogram")
    plt.axvline(threshold, linestyle="--",
                label=f"Selected tip threshold = {threshold}")
    plt.xlabel("Grayscale intensity")
    plt.ylabel("Pixel count")
    plt.title("SEM Grayscale Intensity Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "intensity_histogram.png", dpi=150)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")

    gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    threshold, hist, smooth = first_low_intensity_peak(gray)

    # Stage A: candidate void-tip pixels.
    tip_mask = detect_tips(gray, threshold)

    # Stage B: local-threshold extension.
    final_mask, component_rows = extend_tips(gray, tip_mask, threshold)

    # Visual overlay: detected pixels in red.
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    selected = final_mask > 0
    overlay[selected] = (
        0.4 * overlay[selected].astype(float)
        + 0.6 * np.array([0, 0, 255], dtype=float)
    ).astype(np.uint8)

    cv2.imwrite(str(OUT / "void_tip_mask.png"), tip_mask)
    cv2.imwrite(str(OUT / "binary_void_mask.png"), final_mask)
    cv2.imwrite(str(OUT / "void_overlay.png"), overlay)
    save_histogram(hist, smooth, threshold)

    with open(OUT / "component_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "component_id", "tip_area_pixels", "local_mean",
            "extension_threshold", "status"
        ])
        writer.writerows(component_rows)

    total = gray.size
    tip_pixels = int(np.count_nonzero(tip_mask))
    final_pixels = int(np.count_nonzero(final_mask))
    added_pixels = int(np.count_nonzero((final_mask > 0) & (tip_mask == 0)))
    removed_tip_pixels = int(np.count_nonzero((tip_mask > 0) & (final_mask == 0)))

    with open(OUT / "void_detection_summary.txt", "w", encoding="utf-8") as f:
        f.write("TASK 6: VOID DETECTION AND EXTENSION\n")
        f.write(f"Image: {IMAGE_PATH.name}\n")
        f.write(f"Dimensions: {gray.shape[1]} x {gray.shape[0]}\n")
        f.write(f"Tip intensity threshold: {threshold}\n")
        f.write(f"Tip pixels: {tip_pixels}\n")
        f.write(f"Tip coverage (%): {tip_pixels / total * 100:.2f}\n")
        f.write(f"Final void-candidate pixels: {final_pixels}\n")
        f.write(f"Final coverage (%): {final_pixels / total * 100:.2f}\n")
        f.write(f"Pixels added outside tip mask: {added_pixels}\n")
        f.write(f"Tip pixels absent from final mask: {removed_tip_pixels}\n")
        f.write("Deep/shallow classification: not validated or claimed\n")
        f.write("Ground-truth validation: not performed\n")

    print("Task 6 completed.")
    print(f"Image dimensions: {gray.shape[1]} x {gray.shape[0]}")
    print(f"Void-tip threshold: {threshold}")
    print(f"Tip candidate pixels: {tip_pixels}")
    print(f"Final void-candidate pixels: {final_pixels}")
    print(f"Pixels added outside tip mask: {added_pixels}")
    print(f"Tip pixels absent from final mask: {removed_tip_pixels}")
    print(f"Final void coverage: {final_pixels / total * 100:.2f}%")
    print(f"Outputs saved to: {OUT}")


if __name__ == "__main__":
    main()
