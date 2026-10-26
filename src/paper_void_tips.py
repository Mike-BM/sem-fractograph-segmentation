
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "data" / "raw" / "Brittle" / "100Cr6_AR_05_11.png"
OUTPUT_DIR = ROOT / "data" / "processed" / "paper_void_tips"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    # --------------------------------------------------
    # 1. Calculate and smooth the intensity histogram
    # --------------------------------------------------
    histogram = np.bincount(image.ravel(), minlength=256).astype(float)
    smoothed = np.convolve(histogram, np.ones(5) / 5, mode="same")

    # --------------------------------------------------
    # 2. Find candidate histogram peaks
    # --------------------------------------------------
    peaks, _ = find_peaks(
        smoothed,
        prominence=10,
        distance=5,
    )

    # Exploratory filter, not a parameter specified by the paper.
    low_peaks = peaks[(peaks >= 20) & (peaks <= 127)]

    print("\n--- HISTOGRAM PEAK ANALYSIS ---")
    print("Candidate low-intensity peaks:", low_peaks.tolist())

    if len(low_peaks) == 0:
        raise RuntimeError(
            "No suitable low-intensity peak found. "
            "Inspect the histogram before continuing."
        )

    tip_intensity = int(low_peaks[0])

    # --------------------------------------------------
    # 3. Calculate image intensity statistics
    # --------------------------------------------------
    total_pixels = image.size
    mean_intensity = float(np.mean(image))
    std_intensity = float(np.std(image))
    min_intensity = int(np.min(image))
    max_intensity = int(np.max(image))

    print("\n--- IMAGE INTENSITY DIAGNOSTICS ---")
    print(f"Image path: {IMAGE_PATH}")
    print(f"Dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Minimum intensity: {min_intensity}")
    print(f"Maximum intensity: {max_intensity}")
    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Standard deviation: {std_intensity:.2f}")
    print(
        f"Pixels at intensity 0: "
        f"{100 * np.count_nonzero(image == 0) / total_pixels:.2f}%"
    )
    print(
        f"Pixels at intensity 255: "
        f"{100 * np.count_nonzero(image == 255) / total_pixels:.2f}%"
    )

    # Check how much of the image lies below each threshold.
    print("\n--- THRESHOLD COVERAGE ---")
    for threshold in [4, 25, 50, 72, 92, 127, 200, 250]:
        count = int(np.count_nonzero(image < threshold))
        percentage = 100 * count / total_pixels
        print(
            f"Pixels with I < {threshold:3}: "
            f"{count:6} ({percentage:.2f}%)"
        )

    # Inspect raw histogram counts near the selected tip threshold.
    print("\n--- HISTOGRAM COUNTS NEAR TIP THRESHOLD ---")
    for intensity in range(
        max(0, tip_intensity - 5),
        min(255, tip_intensity + 5) + 1,
    ):
        print(f"I={intensity:3}: {int(histogram[intensity])} pixels")

    # --------------------------------------------------
    # 4. Generate the candidate tip and extension masks
    # --------------------------------------------------
    extension_threshold = max(
        mean_intensity - 0.5 * std_intensity,
        float(tip_intensity),
    )

    # White pixels represent original intensities BELOW the threshold.
    tip_mask = (image < tip_intensity).astype(np.uint8) * 255
    extension_mask = (
        image.astype(float) < extension_threshold
    ).astype(np.uint8) * 255

    tip_count = int(np.count_nonzero(tip_mask))
    extension_count = int(np.count_nonzero(extension_mask))

    print("\n--- MASK RESULTS ---")
    print(f"Estimated tip intensity (It): {tip_intensity}")
    print(f"Extension threshold (Iv): {extension_threshold:.2f}")
    print(
        f"Tip pixels: {tip_count} "
        f"({100 * tip_count / total_pixels:.2f}%)"
    )
    print(
        f"Extension-mask pixels: {extension_count} "
        f"({100 * extension_count / total_pixels:.2f}%)"
    )
    print("Mask polarity: white = intensity below threshold")

    # Save masks.
    cv2.imwrite(str(OUTPUT_DIR / "void_tip_mask.png"), tip_mask)
    cv2.imwrite(
        str(OUTPUT_DIR / "void_extension_mask.png"),
        extension_mask,
    )

    # --------------------------------------------------
    # 5. Visualize the original image and masks
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original SEM image")

    axes[0, 1].imshow(tip_mask, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(f"Candidate tips: I < {tip_intensity}")

    axes[1, 0].imshow(
        extension_mask, cmap="gray", vmin=0, vmax=255
    )
    axes[1, 0].set_title(
        f"Extension mask: I < {extension_threshold:.1f}"
    )

    axes[1, 1].plot(np.arange(256), histogram)
    axes[1, 1].plot(np.arange(256), smoothed)
    axes[1, 1].axvline(
        tip_intensity,
        linestyle="--",
        label=f"It = {tip_intensity}",
    )
    axes[1, 1].axvline(
        extension_threshold,
        linestyle=":",
        label=f"Iv = {extension_threshold:.1f}",
    )
    axes[1, 1].set_title("Intensity histogram")
    axes[1, 1].set_xlabel("Pixel intensity (0 = black, 255 = white)")
    axes[1, 1].set_ylabel("Pixel count")
    axes[1, 1].legend()

    for axis in axes.flat[:3]:
        axis.axis("off")

    plt.tight_layout()
    output_plot = OUTPUT_DIR / "void_tip_comparison.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\nSaved diagnostic figure: {output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
