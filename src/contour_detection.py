
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "contours"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_contours(image_path):
    """Load a grayscale SEM image and detect candidate contours."""

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Reduce isolated noise while preserving major fracture features.
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Otsu thresholding provides an initial global intensity threshold.
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Detect candidate edges from the binary image.
    edges = cv2.Canny(binary, 50, 150)

    # Find contours in the edge image.
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(
        contour_image,
        contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=1,
    )

    return image, binary, edges, contour_image, threshold_value, contours


def main():
    # Select the first image found in the raw dataset.
    image_paths = sorted(
        path
        for category in ("Brittle", "Ductile", "Mixed")
        for path in (RAW_DIR / category).glob("*")
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found inside {RAW_DIR}"
        )

    image_path = image_paths[0]

    image, binary, edges, overlay, threshold, contours = detect_contours(
        image_path
    )

    # Save the intermediate outputs.
    stem = image_path.stem
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_binary.png"), binary)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_edges.png"), edges)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_contours.png"), overlay)

    print(f"Image: {image_path}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Otsu threshold: {threshold:.2f}")
    print(f"Contours detected: {len(contours)}")
    print(f"Results saved to: {OUTPUT_DIR}")

    # Display each processing stage for visual inspection.
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original SEM image")

    axes[1].imshow(binary, cmap="gray")
    axes[1].set_title("Global threshold")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Candidate edges")

    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Detected contours")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
