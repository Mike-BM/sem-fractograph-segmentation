
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"
}


def main():
    images = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not images:
        print(f"No images found in: {RAW_DIR}")
        return

    print(f"Total images found: {len(images)}\n")

    readable = 0

    for path in sorted(images):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"UNREADABLE: {path.relative_to(RAW_DIR)}")
            continue

        height, width = image.shape

        print(f"Image: {path.relative_to(RAW_DIR)}")
        print(f"  Dimensions: {width} x {height} pixels")
        print(f"  Data type: {image.dtype}")
        print(f"  Minimum intensity: {image.min()}")
        print(f"  Maximum intensity: {image.max()}")
        print(f"  Mean intensity: {np.mean(image):.2f}")
        print(f"  Standard deviation: {np.std(image):.2f}")
        print()

        readable += 1

    print("Inspection completed.")
    print(f"Readable images: {readable}/{len(images)}")


if __name__ == "__main__":
    main()
