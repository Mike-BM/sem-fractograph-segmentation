
from pathlib import Path
import csv

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "image_summary.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipped unreadable image: {path.name}")
            continue

        height, width = image.shape
        category = path.relative_to(RAW_DIR).parts[0]

        rows.append({
            "filename": str(path.relative_to(RAW_DIR)),
            "category": category,
            "width": width,
            "height": height,
            "dtype": str(image.dtype),
            "min_intensity": int(image.min()),
            "max_intensity": int(image.max()),
            "mean_intensity": round(float(np.mean(image)), 2),
            "std_intensity": round(float(np.std(image)), 2),
        })

    columns = [
        "filename", "category", "width", "height", "dtype",
        "min_intensity", "max_intensity",
        "mean_intensity", "std_intensity",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images summarized: {len(rows)}")
    print(f"Report saved to: {OUTPUT_FILE}")

    for category in sorted({row["category"] for row in rows}):
        count = sum(row["category"] == category for row in rows)
        print(f"{category}: {count} images")


if __name__ == "__main__":
    main()
