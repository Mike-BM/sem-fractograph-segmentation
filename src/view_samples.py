
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, category in zip(axes, ["Ductile", "Mixed", "Brittle"]):
    folder = RAW_DIR / category

    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if not images:
        ax.set_title(f"{category}: no images found")
        ax.axis("off")
        continue

    path = random.choice(images)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        ax.set_title(f"{category}: unreadable image")
        ax.axis("off")
        continue

    ax.imshow(image, cmap="gray")
    ax.set_title(f"{category}\n{path.name}")
    ax.axis("off")

plt.tight_layout()
plt.show()
