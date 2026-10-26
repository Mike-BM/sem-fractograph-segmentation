
from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.segmentation import slic


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Brittle"
    / "100Cr6_AR_05_11.png"
)

REGION_MASK_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "contour_refinement" / "refined_regions_mask.png"
)

VOID_MASK_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "void_detection" / "binary_void_mask.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "region_classification"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. CONFIGURABLE PARAMETERS
# =========================================================

# SLIC candidate-region settings
N_SEGMENTS = 80
SLIC_COMPACTNESS = 2.0
SLIC_SIGMA = 1.0



# Minimum region area retained for classification
MIN_REGION_PIXELS = 20

# Experimental classification settings.
# These are not confirmed paper thresholds.
VOID_COVERAGE_THRESHOLD = 0.90
VOID_TIP_THRESHOLD = 4
INTENSITY_PERCENTILE = 80
GLOBAL_STD_MULTIPLIER = 1.0


# 3. LOAD AND VALIDATE INPUTS

def load_grayscale_image(path: Path, description: str):
    """Load a grayscale image or raise a helpful error."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"{description} could not be loaded: {path}"
        )

    return image


gray = load_grayscale_image(
    IMAGE_PATH, "Original SEM image"
)

region_mask = load_grayscale_image(
    REGION_MASK_PATH, "Task 5 region mask"
)

void_mask = load_grayscale_image(
    VOID_MASK_PATH, "Task 6 candidate void mask"
)

if not (
    gray.shape == region_mask.shape == void_mask.shape
):
    raise ValueError(
        "The original SEM image, Task 5 mask, and Task 6 "
        "mask must have identical dimensions."
    )

print("Images loaded successfully.")
print("Image dimensions:", gray.shape)
print(
    "Task 5 contour pixels:",
    int(np.count_nonzero(region_mask))
)
print(
    "Task 6 candidate void pixels:",
    int(np.count_nonzero(void_mask))
)

# 4. GENERATE CANDIDATE REGIONS USING SLIC
# The Task 5 mask contains sparse contours. We therefore
# generate candidate regions from the original SEM image.
# SLIC regions are computational image segments, not
# automatically verified physical fracture regions.

region_labels = slic(
    gray,
    n_segments=N_SEGMENTS,
    compactness=SLIC_COMPACTNESS,
    sigma=SLIC_SIGMA,
    start_label=1,
    channel_axis=None,
    enforce_connectivity=True
).astype(np.int32)

region_ids = np.unique(region_labels)
region_ids = region_ids[region_ids > 0]

np.save(
    OUTPUT_DIR / "region_labels.npy",
    region_labels
)

print("\nSLIC segmentation completed.")
print("Candidate regions generated:", len(region_ids))
print(
    "Pixels assigned to candidate regions:",
    int(np.count_nonzero(region_labels > 0))
)
print(
    "Candidate-region coverage:",
    f"{np.count_nonzero(region_labels > 0) / gray.size * 100:.2f}%"
)


# 5. SAVE REGION PREVIEW AND BOUNDARY OVERLAY

# A deterministic grayscale label preview.
preview = np.zeros_like(gray, dtype=np.uint8)

preview[region_labels > 0] = (
    (region_labels[region_labels > 0] * 37) % 256
).astype(np.uint8)

cv2.imwrite(
    str(OUTPUT_DIR / "region_labels_preview.png"),
    preview
)

# Mark boundaries between neighboring region labels.
boundaries = np.zeros_like(gray, dtype=bool)

boundaries[1:, :] |= (
    region_labels[1:, :] != region_labels[:-1, :]
)

boundaries[:, 1:] |= (
    region_labels[:, 1:] != region_labels[:, :-1]
)

boundary_overlay = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2BGR
)
boundary_overlay[boundaries] = (0, 0, 255)

cv2.imwrite(
    str(OUTPUT_DIR / "region_boundaries_overlay.png"),
    boundary_overlay
)

# 6. COMPUTE GLOBAL IMAGE STATISTICS

global_mean = float(np.mean(gray))
global_std = float(np.std(gray))

global_limit = (
    global_mean
    + GLOBAL_STD_MULTIPLIER * global_std
)

# Any nonzero Task 6 mask pixel is treated as a candidate
# void pixel. This is not confirmed ground-truth labeling.
void_binary = void_mask > 0

# Classification codes:
# 0 = background / not classified
# 1 = Surface
# 2 = Void
# 3 = Mixed

class_codes = {
    "Surface": 1,
    "Void": 2,
    "Mixed": 3,
}

classification_map = np.zeros(
    gray.shape, dtype=np.uint8
)

records = []

# 7. CLASSIFY EACH CANDIDATE REGION

for region_id in region_ids:

    region_pixels = region_labels == region_id
    area = int(np.count_nonzero(region_pixels))

    if area < MIN_REGION_PIXELS:
        continue

    intensities = gray[region_pixels]
    candidate_void_pixels = void_binary[region_pixels]

    void_pixels = int(
        np.count_nonzero(candidate_void_pixels)
    )

    void_coverage = void_pixels / area

    # Most frequent grayscale intensity in this region.
    histogram = np.bincount(
        intensities,
        minlength=256
    )

    histogram_peak = int(np.argmax(histogram))

    # Region intensity percentile.
    intensity_percentile = float(
        np.percentile(
            intensities,
            INTENSITY_PERCENTILE
        )
    )

    # Preliminary experimental rules.
    if void_pixels == 0:
        class_name = "Surface"

    elif void_coverage >= VOID_COVERAGE_THRESHOLD:
        class_name = "Void"

    elif (
        histogram_peak <= VOID_TIP_THRESHOLD
        and intensity_percentile < global_limit
    ):
        class_name = "Void"

    else:
        class_name = "Mixed"

    classification_map[region_pixels] = (
        class_codes[class_name]
    )

    records.append({
        "region_id": int(region_id),
        "area_pixels": area,
        "candidate_void_pixels": void_pixels,
        "void_coverage": round(void_coverage, 4),
        "histogram_peak": histogram_peak,
        "intensity_percentile": round(
            intensity_percentile, 2
        ),
        "global_mean": round(global_mean, 2),
        "global_std": round(global_std, 2),
        "global_limit": round(global_limit, 2),
        "classification": class_name,
    })


# =========================================================
# 8. SAVE CLASSIFICATION CSV
# =========================================================

csv_path = OUTPUT_DIR / "region_classification.csv"

fieldnames = [
    "region_id",
    "area_pixels",
    "candidate_void_pixels",
    "void_coverage",
    "histogram_peak",
    "intensity_percentile",
    "global_mean",
    "global_std",
    "global_limit",
    "classification",
]

try:
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(records)

except PermissionError as exc:
    raise PermissionError(
        f"Cannot write to {csv_path}. Close the CSV in "
        "Excel or another application and try again."
    ) from exc


# =========================================================
# 9. SAVE CLASSIFICATION MAP AND OVERLAY
# =========================================================

cv2.imwrite(
    str(OUTPUT_DIR / "classification_map.png"),
    classification_map
)

# OpenCV uses BGR color order.
colors = {
    0: (0, 0, 0),        # Unclassified
    1: (0, 200, 0),      # Surface: green
    2: (0, 0, 255),      # Void: red
    3: (0, 200, 255),    # Mixed: yellow/orange
}

color_map = np.zeros(
    (gray.shape[0], gray.shape[1], 3),
    dtype=np.uint8
)

for code, color in colors.items():
    color_map[classification_map == code] = color

gray_bgr = cv2.cvtColor(
    gray, cv2.COLOR_GRAY2BGR
)

overlay = gray_bgr.copy()
active = classification_map > 0

blended = cv2.addWeighted(
    gray_bgr, 0.40,
    color_map, 0.60,
    0
)

overlay[active] = blended[active]

cv2.imwrite(
    str(OUTPUT_DIR / "classification_overlay.png"),
    overlay
)


# =========================================================
# 10. REPORT RESULTS AND COVERAGE
# =========================================================

print("\nClassification counts:")

for name, code in class_codes.items():

    region_total = sum(
        row["classification"] == name
        for row in records
    )

    pixels = int(
        np.count_nonzero(classification_map == code)
    )

    percentage = (
        pixels / classification_map.size * 100
    )

    print(
        f"{name}: {region_total} regions, "
        f"{pixels} pixels ({percentage:.2f}%)"
    )

unclassified = int(
    np.count_nonzero(classification_map == 0)
)

print("\nClassified regions:", len(records))
print(
    f"Unclassified/background pixels: {unclassified} "
    f"({unclassified / classification_map.size * 100:.2f}%)"
)

print("\nOutputs saved in:", OUTPUT_DIR)
print(" - region_labels.npy")
print(" - region_labels_preview.png")
print(" - region_boundaries_overlay.png")
print(" - region_classification.csv")
print(" - classification_map.png")
print(" - classification_overlay.png")

print(
    "\nNOTE: Region labels and class assignments are "
    "experimental. Validate segmentation and thresholds "
    "against the reference methodology and image evidence."
)
