
from pathlib import Path
import csv
import math

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



# 1. PROJECT PATHS


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Brittle"
    / "100Cr6_AR_05_11.png"
)

VOID_MASK_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "void_detection" / "binary_void_mask.png"
)

CLASSIFICATION_MAP_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "region_classification" / "classification_map.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "quantitative_fractography"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 2. CONFIGURATION


# Set this ONLY when a valid SEM calibration is available.
# Example: 0.25 means each pixel represents 0.25 micrometres.
# Leave as None until the calibration is verified.
PIXEL_SIZE_UM = None

# Ignore very small connected components that may be noise.
# This is an experimental detection setting, not a paper value.
MIN_VOID_AREA_PIXELS = 5

# Connectivity: 8 connects diagonal neighboring pixels.
CONNECTIVITY = 8



# 3. LOAD AND VALIDATE IMAGES


def load_gray(path, description):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"{description} not found or unreadable: {path}"
        )
    return image


gray = load_gray(IMAGE_PATH, "Original SEM image")
void_mask = load_gray(VOID_MASK_PATH, "Task 6 void mask")
classification_map = load_gray(
    CLASSIFICATION_MAP_PATH, "Task 7 classification map"
)

if not (
    gray.shape == void_mask.shape == classification_map.shape
):
    raise ValueError(
        "SEM image, void mask, and classification map "
        "must have identical dimensions."
    )

height, width = gray.shape
image_area_pixels = height * width

# The mask is treated as candidate-void detections.
binary_void = (void_mask > 0).astype(np.uint8)

print("Images loaded successfully.")
print("Image dimensions:", width, "x", height)
print("Image area:", image_area_pixels, "pixels")



# 4. DETECT CONNECTED CANDIDATE VOIDS


num_labels, component_labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary_void,
        connectivity=CONNECTIVITY
    )
)

records = []
kept_mask = np.zeros_like(binary_void)

for component_id in range(1, num_labels):
    x = int(stats[component_id, cv2.CC_STAT_LEFT])
    y = int(stats[component_id, cv2.CC_STAT_TOP])
    box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
    box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    area_pixels = int(stats[component_id, cv2.CC_STAT_AREA])

    if area_pixels < MIN_VOID_AREA_PIXELS:
        continue

    component = (
        component_labels[
            y:y + box_height,
            x:x + box_width
        ] == component_id
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    perimeter_pixels = sum(
        cv2.arcLength(contour, True)
        for contour in contours
    )

    # Equivalent circular diameter based on pixel area.
    equivalent_diameter_pixels = math.sqrt(
        4.0 * area_pixels / math.pi
    )

    # Aspect ratio of the axis-aligned bounding rectangle.
    aspect_ratio = (
        max(box_width, box_height)
        / max(1, min(box_width, box_height))
    )

    # Circularity = 4*pi*area / perimeter^2.
    # It approaches 1 for an ideal circle.
    circularity = (
        4.0 * math.pi * area_pixels / perimeter_pixels**2
        if perimeter_pixels > 0
        else 0.0
    )

    centroid_x, centroid_y = centroids[component_id]

    # Convert to physical units only if calibration is known.
    if PIXEL_SIZE_UM is not None:
        area_um2 = area_pixels * PIXEL_SIZE_UM**2
        diameter_um = (
            equivalent_diameter_pixels * PIXEL_SIZE_UM
        )
        perimeter_um = perimeter_pixels * PIXEL_SIZE_UM
    else:
        area_um2 = None
        diameter_um = None
        perimeter_um = None

    records.append({
        "void_id": len(records) + 1,
        "component_label": component_id,
        "area_pixels": area_pixels,
        "equivalent_diameter_pixels": round(
            equivalent_diameter_pixels, 4
        ),
        "perimeter_pixels": round(perimeter_pixels, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "circularity": round(circularity, 4),
        "centroid_x_pixels": round(float(centroid_x), 2),
        "centroid_y_pixels": round(float(centroid_y), 2),
        "bounding_box_x": x,
        "bounding_box_y": y,
        "bounding_box_width_pixels": box_width,
        "bounding_box_height_pixels": box_height,
        "area_um2": (
            round(area_um2, 6)
            if area_um2 is not None else ""
        ),
        "equivalent_diameter_um": (
            round(diameter_um, 6)
            if diameter_um is not None else ""
        ),
        "perimeter_um": (
            round(perimeter_um, 6)
            if perimeter_um is not None else ""
        ),
    })

    kept_mask[
        component_labels == component_id
    ] = 255



# 5. SAVE INDIVIDUAL VOID MEASUREMENTS


measurements_path = OUTPUT_DIR / "void_measurements.csv"

measurement_fields = [
    "void_id",
    "component_label",
    "area_pixels",
    "equivalent_diameter_pixels",
    "perimeter_pixels",
    "aspect_ratio",
    "circularity",
    "centroid_x_pixels",
    "centroid_y_pixels",
    "bounding_box_x",
    "bounding_box_y",
    "bounding_box_width_pixels",
    "bounding_box_height_pixels",
    "area_um2",
    "equivalent_diameter_um",
    "perimeter_um",
]

with measurements_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=measurement_fields
    )
    writer.writeheader()
    writer.writerows(records)



# 6. CALCULATE IMAGE-LEVEL VOID STATISTICS


void_count = len(records)

detected_void_pixels = int(
    np.count_nonzero(kept_mask)
)

void_area_fraction = (
    detected_void_pixels / image_area_pixels
)

void_area_fraction_percent = (
    void_area_fraction * 100
)

# Number of detected components per million image pixels.
number_density_per_million_pixels = (
    void_count / image_area_pixels * 1_000_000
)

areas = np.array(
    [r["area_pixels"] for r in records],
    dtype=float
)

diameters = np.array(
    [r["equivalent_diameter_pixels"] for r in records],
    dtype=float
)

aspect_ratios = np.array(
    [r["aspect_ratio"] for r in records],
    dtype=float
)

circularities = np.array(
    [r["circularity"] for r in records],
    dtype=float
)


def descriptive_stats(values, prefix, unit):
    """Return summary statistics for a nonempty numeric array."""
    if len(values) == 0:
        return {
            f"{prefix}_mean_{unit}": "",
            f"{prefix}_median_{unit}": "",
            f"{prefix}_std_{unit}": "",
            f"{prefix}_min_{unit}": "",
            f"{prefix}_max_{unit}": "",
        }

    return {
        f"{prefix}_mean_{unit}": round(
            float(np.mean(values)), 4
        ),
        f"{prefix}_median_{unit}": round(
            float(np.median(values)), 4
        ),
        f"{prefix}_std_{unit}": round(
            float(np.std(values)), 4
        ),
        f"{prefix}_min_{unit}": round(
            float(np.min(values)), 4
        ),
        f"{prefix}_max_{unit}": round(
            float(np.max(values)), 4
        ),
    }


summary = {
    "image_width_pixels": width,
    "image_height_pixels": height,
    "image_area_pixels": image_area_pixels,
    "raw_candidate_mask_pixels": int(
        np.count_nonzero(binary_void)
    ),
    "raw_candidate_mask_fraction_percent": round(
        np.mean(binary_void > 0) * 100, 4
    ),
    "minimum_component_area_pixels": MIN_VOID_AREA_PIXELS,
    "connectivity": CONNECTIVITY,
    "detected_components_after_filtering": void_count,
    "detected_void_pixels_after_filtering": detected_void_pixels,
    "void_area_fraction": round(void_area_fraction, 6),
    "void_area_fraction_percent": round(
        void_area_fraction_percent, 4
    ),
    "void_number_density_per_million_pixels": round(
        number_density_per_million_pixels, 4
    ),
    "pixel_size_um": (
        PIXEL_SIZE_UM if PIXEL_SIZE_UM is not None else ""
    ),
    "void_area_fraction_basis": (
        "filtered connected components from Task 6 candidate mask"
    ),
}

summary.update(
    descriptive_stats(areas, "area", "pixels")
)
summary.update(
    descriptive_stats(diameters, "equivalent_diameter", "pixels")
)
summary.update(
    descriptive_stats(aspect_ratios, "aspect_ratio", "unitless")
)
summary.update(
    descriptive_stats(circularities, "circularity", "unitless")
)



# 7. CALCULATE TASK 7 CLASS AREA FRACTIONS


# Task 7 codes:
# 0 = background/unclassified
# 1 = Surface
# 2 = Void
# 3 = Mixed

class_names = {
    0: "Background_or_unclassified",
    1: "Surface",
    2: "Void",
    3: "Mixed",
}

for code, name in class_names.items():
    pixels = int(
        np.count_nonzero(classification_map == code)
    )
    summary[f"{name}_pixels"] = pixels
    summary[f"{name}_area_fraction"] = round(
        pixels / image_area_pixels, 6
    )
    summary[f"{name}_area_fraction_percent"] = round(
        pixels / image_area_pixels * 100, 4
    )


summary_path = OUTPUT_DIR / "quantitative_summary.csv"

with summary_path.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=list(summary.keys())
    )
    writer.writeheader()
    writer.writerow(summary)



# 8. SAVE VOID DETECTION OVERLAY


overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

overlay[kept_mask > 0] = (0, 0, 255)

cv2.imwrite(
    str(OUTPUT_DIR / "detected_voids_overlay.png"),
    overlay
)



# 9. CREATE QUANTITATIVE PLOTS


# Plot 1: Void area distribution.
plt.figure(figsize=(8, 5))

if len(areas) > 0:
    plt.hist(areas, bins=20, edgecolor="black")
else:
    plt.text(
        0.5, 0.5, "No candidate voids detected",
        ha="center", va="center",
        transform=plt.gca().transAxes
    )

plt.xlabel("Connected-component area (pixels²)")
plt.ylabel("Number of candidate components")
plt.title("Candidate Void Area Distribution")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "void_area_distribution.png",
    dpi=200
)
plt.close()


# Plot 2: Equivalent circular diameter distribution.
plt.figure(figsize=(8, 5))

if len(diameters) > 0:
    plt.hist(diameters, bins=20, edgecolor="black")
else:
    plt.text(
        0.5, 0.5, "No candidate voids detected",
        ha="center", va="center",
        transform=plt.gca().transAxes
    )

plt.xlabel("Equivalent circular diameter (pixels)")
plt.ylabel("Number of candidate components")
plt.title("Candidate Void Equivalent Diameter Distribution")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "void_diameter_distribution.png",
    dpi=200
)
plt.close()


# Plot 3: Surface, Void and Mixed area fractions.
plot_names = ["Surface", "Void", "Mixed"]

plot_values = [
    summary[f"{name}_area_fraction_percent"]
    for name in plot_names
]

plt.figure(figsize=(7, 5))
plt.bar(plot_names, plot_values)
plt.ylabel("Fraction of full image (%)")
plt.title("Task 7 Classification Area Fractions")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "classification_area_fractions.png",
    dpi=200
)
plt.close()


# Plot 4: Candidate void shape metrics.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

if len(aspect_ratios) > 0:
    axes[0].hist(aspect_ratios, bins=20, edgecolor="black")
    axes[1].hist(circularities, bins=20, edgecolor="black")
else:
    axes[0].text(
        0.5, 0.5, "No data", ha="center", va="center",
        transform=axes[0].transAxes
    )
    axes[1].text(
        0.5, 0.5, "No data", ha="center", va="center",
        transform=axes[1].transAxes
    )

axes[0].set_xlabel("Bounding-box aspect ratio")
axes[0].set_ylabel("Number of components")
axes[0].set_title("Aspect Ratio Distribution")

axes[1].set_xlabel("Circularity")
axes[1].set_ylabel("Number of components")
axes[1].set_title("Circularity Distribution")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "void_shape_distributions.png",
    dpi=200
)
plt.close(fig)



# 10. PRINT RESULTS


print("\n=== TASK 8: QUANTITATIVE FRACTOGRAPHY ===")
print("Candidate void components:", void_count)
print("Detected void pixels:", detected_void_pixels)
print(
    "Void area fraction:",
    f"{void_area_fraction_percent:.2f}%"
)
print(
    "Candidate components per million image pixels:",
    f"{number_density_per_million_pixels:.2f}"
)

if PIXEL_SIZE_UM is None:
    print(
        "Physical units: NOT calculated; "
        "valid SEM calibration was not supplied."
    )
else:
    print(
        "Pixel calibration:",
        PIXEL_SIZE_UM, "micrometres per pixel"
    )

print("\nOutput directory:", OUTPUT_DIR)
print(" - void_measurements.csv")
print(" - quantitative_summary.csv")
print(" - detected_voids_overlay.png")
print(" - void_area_distribution.png")
print(" - void_diameter_distribution.png")
print(" - classification_area_fractions.png")
print(" - void_shape_distributions.png")

print(
    "\nNOTE: These are candidate-object measurements. "
    "Connected components may merge adjacent voids or split "
    "one physical void. Validate the mask before interpreting "
    "the measurements as physical void statistics."
)
