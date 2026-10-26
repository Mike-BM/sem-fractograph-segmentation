# Task 8: Quantitative Fractography

## Objective
To quantify candidate void-like regions detected in the SEM fractograph using Python image-processing techniques.

## Implementation
The script `src/quantitative_fractography_task8.py` analyses the candidate void mask and calculates connected-component measurements and summary statistics.

The analysis includes:
- Candidate component count
- Number of pixels marked as candidate voids
- Candidate void area fraction
- Candidate components per million image pixels
- Component area and equivalent diameter distributions
- Shape-related measurements and visual overlays
- Region-classification area fractions

## Sample Results

| Metric | Result |
|---|---:|
| Image dimensions | 500 × 500 pixels |
| Candidate components | 502 |
| Candidate void pixels | 59,367 |
| Candidate void area fraction | 23.75% |
| Components per million image pixels | 2,008 |
| Physical calibration | Not supplied |

These values describe the output of the current experimental candidate mask for the sample image. They are not validated physical measurements of porosity.

## Output Files

Results are saved in `data/processed/quantitative_fractography/`:

- `void_measurements.csv`
- `quantitative_summary.csv`
- `detected_voids_overlay.png`
- `void_area_distribution.png`
- `void_diameter_distribution.png`
- `classification_area_fractions.png`
- `void_shape_distributions.png`

## Limitations
1. The candidate mask has not been validated against expert-labelled ground truth.
2. Connected-component analysis may merge adjacent voids or split one physical void into multiple components.
3. Pixel measurements cannot be converted into physical units without valid SEM image calibration.
4. The candidate void area fraction must not be interpreted as validated porosity.
5. Results are based on the current detection and classification settings and may change when those settings are refined.

## Validation Required
Future work should compare the candidate mask with expert annotations, calculate appropriate segmentation metrics, and verify the segmentation methodology against the reference paper.

## Status
**Implemented and executed; scientific validation remains pending.**