# Task 9: Validation and Refinement of SEM Void Detection

## 1. Objective

The objective of Task 9 was to evaluate the candidate void-detection mask generated from an SEM fractograph, investigate threshold sensitivity, analyze connected components, and document segmentation limitations.

## 2. Input Data and Method

**Input image:** `data/raw/Brittle/100Cr6_AR_05_11.png`

The experimental detector estimates local intensity statistics using Gaussian blurring. It identifies candidate dark pixels using the threshold:

\[
T(x,y)=\mu_{\mathrm{local}}(x,y)-k\sigma_{\mathrm{local}}(x,y)
\]

where:

- \(T(x,y)\) is the local threshold.
- \(\mu_{\mathrm{local}}\) is the local mean intensity.
- \(\sigma_{\mathrm{local}}\) is the local intensity standard deviation.
- \(k\) is the experimental threshold factor.

Pixels darker than the threshold are classified as candidate pixels. Morphological opening and connected-component analysis are then used to examine the resulting mask.

These operations generate candidate regions; they do not independently confirm that a region is a physical void.

## 3. Threshold Comparison

Five threshold factors were evaluated.

| Threshold factor | Candidate pixels | Image coverage | Components |
|---:|---:|---:|---:|
| 0.5 | 33,658 | 13.46% | 982 |
| 0.8 | 12,059 | 4.82% | 571 |
| 1.0 | 3,980 | 1.59% | 237 |
| 1.2 | 1,183 | 0.47% | 84 |
| 1.5 | 98 | 0.04% | 7 |

The image contains 250,000 pixels (500 × 500).

Coverage is calculated as:

Candidate pixel count / total image pixel count × 100.

The results show that candidate coverage and component count are highly sensitive to the selected threshold factor. Increasing the factor makes the detector more selective under the implemented threshold equation.

The coverage percentages are algorithmic measurements, not validated porosity or true material void fractions.

## 4. Provisional Threshold Selection

Factor 1.0 was provisionally selected for further review because visual inspection suggested it was closer to the visible features than the alternatives.

This selection is based on preliminary visual judgment, not quantitative validation against a ground-truth mask. Factor 1.0 must therefore not be described as an optimal or scientifically validated threshold.

## 5. Connected-Component Analysis

The factor 1.0 candidate mask produced the following component-size distribution:

| Component area | Number of components |
|---|---:|
| Under 10 pixels | 0 |
| 10–49 pixels | 236 |
| 50–199 pixels | 1 |
| 200 pixels or more | 0 |
| **Total** | **237** |

The largest component contained 59 pixels.

The absence of components smaller than 10 pixels is consistent with the minimum-area filtering already applied by the refinement script.

Most detected components are small and disconnected. This may indicate fragmented features, surface texture, or genuine small features. Component size alone cannot distinguish among these possibilities.

No components were automatically removed based on circularity, and watershed splitting was not applied because further evidence is needed to establish whether components are fragmented or merged.

## 6. Visual Validation Findings

Visual inspection raised concerns that some candidate detections may follow surface texture rather than distinct physical void boundaries.

The threshold comparison demonstrates substantial changes in candidate coverage as the threshold factor changes. A more restrictive threshold reduces detections, but fewer detections do not necessarily mean greater accuracy.

The current analysis has not established whether all candidate regions correspond to actual voids.

## 7. Outputs

The analysis generated the following outputs.

### Initial void detection

Directory: `data/processed/void_detection/`

- `candidate_voids.png`
- `candidate_voids_cleaned.png`

### Threshold refinement

Directory: `data/processed/task9_refinement/`

- `candidates_factor_0_8.png`
- `refined_factor_0_8.png`
- `candidates_factor_1_0.png`
- `refined_factor_1_0.png`
- `candidates_factor_1_2.png`
- `refined_factor_1_2.png`
- `threshold_comparison.png`

Additional outputs from earlier tests at factors 0.5 and 1.5 may also be present.

### Validation and shape analysis

Directory: `data/processed/task9_validation/`

- `validation_overlay.png`
- `validation_mask.png`
- `labelled_components.png`
- `component_analysis.csv`
- `shape_analysis_factor_1_0.csv`
- `shape_analysis_overlay.png`

The files present in the local project directory should be checked before final submission.

## 8. Limitations

1. Candidate regions have not been independently confirmed as physical voids.
2. Threshold settings are experimental and have not been scientifically calibrated.
3. The analysis has not been evaluated against an independently labelled ground-truth mask.
4. Connected components may represent fragments of one feature or multiple adjacent features.
5. Circularity alone cannot establish whether a component is a genuine void.
6. Physical dimensions cannot be established without suitable image calibration.
7. Candidate pixel coverage must not be reported as true porosity or void fraction.
8. Precision, recall, intersection-over-union, and other ground-truth-based accuracy metrics have not been calculated.

## 9. Conclusion and Status

Task 9 established a reproducible experimental workflow for testing threshold sensitivity, comparing candidate masks, and examining connected-component sizes and shapes.

Factor 1.0 was retained as a provisional candidate for further review based on preliminary visual inspection. Additional tests at factors 0.8 and 1.2 confirmed that candidate coverage and component counts vary substantially with threshold selection.

**Status: Experimental analysis documented; scientific validation pending.**

The current mask is not a verified segmentation of physical voids. No claim of validated porosity, true void fraction, or segmentation accuracy should be made until the candidate regions have been compared against independently labelled reference data or assessed by a qualified domain expert.

## 10. Recommended Future Work

1. Obtain expert-confirmed void annotations or a suitable reference segmentation.
2. Compare candidate masks against the reference data.
3. Evaluate threshold settings using appropriate segmentation metrics.
4. Investigate noise filtering or watershed splitting only where visual evidence supports those changes.
5. Calibrate image pixels to physical dimensions if quantitative measurements are required.
6. Document the final parameter choices and their validation evidence before reporting scientific measurements.
