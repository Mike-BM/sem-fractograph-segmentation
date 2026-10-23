# SEM Fractograph Dataset Notes

## 1. Dataset Overview

- **Purpose:** Support research and development of a Python-based SEM image-processing pipeline.
- **Dataset title:** SEM Image dataset
- **Data collector:** Alessandro Campari
- **Affiliation:** Norwegian University of Science and Technology
- **Total readable images:** 1,961
- **Image categories:** Ductile, Brittle, Mixed
- **Image format:** PNG
- **Dimensions observed:** 500 × 500 pixels in the inspected examples
- **Image inspection:** All 1,961 images were readable using OpenCV.
- **Label validation:** All 1,961 image entries matched their category folders after excluding the Windows system-file entry `desktop.ini`.

## 2. Category Counts

| Category | Number of images | Percentage |
|---|---:|---:|
| Ductile | 996 | 50.79% |
| Brittle | 822 | 41.92% |
| Mixed | 143 | 7.29% |
| **Total** | **1,961** | **100%** |

## 3. Source and Licence

- **Source:** https://zenodo.org/records/15510590
- **DOI:** https://doi.org/10.5281/zenodo.15510590
- **Licence stated in the downloaded dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Licence terms:** https://creativecommons.org/licenses/by/4.0/

When reusing or redistributing material under CC BY 4.0, provide appropriate attribution, link to the licence, and indicate whether changes were made. Do not imply that the original creator endorses this project.

Confirm the licence displayed on the original Zenodo record before public redistribution.

## 4. Dataset Description

The dataset contains scanning electron microscopy images of steel fracture surfaces associated with different testing conditions, including hydrogen exposure and loading conditions.

The accompanying spreadsheet maps image filenames to image-level labels.

## 5. Project Use and Limitations

This project aims to implement the traditional image-processing methodology described in the assigned research paper, using Python and OpenCV.

The labels `Ductile`, `Brittle`, and `Mixed` describe image-level fracture categories. They are not verified pixel-level segmentation masks for surface and void regions. These labels will not be treated as ground-truth segmentation annotations unless their meaning is independently verified.

## 6. Generated Reports

- `data/processed/image_summary.csv`
- `data/processed/label_validation_report.csv`

## 7. Processing Notes

The original downloaded dataset files were retained separately. Image copies are stored in `data/raw/`.

The inspection script successfully read 1,961 images. The summary report records image dimensions and grayscale intensity statistics, while the label-validation report records filename and category consistency.
