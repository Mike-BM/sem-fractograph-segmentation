# Automated SEM Fractograph Segmentation Using Python

## Project Overview

This project implements a conventional image-processing pipeline for analysing scanning electron microscopy (SEM) fractographs. Its goal is to identify candidate void regions, refine image contours, classify image regions, and produce quantitative measurements using Python.

The project follows the intended direction of the reference paper, *Segmentation of SEM Fractograph into Void and Surface Regions*, by Siddhartha Banerjee, Kaustav Basu, Pravash Chandra Chakraborti, and Sanjoy Kumar Saha. The implementation must be checked against the paper's specific methodology before full methodological compliance can be claimed.

This is an image-processing project, not a trained deep-learning model.

## Features

- SEM image inspection and preprocessing.
- Contour detection and refinement.
- Experimental candidate-void detection.
- Surface, Void, and Mixed region classification.
- Candidate-void measurements and visualizations.
- Binary-mask validation metrics: IoU, Dice, precision, and recall.
- Command-line execution for candidate-void detection.
- Automated software tests using pytest.

## Technology Stack

- Python
- NumPy
- OpenCV
- scikit-image
- SciPy
- Matplotlib
- pandas
- pytest

## Project Structure

```text
sem-fractograph-segmentation/
├── data/
│   ├── raw/                 # Local SEM dataset; excluded from Git
│   └── processed/            # Generated results; local outputs
├── src/
│   ├── void_detection_task6.py
│   ├── region_classification_task7.py
│   ├── quantitative_fractography_task8.py
│   └── validation_metrics.py
├── tests/
│   └── test_project.py
├── TASK4_IMAGE_INSPECTION.md
├── TASK5_CONTOUR_REFINEMENT.md
├── TASK6_VOID_DETECTION.md
├── TASK7_REGION_CLASSIFICATION.md
├── TASK8_QUANTITATIVE_FRACTOGRAPHY.md
├── TASK9_VALIDATION.md
├── TASK10_VALIDATION_REPORT.md
├── main.py
├── environment.py
├── requirements.txt
└── README.md
```

Additional experimental scripts and output directories may exist locally.

## Requirements

- Windows, macOS, or Linux.
- A compatible Python installation.
- Dependencies listed in `requirements.txt`.
- SEM images available locally and readable by OpenCV.

## Installation on Windows

Open PowerShell in the project directory.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the project's Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Command-Line Application

Display available options:

```powershell
python main.py --help
```

Run candidate-void detection on an SEM image:

```powershell
python main.py --image "data\raw\Brittle\100Cr6_AR_05_11.png"
```

Choose a different output directory:

```powershell
python main.py --image "data\raw\Brittle\100Cr6_AR_05_11.png" --output "outputs\sample_run"
```

The current command-line interface runs the Task 6 candidate-void detection script. It does not yet combine every project task into one unified pipeline.

## Run the Tests

Run the automated tests:

```powershell
python -m pytest -v
```

The current test suite includes nine passing tests covering basic project setup, binary masks, overlap calculations, and segmentation metric behavior on synthetic masks.

These tests verify software behavior; they do not establish scientific segmentation accuracy.

## Outputs

The candidate-void detection script generates files under `data/processed/void_detection/`, including candidate masks, overlays, and diagnostic outputs.

Other experimental scripts save their results to their corresponding subdirectories under `data/processed/`.

## Validation Status

The project has been executed on sample SEM data and has a working automated test suite. The current results are experimental.

Validation against manually annotated ground-truth masks has not yet been established. Consequently, IoU, Dice, precision, and recall have not been reported as real-world segmentation performance.

The candidate-void coverage percentage is a pixel-mask statistic and should not be interpreted as verified physical porosity. Physical measurements require valid SEM calibration.

Further work should compare results across multiple images and magnifications, inspect false-positive and false-negative detections, assess parameter sensitivity, and verify the implementation against the reference paper.

## Data and Reproducibility

The raw dataset is excluded from Git using `.gitignore`. Users must obtain the dataset with the appropriate permissions and place the images in the expected local directories.

Do not redistribute dataset images unless their licence permits it. Record image sources, magnification, calibration, and processing parameters when available.

## Research Integrity

This repository documents a developing conventional image-processing implementation. Experimental outputs should not be described as validated scientific results until they have been compared with suitable reference annotations and the specified methodology.

## Licence

Check and document the licence for the source dataset and project code before redistributing either.
