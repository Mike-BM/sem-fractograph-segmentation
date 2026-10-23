
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"

# Change this to the actual path of your Excel file.
Labels = Path.home() / "Downloads" / "SEM Image Database" / "SEM Image Database" / "labels.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(Labels)
df.columns = df.columns.astype(str).str.strip().str.lower()

required = {"filename", "label"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Expected columns {required}; found {list(df.columns)}"
    )
df = df[["filename", "label"]].dropna()

# Exclude Windows system files from image-label validation.
df = df[
    ~df["filename"].str.lower().isin(["desktop.ini", "thumbs.db"])
].copy()
df["filename"] = df["filename"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Index copied images by filename.
image_index = {}
for path in RAW_DIR.rglob("*"):
    if path.is_file():
        image_index.setdefault(path.name.lower(), []).append(path)

def check_row(row):
    matches = image_index.get(row["filename"].lower(), [])

    if not matches:
        return pd.Series(["MISSING", ""])

    if len(matches) > 1:
        return pd.Series(["DUPLICATE_FILENAME", ""])

    image_path = matches[0]
    folder_label = image_path.relative_to(RAW_DIR).parts[0]

    status = (
        "MATCH"
        if folder_label.casefold() == row["label"].casefold()
        else "LABEL_FOLDER_MISMATCH"
    )
    return pd.Series([status, str(image_path.relative_to(RAW_DIR))])

df[["validation_status", "image_path"]] = df.apply(check_row, axis=1)

report = OUTPUT_DIR / "label_validation_report.csv"
df.to_csv(report, index=False)

print(f"Spreadsheet rows checked: {len(df)}")
print("\nValidation results:")
print(df["validation_status"].value_counts().to_string())
print(f"\nReport saved to: {report}")
