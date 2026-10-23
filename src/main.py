
from pathlib import Path


def main():
    # Locate the main project directory
    project_dir = Path(__file__).resolve().parent.parent

    # Define the project directories
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    output_dir = project_dir / "outputs"

    # Create the directories if they do not exist
    for directory in (raw_dir, processed_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print("SEM Fractograph Segmentation Project")
    print("------------------------------------")
    print(f"Raw images:       {raw_dir}")
    print(f"Processed images: {processed_dir}")
    print(f"Outputs:          {output_dir}")
    print("\nProject structure is ready.")
    print("Next: implement the SEM segmentation algorithm.")


if __name__ == "__main__":
    main()
