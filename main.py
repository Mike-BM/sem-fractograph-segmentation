
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASK6_SCRIPT = ROOT / "src" / "void_detection_task6.py"


def main():
    parser = argparse.ArgumentParser(
        description="SEM fractograph candidate-void detection"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input SEM image",
    )
    parser.add_argument(
        "--output",
        default="data/processed/void_detection",
        help="Directory for generated outputs",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()

    if not image_path.is_file():
        parser.error(f"Input image does not exist: {image_path}")

    if not TASK6_SCRIPT.is_file():
        parser.error(f"Task 6 script not found: {TASK6_SCRIPT}")

    # The existing Task 6 script currently uses fixed paths.
    # This launcher passes the requested paths as environment variables
    # for the next integration step.
    import os

    env = os.environ.copy()
    env["SEM_IMAGE_PATH"] = str(image_path)
    env["SEM_OUTPUT_DIR"] = str(output_path)

    print(f"Input image: {image_path}")
    print(f"Output directory: {output_path}")
    print("Launching Task 6 candidate-void detection...")

    result = subprocess.run(
        [sys.executable, str(TASK6_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print("Processing script finished.")


if __name__ == "__main__":
    main()
