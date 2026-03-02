#!/usr/bin/env python
"""
Script to build and upload the canary-fastrtc package to PyPI.

This script:
1. Builds the package using the build module
2. Uploads it to PyPI using twine and the API token from .env
"""

import subprocess
import sys
import os
from pathlib import Path
import dotenv


def main():
    """Build and upload the package to PyPI."""
    # Load environment variables from .env file
    dotenv.load_dotenv()
    pypi_token = os.getenv("PYPI_TOKEN")

    if not pypi_token:
        print("Error: PYPI_TOKEN not found in .env file")
        sys.exit(1)

    print("Building canary-fastrtc package...")

    # Check if build module is installed
    try:
        import build
    except ImportError:
        print("The 'build' package is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build"])

    # Make sure the dist directory is clean
    dist_path = Path("dist")
    if dist_path.exists():
        print("Cleaning dist directory...")
        for file in dist_path.glob("*"):
            file.unlink()

    # Build the package
    subprocess.check_call([sys.executable, "-m", "build"])

    print("\nPackage built successfully!")
    print("\nDistribution files created in the 'dist/' directory:")

    # List the distribution files
    if dist_path.exists():
        for file in dist_path.glob("*"):
            print(f"  - {file.name}")

    # Check if twine is installed
    try:
        import twine
    except ImportError:
        print("\nTwine is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "twine"])

    # Check the package
    print("\nChecking the distribution with twine...")
    subprocess.check_call([sys.executable, "-m", "twine", "check", "dist/*"])

    # Ask for confirmation before uploading
    print("\nReady to upload to PyPI.")
    answer = input("Do you want to proceed with the upload? (y/n): ")

    if answer.lower() != "y":
        print("Upload canceled.")
        sys.exit(0)

    # Upload to PyPI
    print("\nUploading to PyPI...")
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = pypi_token

    try:
        subprocess.check_call(
            [sys.executable, "-m", "twine", "upload", "dist/*"],
            env=env,
        )
        print("\nPackage successfully uploaded to PyPI!")
        print("\nYou can now install it with:")
        print("  pip install canary-fastrtc")
    except subprocess.CalledProcessError as e:
        print(f"\nError uploading to PyPI: {e}")
        print("Please check your token and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
