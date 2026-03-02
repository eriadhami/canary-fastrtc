"""
Script to build the canary-fastrtc package.

This script builds the package using the build module, which is the
recommended way to build Python packages.
"""

import subprocess
import sys
import os


def main():
    """Build the package."""
    print("Building canary-fastrtc package...")

    # Check if build module is installed
    try:
        import build
    except ImportError:
        print("The 'build' package is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "build"])

    # Build the package
    subprocess.check_call([sys.executable, "-m", "build"])

    print("\nPackage built successfully!")
    print("\nDistribution files created in the 'dist/' directory:")

    # List the distribution files
    if os.path.exists("dist"):
        for file in os.listdir("dist"):
            print(f"  - {file}")

    print("\nTo install the package locally:")
    print("  pip install dist/*.whl")

    print("\nTo upload to PyPI (if you have the credentials):")
    print("  pip install twine")
    print("  twine upload dist/*")


if __name__ == "__main__":
    main()
