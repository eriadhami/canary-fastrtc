"""
Setup script for canary-fastrtc.

This file is provided for backward compatibility with older pip versions.
Modern Python packaging prefers pyproject.toml.
"""

from setuptools import setup, find_packages

# This setup.py is minimal and delegates to pyproject.toml
# for most configuration
setup(
    name="canary-fastrtc",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
