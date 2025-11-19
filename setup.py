"""
ACRF-QIU: Adaptive Causal Random Forest with Quantum-Inspired Uncertainty
Quantification for Multi-Class Prediction in High-Dimensional Data

Setup configuration for pip installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="acrf-qiu",
    version="1.0.0",
    author="Luke Rimmo Lego, Samantha Gauthier, Denver Jn. Baptiste",
    author_email="djnbaptiste@stevens.edu",
    description="Adaptive Causal Random Forest with Quantum-Inspired Uncertainty Quantification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/acrf-qiu",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "causal-inference",
        "machine-learning",
        "random-forest",
        "uncertainty-quantification",
        "conformal-prediction",
        "quantum-inspired",
        "healthcare",
        "biomedical",
        "ensemble-learning",
    ],
)
