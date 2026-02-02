#!/usr/bin/env python3
"""
QEDMMA-Lite Setup Script
========================
Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm
Open Source Edition for radar target tracking.

Author:  Dr. Mladen Mešter / Nexellum d.o.o.
License: MIT (see LICENSE file)
Contact: mladen@nexellum.com | +385 99 737 5100
Website: https://www.nexellum.com
"""

import os
import re
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    init_path = Path(__file__).parent / "qedmma_lite" / "__init__.py"
    if init_path.exists():
        content = init_path.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.0"

# Read README for long description
def get_long_description():
    readme_path = Path(__file__).parent.parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "QEDMMA-Lite: Open source radar tracking algorithms"

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=1.0.0",
    ],
    "fpga": [
        "cocotb>=1.8.0",
        "cocotb-test>=0.2.0",
    ],
    "visualization": [
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
    ],
}

# All extras combined
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

setup(
    # Package metadata
    name="qedmma-lite",
    version=get_version(),
    author="Dr. Mladen Mešter",
    author_email="mladen@nexellum.com",
    maintainer="Nexellum d.o.o.",
    maintainer_email="mladen@nexellum.com",
    
    # Description
    description="Open source radar target tracking with IMM/UKF/CKF filters and Zero-DSP FPGA correlation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/mladen1312/qedmma-lite",
    project_urls={
        "Documentation": "https://github.com/mladen1312/qedmma-lite#readme",
        "Source": "https://github.com/mladen1312/qedmma-lite",
        "Bug Tracker": "https://github.com/mladen1312/qedmma-lite/issues",
        "Changelog": "https://github.com/mladen1312/qedmma-lite/blob/main/CHANGELOG.md",
        "Commercial": "https://www.nexellum.com",
    },
    
    # License
    license="MIT",
    license_files=["../LICENSE"],
    
    # Packages
    packages=find_packages(exclude=["tests", "tests.*", "benchmark", "benchmark.*"]),
    package_dir={"": "."},
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points (CLI tools)
    entry_points={
        "console_scripts": [
            "qedmma-benchmark=qedmma_lite.cli:benchmark_cli",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topic
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Type hints
        "Typing :: Typed",
        
        # Environment
        "Environment :: Console",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords for searchability
    keywords=[
        "radar",
        "tracking",
        "kalman-filter",
        "imm",
        "ukf",
        "ckf",
        "sensor-fusion",
        "fpga",
        "signal-processing",
        "aerospace",
        "defense",
        "dsp",
        "target-tracking",
        "maneuvering-target",
        "multi-sensor",
        "adaptive-filter",
    ],
    
    # Additional options
    zip_safe=False,
    platforms=["any"],
)
