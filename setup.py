from setuptools import setup, find_packages

setup(
    name="qedmma-lite",
    version="1.0.0",
    author="Dr. Mladen Mešter",
    author_email="info@mester-labs.com",
    description="Quantum-Enhanced Dynamically-Managed Multi-Model Algorithm for radar tracking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mladen1312/qedmma-lite",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
        "viz": ["matplotlib>=3.4.0"],
    },
)
