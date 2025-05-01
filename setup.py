from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clarivuexai",
    version="0.1.0",
    author="wizcodes",
    author_email="wizcodes12@gmail.com",
    description="A unified framework for explainable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clarivuexai",
    project_urls={
        "Bug Tracker": "https://github.com/wizcodes12/clarivuexai/issues",
        "Documentation": "https://clarivuexai.readthedocs.io/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(include=["clarivuexai", "clarivuexai.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "shap>=0.39.0",
        "lime>=0.2.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow>=2.4.0"],
        "torch": ["torch>=1.8.0", "torchvision>=0.9.0"],
        # In setup.py
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",  # Add this
            "black>=21.5b2",
            "isort>=5.9.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "tensorflow>=2.4.0",
            "torch>=1.8.0",
            "torchvision>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clarivuexai=clarivuexai.cli:main",
        ],
    },
)
