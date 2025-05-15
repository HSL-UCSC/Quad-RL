from setuptools import setup, find_packages
import io
import os

here = os.path.abspath(os.path.dirname(__file__))

# long_description from README
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="QuadRL",
    version="1.0.0",
    description="Hybrid Reinforcement Learning for Obstacle Avoidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HSL-UCSC",
    url="https://github.com/HSL-UCSC/ObstacleAvoidanceHyRL",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests*", "examples*"]),
    include_package_data=True,
    package_data={
        # key = the package name (folder under src/).
        # Here 'models' means src/models
        "models": ["*"],  # all files in models/
        # If you have deep subfolders, e.g. src/models/foo/*.dat, you can do:
        # "models": ["**/*"],
    },
    install_requires=[
        "gym",
        "torch",
        "scikit-learn",
        "numpy",
        "stable-baselines3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",  # or whatever minimum you need
)
