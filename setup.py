from setuptools import find_packages, setup

setup(
    name="triton_kernels",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
)
