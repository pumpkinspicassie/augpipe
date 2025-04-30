from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="augpipe",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    author="Zhuoli Lu",
    description="A flexible image augmentation pipeline for OCR",
)
