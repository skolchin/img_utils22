"""
Setup routine for img_utils22 package
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="img_utils22",
    version="1.0.0",
    author="Kol",
    author_email="skolchin@gmail.ru",
    description="Image utility routines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gihub.com/skolchin/img_utils22",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
)
