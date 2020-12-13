import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-pepr",
    version="0.1a1",
    author="University of Luebeck: ITS KI-Lab Group",
    author_email="jonas.sander@student.uni-luebeck.de",
    description="PePR is a library for pentesting the privacy risk and robustness of machine learning models.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/hallojs/ml-pepr",
    packages=['pepr'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)