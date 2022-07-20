import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gmpi",
    version="1.0.0",
    description="Generative MPI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apple/ml-gmpi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    py_modules=["gmpi"],
)
