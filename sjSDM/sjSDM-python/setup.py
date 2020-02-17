import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sjSDM_py", 
    version="0.0.1",
    author="Maximilian Pichler",
    author_email="Maximilian.Pichler@ur.de",
    description="jSDM package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaximilianPi/fajsm/inst/python/fajsm_py",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: GPL-3 License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ]
)