import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sjSDM_py", 
    version="0.1.2",
    author="Maximilian Pichler",
    author_email="Maximilian.Pichler@ur.de",
    description="jSDM package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheoreticalEcology/s-jSDM",
    packages=setuptools.find_packages(),
    install_requires = [
        "numpy"
    ],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ]
)
