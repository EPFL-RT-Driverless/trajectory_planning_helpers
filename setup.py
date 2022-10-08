import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()
    # remove lines that start with #
    requirements = [
        r
        for r in requirements
        if not (r.startswith("#") or r.startswith("-e git+") or r.startswith("git+"))
    ]

setuptools.setup(
    name="trajectory-planning-helpers",
    version="2.0.2",
    url="https://github.com/EPFL-RT-Driverless/trajectory_planning_helpers",
    author="Alexander Heilmeier, Tim Stahl, Fabian Christ, Tudor Oancea",
    description="Useful functions used for path and trajectory planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages("trajectory_planning_helpers"),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable"
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Private :: Do Not Upload",
    ],
)
