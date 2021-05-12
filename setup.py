import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fms_analysis_tool",
    version="0.0.1",
    author="jkyu",
    author_email="jkyu@stanford.edu",
    description="Tool for extracting, cleaning, and analyzing FMS simulation data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkyu/fat",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
