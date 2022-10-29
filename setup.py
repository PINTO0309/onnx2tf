
from setuptools import setup, find_packages
from os import path
import re

package_name="onnx2tf"
root_dir = path.abspath(path.dirname(__file__))

with open("README.md") as f:
    long_description = f.read()

with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    version=version,
    description=\
        "Self-Created Tools to convert ONNX files (NCHW) to TensorFlow format (NHWC). "+
        "The purpose of this tool is to solve the massive Transpose extrapolation problem in onnx-tensorflow (onnx-tf).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katsuya Hyodo",
    author_email="rmsdh122@yahoo.co.jp",
    url="https://github.com/PINTO0309/onnx2tf",
    license="MIT License",
    packages=find_packages(),
    platforms=["linux", "unix"],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            "onnx2tf=onnx2tf:main"
        ]
    }
)
