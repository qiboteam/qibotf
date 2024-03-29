# Installation script for python
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.install import install
from setuptools.dist import Distribution
import subprocess
import os
import re
import sys

PACKAGE = "qibotf"


# extract tensorflow version
try:
    import tensorflow as tf
    TF_VERSION = tf.__version__
except:
    raise ModuleNotFoundError('Please install TensorFlow before qibotf.')


# replace tensorflow placehold version in __init__.py
PLACEHOLDER = os.path.join("src", PACKAGE, "__init__.py.in")
VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
with open(PLACEHOLDER, 'r') as f:
    content = f.read()
    content = content.replace('TF_VERSION', TF_VERSION)
with open(VERSIONFILE, 'w') as f:
    f.write(content)


# Returns the version
def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


# Custom compilation step
class Build(_build_py):
    def run(self):
        if os.name != 'nt':  # skip windows
            commands = [
                ["make",
                 "-C", "src/qibotf/custom_operators/",
                 "clean"],
                ["make", "-j", "%s" % os.cpu_count(),
                 "-C", "src/qibotf/custom_operators/"], ]
            for command in commands:
                if subprocess.call(command) != 0:
                    sys.exit(-1)
        _build_py.run(self)


# Register wheel with binary version
class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels.
    """

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False

# Patch to generate manylinux2010 packages


class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=PACKAGE,
    version=get_version(),
    description="Simulation tools based on tensorflow.",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibotf",
    packages=find_packages("src"),
    package_dir={"": "src"},
    cmdclass={"build_py": Build, "install": InstallPlatlib},
    package_data={"": ["*.so", "*.out"]},
    include_package_data=True,
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=requirements + ['qibo>=0.1.7rc0'],
    python_requires=">=3.7.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
