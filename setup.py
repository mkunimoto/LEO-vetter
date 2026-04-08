import codecs
import os.path

from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "brokenaxes",
    "batman-package",
    "lmfit",
]

setup(
    name="leo_vetter",
    version=get_version("leo_vetter/__init__.py"),
    author="Michelle Kunimoto",
    author_email="michelle.kunimoto@gmail.com",
    maintainer="Michelle Kunimoto",
    maintainer_email="michelle.kunimoto@gmail.com",
    url="https://github.com/mkunimoto/LEO-Vetter",
    license="GPLv3",
    license_files=["LICENSE.txt"],
    description="LEO-Vetter: Automated Vetting for TESS Planet Candidates",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=["leo_vetter"],
    package_data={"leo_vetter": [
        "data/claret_2017_table15.csv.gz",
        "data/claret_2017_table25.csv.gz",
    ]},
    install_requires=INSTALL_REQUIRES,
    classifiers=CLASSIFIERS,
)

