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


with open("requirements.txt", "rt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="leo_vetter",
    version=get_version("leo_vetter/__init__.py"),
    author="Michelle Kunimoto",
    author_email="michelle.kunimoto@gmail.com",
    url="https://github.com/mkunimoto/LEO-vetter",
    license="GPLv3",
    packages=["leo_vetter"],
    include_package_data=True,
    install_requires=requirements,
)
