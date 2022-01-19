import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


direct = []
# "electromagneticpython @ git+git://github.com/lbolla/EMpy@4bf1b01#egg=electromagneticpython",
# "electromagneticpythongpu @ git+git://github.com/hammy4815/EMpy_gpu@master#egg=electromagneticpythongpu",


def get_install_requires():
    with open("requirements.txt", "r") as f:
        return direct + [line.strip() for line in f.readlines() if not line.startswith("-")]


setup(
    name="emepy",
    version="0.2.4",
    url="https://github.com/BYUCamachoLab/emepy",
    license="MIT",
    author="Ian Hammond",
    author_email="ihammond@byu.edu",
    description="Eigenmode Expansion Python",
    # long_description=read("README.md"),
    packages=find_packages(exclude=("tests",)),
    install_requires=get_install_requires(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
)
