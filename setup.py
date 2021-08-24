from setuptools import setup

setup(
    name="emepy",
    version="0.2.0",
    description="Eigenmode Expansion Python",
    url="https://github.com/BYUCamachoLab/emepy.git",
    author="Ian Hammond",
    author_email="ihammond@byu.edu",
    license="MIT",
    packages=["emepy"],
    install_requires=["simphony","numpy","matplotlib","scipy","sklearn","pandas","electromagneticpython"],
    zip_safe=False,
)

