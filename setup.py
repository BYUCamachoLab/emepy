from setuptools import setup

setup(
    name="emepy",
    version="0.1",
    description="Eigenmode Expansion Python",
    url="https://github.com/BYUCamachoLab/emepy.git",
    author="Ian Hammond",
    author_email="ihammond@byu.edu",
    license="MIT",
    packages=["emepy"],
    install_requires=["simphony", "numpy", "matplotlib", "pickle", "random", "ElectroMagneticPython"],
    zip_safe=False,
)

