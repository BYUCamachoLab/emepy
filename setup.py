from setuptools import setup

setup(
    name="emepy",
    version="0.2.4",
    description="Eigenmode Expansion Python",
    url="https://github.com/BYUCamachoLab/emepy.git",
    author="Ian Hammond",
    author_email="ihammond@byu.edu",
    license="MIT",
    packages=["emepy"],
    install_requires=[
        "simphony",
        "numpy",
        "matplotlib",
        "scipy",
        "sklearn",
        "pandas",
        "electromagneticpython @ git+git://github.com/lbolla/EMpy@4bf1b01#egg=electromagneticpython",
    ],
    zip_safe=False,
)
