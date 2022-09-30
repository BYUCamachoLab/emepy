# EMEPy 1.1.0 Eigenmode Expansion Tool for Python

<p align="center">
    <img src="docs/images/logo.png" alt="drawing" width="500"/>
</p>



### An open source tool for simulating EM fields in python using the Eigenmode Expansion Method (EME). Employs neural networks as a means for accelerating the cross sectional field profile generation process.

## Installation

Clone the repo

    git clone git@github.com:BYUCamachoLab/emepy.git

Install the development version of emepy using pip:

    pip install -e .

Or collect the most recent publication to pip

    pip install emepy

## Docs

Read the docs [here](https://emepy.readthedocs.io/en/latest/).

## ANN Models

Optionally, the user can download and use our neural networks for fundamental TE generation.

The neural network models can be found [here](https://byu.box.com/s/xtpp2h8vfwp4l07wdl5559j3vnip5cqj). Simply download the three folders (Hy_chunks, Hx_chunks, neff_pickle) and place them under path/to/repo/emepy/models/ 

This will look like:

    .../emepy/emepy/models/Hy_chunks/
    .../emepy/emepy/models/Hx_chunks/
    .../emepy/emepy/models/neff_pickle/

## BibTeX citation
```
@article{emepy,
    author = {Ian M. Hammond and Alec M. Hammond and Ryan M. Camacho},
    journal = {Opt. Lett.},
    number = {6},
    pages = {1383--1386},
    publisher = {OSA},
    title = {Deep learning-enhanced, open-source eigenmode expansion},
    volume = {47},
    month = {Mar},
    year = {2022},
    doi = {10.1364/OL.443664},
}
``` 
