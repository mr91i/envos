# **envos**

**Env**elope **O**bservation **S**imulator, developed by Shoji Mori.    
This code executes synthetic observation with calculating physical model of young circumstellar systems (i.e. envelope and disk). 

## Features
- Density and velocity structures  
- Temperature structure is calculated consistently with the density structure given by user.
- Calculation of temperature structure and sysnthetic observation is done by RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/)), which is commonly used in astronomical studies. 
- All source codes are written in Python3 (ver. >= 3.6).

## Requirements
- Python packages
    - numpy
    - scipy
    - dataclasses
    - pandas
    - astropy
    - matplotlib (for using RADMC-3D)

- To use RADMC-3D
     - Fortran compiler (e.g. `gfortan`, `intel fortran`)
     - (optional) Fortran openMP library

## Setup
### 1. Install RADMC-3D and radmc3dPy. 
1. Download the installing source code from [github](https://github.com/dullemond/radmc3d-2.0).  
`git clone https://github.com/dullemond/radmc3d-2.0.git`

2. Install RADMC-3D following [a HTML manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) or [a PDF version](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3d.pdf).  
    1. `cd radmc3d-2.0/src`
    2. Edit Makefile if you need
    3. `make`: generating an executable file `radmc3d`
    4. `make install`: distribute `radmc3d` and radmc_tools in `$HOME/bin` in the default
    5. Add the path `$HOME/bin` to your \*rc file (e.g., ~/.bashrc)  
       e.g. `echo export PATH=$HOME/bin:$PATH >> ~/.bashrc ` in command line
    6. reread the \*rc file e.g. `source ~/.bashrc` 

3. Check if `radmc3d` works following the RADMC-3D manual, e.g. execute a example run.

4. Install `radmc3dPy` also following the manual. 
    1. move to `radmc3d-2.0/python/radmc3dPy`  
    2. execute setup.py: `python setup.py install --user`  
  
5. Check if `radmc3dPy` works (e.g. execute `python -c "import radmc3dPy"` in command line)   


### 2. Install envos.
1. Download `envos` from this github repository  
`git clone https://github.com/mr91i/envos.git` 

2. Check if it works e.g., execute an example script `python3 example_run.py`.


<!--

    * Put the dust opacity table and molecular line table that you want to use in RADMC-3D, into a directory.  Initially, (e.g., `storage/dustkappa_MRN20.inp`, `storage/molecule_c18o.inp`)dustkappa_XXX.inp and molecule_XXX.inp file can be found in directories of RADMC-3D package. One can also get any molecule_XXX.inp from [*Leiden Atomic and Molecular Database*](https://home.strw.leidenuniv.nl/~moldata/))

-->

## Tutorial
Read and run `example_run.py`.




