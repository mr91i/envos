# **envos**

**Env**elope **O**bservation **S**imulator.  
This code executes synthetic observation with calculating physical model of young circumstellar system (i.e. envelope and disk). 

## Features

- Temperature structure is calculated consistently with the density structure given by user.
- Calculation of temperature structure and sysnthetic observation is done by RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/)), which is commonly used in astronomical studies. 
- All source codes are written in Python3.

## Requirements
- Python packages (automatically installed when installing envos)
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
1. Install RADMC-3D and radmc3dPy. 

     1. Download the installing source code from [github](https://github.com/dullemond/radmc3d-2.0).
      
     2. Install RADMC-3D following [a HTML manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) or [a PDF version](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3d.pdf).

     3. Check if it works following the RADMC-3D manual, e.g. execute a example run.

     4. Install radmc3dPy also following the manual. 

     5. Check if it works  
     e.g., execute `python -c "import radmc3dPy"` in command line.   


2. Install envos.

    1. Download envos from this github repository  
    `git clone https://github.com/mr91i/envos.git` 

    2. Execute setup.py in envos directory  
    `python setup.py install`
    
    3. Check if it works, e.g., execute a example script in `examples` ditectory.


<!--

    * Put the dust opacity table and molecular line table that you want to use in RADMC-3D, into a directory.  Initially, (e.g., `storage/dustkappa_MRN20.inp`, `storage/molecule_c18o.inp`)dustkappa_XXX.inp and molecule_XXX.inp file can be found in directories of RADMC-3D package. One can also get any molecule_XXX.inp from [*Leiden Atomic and Molecular Database*](https://home.strw.leidenuniv.nl/~moldata/))

-->



 


## Tutorial
Copy `example/simple_run.py` to your working directory. Read and run the example file.


