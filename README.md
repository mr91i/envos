# envos.py

**En**velope **O**bservation **S**imulator.
This code executes synthetic observation with creating an envelope model.

<!--
[](
project name list\
pmodes\
envos: Envelope Observation Simulator\
endo: Envelope-Disk ystem for Observation\
osimen: Observation Simulator for Model of Envelope\
osiire: Observation SImulator for Infalling Rotating Envelope\
obsend: pipeline for synthetic OBServation of ENvelope Disk systems\
obento: pipeline for synthetic OBServation of ENvelope Disk systems\
somen : Synthetic Observation for  Model of Envelope\
oden: Observation simulatior for Disk-Envelope system
)
-->

## Setup
1. Download envos from this github repository:  
`git clone https://github.com/mr91i/envos.git` 

2. Execute setup.py in envos directory:
`python setup.py install`


3. To run this code, one need to install RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/)) and radmc3dPy. Download RADMC-3D and radmc3dPy from [github](https://github.com/dullemond/radmc3d-2.0) and install them following [a HTML manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) or [a PDF version](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3d.pdf).

4. Put the opacity table and molecular line data that you want to use in RADMC-3D, into `storage` directory(e.g., `storage/dustkappa_silicate.inp`, `storage/molecule_c18o.inp`)dustkappa_XXX.inp and molecule_XXX.inp file can be found in directories of RADMC-3D package. One can also get any molecule_XXX.inp from [*Leiden Atomic and Molecular Database*](https://home.strw.leidenuniv.nl/~moldata/).
 


## Tutorial
Copy `example/simple_run.py` to your working directory. Read and run the example file.


