# ***envos***

## 1. Introduction
**Env**elope **O**bservation **S**imulator, developed by Shoji Mori.
This code executes synthetic observation by calculating physical model of young circumstellar systems (i.e. envelope and disk).

I really welcome improvements and requests from users. 

## 2. Features
- Density and velocity structures
- Temperature structure is calculated consistently with the density structure given by user.
- Calculation of temperature structure and sysnthetic observation is done by RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/); github: [https://github.com/dullemond/radmc3d-2.0](https://github.com/dullemond/radmc3d-2.0)), which is commonly used in astronomical studies.
- All source codes are written in Python3 (ver. >= 3.6).

## 3. Requirements
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

## 4. Setup
### 4.1 Install *RADMC-3D* and *radmc3dPy*
1. Download the installing source code from [github](https://github.com/dullemond/radmc3d-2.0).
`git clone https://github.com/dullemond/radmc3d-2.0.git`

2. Install RADMC-3D following [a HTML manual](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/index.html) or [a PDF version](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/radmc3d.pdf).
    1. `cd radmc3d-2.0/src`
    2. Edit Makefile if you need
    3. Generating an executable file `radmc3d`
       `make`
    5. Distribute `radmc3d` and radmc_tools in `$HOME/bin` in the default
       `make install`
    7. Add the path `$HOME/bin` to your \*rc file (e.g., ~/.bashrc)
       e.g. `echo export PATH=$HOME/bin:$PATH >> ~/.bashrc ` in command line
    6. Reload the \*rc file
       `source ~/.bashrc`

3. Check if `radmc3d` works following the RADMC-3D manual, executing example runs
    1. Move to `radmc3d-2.0/examples/run_simple_1`
    2. `python problem_setup.py`
    4. `radmc3d mctherm`
    5. Also `radmc3d mctherm setthreads 2` to check multithreading

4. Install `radmc3dPy` also following the manual.
    1. Move to `radmc3d-2.0/python/radmc3dPy`
    2. Execute setup.py: `python setup.py install --user`
       \*\* This `python` should also be python3. If you use python2, `radmc3dPy` will be installed into python2 libraries.

5. Check if `radmc3dPy` works (e.g. execute `python -c "import radmc3dPy"` in command line)


### 4.2 Install *envos*
1. Download `envos` from this github repository
`git clone https://github.com/mr91i/envos.git`

2. Check if it works e.g., execute an example script `python3 example_run.py`.


<!--

    * Put the dust opacity table and molecular line table that you want to use in RADMC-3D, into a directory.  Initially, (e.g., `storage/dustkappa_MRN20.inp`, `storage/molecule_c18o.inp`)dustkappa_XXX.inp and molecule_XXX.inp file can be found in directories of RADMC-3D package. One can also get any molecule_XXX.inp from [*Leiden Atomic and Molecular Database*](https://home.strw.leidenuniv.nl/~moldata/))

-->

## 5. Tutorial
Read and run `example_run.py`.


## 6. Parameters
Easiest way to controll envos is to use a class `envos.Config`.
In `envos.Config`, basic parameters user uses in envos are set.
Users create the instance of the `envos.Config` class setting below parameters and pass it to envos's classes.

Exapmple -- < *parameter name* >: < *type of parameter* > = < *default value* > 

### 6.1 General input
- `run_dir` : *str* = "./"  
The location of the executing directory. This directory is made automatically if not exists.  
- `fig_dir` : *str* = "./run/fig"  
The location of the directory in which the produced figures are saved. If not set, fig_dir is located in run_dir.
- `n_thread` : *int* = 1  
Number of threads used in RADMC-3D calculation. If this is > 2 and OpenMP is available, OpenMp is used for the radiative transfer calculations for thermal structure and line observation. Default is 1. 

### 6.2 Grid parameters
To generate a computational grid, one needs to set r, θ, φ coordinates as a list (1), or to set some parameters on grid (2).  
(1). Directly set axes
- `ri_ax`, `ti_ax`, `pi_ax` : *ndarary* or *list*  
Coordinates of the cell interface in r, θ, φ directions. When these parameters are set, below parameters are neglected.

(2). Generate axes inputting parameters
- `rau_in`, `rau_out` : *float*  
Coordinate of the cell interface of the inner and outer boundary in r direction, in au.  
- `theta_in`, `theta_out` : *float*  
Coordinate of the cell interface of the inner and outer boundary in θ direction, in radian. In default, this is from 0 to pi/2.  
- `phi_in`, `phi_out` : *float* 
Coordinate of the cell interface of the inner and outer boundary in φ direction, in radian. In default, this is from 0 to 2pi.  
- `nr`: *int*  
Number of cell in r direction.
- `ntheta`: *int*  
Number of cell in θ direction.
- `nphi`: *int*    
Number of cell in φ direction. 　
- `logr`: *bool* = True  
Set the r coordinate to be a logarithmic scale.
- `dr_to_r`: *float*  = None 
When logr is True and this parameter is set, the fraction of the cell length in the r direction to r is set to be `dr_to_r`. For example, when `dr_to_r` is 0.1, the cell size in the r-direction at 100 au is 1 au.
- `aspect_ratio`:  *float* = None
Ratio of the cell size in the θ or φ direction to dr.


### 6.3 Model parameters
Basically the kinetic structure of the UCM model is basically given by three parameters (see below for the definitions): `Mdot_smpy`, `Ms_Msun`, and `Omega`. However, `Mdot_smpy` can be given by `T`, `Ms_Msun` can be given by `t_yr`, and `Omega` can be given by `CR_au` or `jmid`.
- {`Mdot_smpy`, `T`}, {`Ms_Msun`, `t_yr`}, {`Omega`, `CR_au`, `jmid`} : *float*  
    - `Mdot_smpy`\[Msun/yr\] -- accretion rate
    - `T`\[K\] -- temperature of the molecular cloud core
    - `Ms_Msun` \[Msun\] -- central stellar mass
    - `t_yr`\[yr\] -- time from begining of the collapse 
    - `Omega`\[s^-1\] -- angular velocity of the parental cloud core   
    - `CR_au`\[au\] -- the centrifugal radius
    - `jmid`\[cm^2 s^-1\] -- specific angular momentum of the equatorial plane
    
- `meanmolw`: *float* = 2.3  
Mean molecular weight.
- `cavangle_deg`: *float* = 0.0  
Polar angle \[deg\] within which the density is deprecated, mimicking outflow cavities.
- `inenv`: *str* or *InnerEnvelope* = "UCM"  
The model used for the inner region of the envelope 
- `outenv`: *str* or *OuterEnvelope* = None  
The model used for the outer region of the envelope 
- `disk`: *str* or *Disk* = None  
The model used for the disk region 
- `rot_ccw`
If True, the rotation velocity becomes in the opposite direction

### 6.4 RADMC-3D parameters
- `nphot`: *int* = 1e6  
Number of photons used for thermal calculation
- `f_dg`
Dust-to-gas mass ratio
- `opac`
Name of opacity table. xxx in storage/dustkappa_xxx.inp.
- `Lstar_Lsun`
星の光度\[Lsun\]
- `mfrac_H2`
H2分子がガス中の質量を占める割合。デフォルトは0.74
- `molname`
使用する輝線テーブルの名前。storage/molecule_xxx.inpのxxx
- `molabun`
H2分子に対する注目する分子の存在度
- `iline`
テーブル内の何番目の遷移か。1始まり。
- `scattering_mode_max`
1以上でscatteringあり。詳しくはRADMC3Dのマニュアルを参照。
- `mc_scat_maxtauabs`
scatteringする光子の最大光学的深さ。詳しくはRADMC3Dのマニュアルを参照。
- `tgas_eq_tdust`
ガス温度とダスト温度を一緒にする。デフォルトでTrue。Falseは未対応。

### 6.5 Observarion parameters
- `dpc`: float
Distance from observer to object.
- `size_au`: float
Length of one side of a square observation area, \[au\].
When `size_au` is given, `sizex_au` and `sizex_au` are neglected.
- `sizex_au`: float
縦の観測範囲\[au\]
- `sizey_au`
横の観測範囲\[au\]
- `pixsize_au`
Pixel size \[au\]
- `vfw_kms`
Total velocity width, \[km/s\].
- `dv_kms`
Velocity resolution, \[km/s\]
- `convmode`: str
    - "normal": a standard convolution function in *astropy*
    - "fft": a convolution function in *astropy*, using fast foulier transform. This is faster than `normal` but requires bigger memory.
    - "scipy": a convolution function in *scipy*, slightly faster and cheeper than `fft`.
- `beam_maj_au`
ビームの長半径\[au\]
- `beam_min_au`
ビームの短半径\[au\]
- `vreso_kms`
Full velocity width of convolution \[km/s\]
- `beam_pa_deg`
ビームの位置角\[deg\]
- `incl`
Angle \[deg\] between polar axis and line of sight (incl=0: face-on, incl=90: edge-on)
- `phi`
経度\[deg\]
- `posang`
カメラの位置角\[deg\]

## 7. Functions/Classes



