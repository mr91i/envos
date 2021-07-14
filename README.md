# **envos**

## 1. Introduction
**Env**elope **O**bservation **S**imulator, developed by Shoji Mori.
This code executes synthetic observation with calculating physical model of young circumstellar systems (i.e. envelope and disk).

## 2. Features
- Density and velocity structures
- Temperature structure is calculated consistently with the density structure given by user.
- Calculation of temperature structure and sysnthetic observation is done by RADMC-3D (Dullemond et al. 2012; website: [https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/](https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/)), which is commonly used in astronomical studies.
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


### 6.1 General input
- `run_dir`
実行ディレクトリの位置を指定する
- `fig_dir`
画像を保存するディレクトリを指定する
- `n_thread`
使用するスレッドの数。デフォルトは1。2以上に設定した場合、温度計算と模擬観測計算でOpenMPを使用する。ただしRADMC-3Dのインストールの際に、OpenMPの利用を可能にしておく必要がある。

### 6.2 Grid parameters
グリッドの生成方法は２通りある。一つはメッシュのr, θ, φ座標をリストで渡す方法。
もう一つはパラメーターからグリッドを生成する。
1. Directly set axes
- `ri_ax`, `ti_ax`, `pi_ax`
それぞれメッシュのr, θ, φ座標のlist-like object。cell centerの座標ではなく、cell interfaceの座標。これらが与えられた時以下のパラメーターは無視される。
2. Generate axes inputting parameters
- `rau_in`, `rau_out`
r 方向の内側/外側境界の座標。単位はau。
- `theta_in`, `theta_out`
θ方向の内側/外側境界の座標。単位はradian。デフォルトは0からπ/2。
- `phi_in`, `phi_out`
φ方向の内側/外側境界の座標。単位はradian。デフォルトは0から2π。
- `nr`
r方向のセル数
- `ntheta`
θ方向のセル数
- `nphi`
φ方向のセル数
- `dr_to_r`
logrがTrueの時、あるrにおけるr方向のセルの長さとrの比がdr_to_rになる。例えば、dr_to_r = 0.1なら、100auにおけるr方向の解像度は1au。
- `aspect_ratio`
r方向のセルの長さ(= dr)に対する、θ方向のセルの長さ(=r dθ)の比を与える。
- `logr`
Bool. r方向のメッシュをlog scaleにする。デフォルトでTrue。

### 6.3 Model parameters
- `T`, `CR_au`, `Ms_Msun`, `t_yr`, `Omega`, `jmid`, `Mdot_smpy`
UCMモデルは３つのパラメーターから与えられる: 降着率\[Msun/yr\] `Mdot_smpy`、中心星質量\[Msun\] `Ms_Msun`、分子雲コアの回転角速度\[s^-1\] `Omega`.
ただし、Mdotは分子雲コアの温度[K] `T`から、`Ms_Msun`は収縮開始後の時間\[yr\] `t_yr`から、Omegaは遠心力半径\[au\] `CR_au`や赤道面の比角運動量`jmid` \[cm^2 s^-1\]からでも与えられる。
- `meanmolw`
Mean molecular weight。デフォルトは 2.3。
- `cavangle_deg`
Polar angle \[deg\] within which the density is deprecated, mimicking outflow cavities.
- `inenv`
"UCM"かInnerEnvelopeインスタンス
- `outenv`
   “TSC”かOuterEnvelopeインスタンス
- `disk`
   “exptail”かDiskインスタンス
- `rot_ccw`
回転方向を反転させるかどうか。デフォルトはFalse。

### 6.4 RADMC-3D parameters
- `nphot`
温度計算時に使われるphoton数
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



