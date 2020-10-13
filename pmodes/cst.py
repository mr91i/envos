    #### Constant Physical parameter ####
pi       = 3.141592653589793238462643383279502884197169399375105820974
c        = 2.99792458e10
h        = 6.62607004e-27
G        = 6.67408e-8
e        = 4.803205e-10
m_e      = me = 9.10938291e-28
m_n      = mn = 3.88549512e-24
sigma_SB = 5.670367e-5
a        = 4 * sigma_SB/c
k_B      = kB = k = 1.38065e-16
amu      = 1.660468e-24
sigma_en = 1e-15
AU       = au = 1.49597870700e13
pc       = 3.0856775814913674e+18
#pc      = 206264.806247*au #3.085677581e18
m        = 100
Angstrom = 1e-8
inch     = 2.54
bar      = 1e6
eV       = 1.60217662e-12 #erg
J        = 1e7            #erg
Msun     = 1.988435e33
Lsun     = 3.848e33
Rsun     = 6.957e10   #0.00465    #au
Tsun     = 5772       #K
Rj       = 6.9911e9   #cm
Mj       = 1.89813e30 #g
year     = yr = 3.1454e7
Myr      = 3.1454e13
Mpyr     = Msun/year
Mj_p_Myr = Mj/(1e6*year)
gamma    = 7.0/5.0
cV       = kB/mn/(gamma-1) # heat capacity per mass
#R       = 8.31446261815324e7
#CV      = 5/2*R
kms      = 1e5


if __name__ == '__main__':
    import cst
    print( vars(cst) )

