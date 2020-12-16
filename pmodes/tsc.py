import numpy as np
from scipy import integrate, interpolate, optimize
from pmodes import cst
import pmodes.myplot as myp
import logging
logger = logging.getLogger(__name__)

class solve_TSC:
    def __init__(self, t, cs, Omega, run=True, r_crit=None):
        self.t = t
        self.cs = cs
        self.Omega = Omega
        self.tau = t * Omega
        self.f_rho0 = None
        eps = 1e-10
        self.r_crit = r_crit
        # large-r to small-r
        self.x = np.logspace(np.log10(1/self.tau)+1, np.log10(self.tau**2)-1, 2000)
        self.xout = self.x[self.x>1]
        self.xin = self.x[self.x<=1]
        logger.info(f"xin is {self.x[-1]*cs*t/cst.au} au")
        ## Run
        if run:
            logger.debug("Start all_run")
            self.solve_all()

    def solve_all(self):
        # self.solve_Spherically_Symmetric_Solution()
        self.solve_0thorder(solve_type=2)
        self.solve_Qualdrapolar(search_K=False)
        self.solve_Monopolar()
        # self.plot_TSC_figs()
        self.set_functions()

    def solve_Spherically_Symmetric_Solution(self):
        # Solve Spherically symmetric solutions
        # unpertubed rotating equilibria
        # see Eq (11) in TSC with Psi = Psi(ksi)
        def f(x, y): # y = [ dPhi, Phi]
            dPhi = y[1]
            ddPhi = - 2*y[1]/x - 2*np.exp(y[0])/x**2 + 2*(1+x**2)/x**2
            return dPhi, ddPhi
        logger.debug("Start solve_Spherically_Symmetric_Solution")

        ksi = np.logspace(-2, 2, 200)
        sol = integrate.solve_ivp(f, (ksi[0], ksi[-1]), (0.5*ksi[0], 0.25*ksi[0]**2), t_eval=ksi, method='BDF')
        self.rho_eq = np.exp(sol.y[0])*ksi**(-2)
        self.M_eq = integrate.cumtrapz(np.exp(sol.y[0]), ksi, initial=0) + ksi[0]
        self.ksi_eq = ksi
        self.f_rho0 = self.fnize_loglog(ksi, np.exp(sol.y[0])/ksi**2,extrapolate=True)
        # rho/(Omg^2/2piG) = a^2 /Omg^2r^2 e^Phi = ksi^-2 e^Phi

    def solve_0thorder(self, solve_type=2):
        logger.debug("Start solve_0thorder")
        def f_0thorder(x, y):
            V_0, al_0 = y
            dV_0 =      (al_0*(x - V_0) - 2/x)*(x-V_0)/((x-V_0)**2 -1) # if x < 1 else 0
            dal_0 = al_0*(al_0 - 2/x*(x - V_0))*(x-V_0)/((x-V_0)**2 -1) #if x < 1 else -4/x**3
            return dV_0, dal_0

        if solve_type==1:
            sol = integrate.solve_ivp(f_0thorder, (xin[0], xin[-1]), [0, 2], t_eval=self.xin,
                                  method='BDF', rtol=1e-8, atol=[1e-8, 1e-8])
            V_0  = np.hstack((np.zeros_like(self.xout), sol.y[0]))
            al_0 = np.hstack((2/self.xout**2, sol.y[1]))
            sol0x = self.x
        elif solve_type==2:
            sol = integrate.solve_ivp(f_0thorder, (self.x[0], self.x[-1]), [0, 2/self.x[0]**2], t_eval=self.x,
                                      method='BDF', atol=1e-12, rtol=1e-12)
            V_0 = sol.y[0]
            al_0 = sol.y[1]
            sol0x = self.x

        logger.debug(sol)
        logger.info(f"m0 is {sol0x[-1]**2 * al_0[-1]*(sol0x[-1]-V_0[-1])}")

        self.V_0 = V_0
        self.al_0 = al_0
        self.Gamma_mid0 = 0.25 * (sol0x**2*al_0*(sol0x-V_0))**2
        self.f_Gamma_mid0 = self.fnize(self.x, self.Gamma_mid0)
        self.save_0th_solution(al_0, V_0)

    def save_0th_solution(self, al_0, V_0, mode=1):
        logger.debug("Start solve_0thorder")
        self.f_al0 = self.fnize(self.x, al_0)
        self.f_V0 = self.fnize(self.x, V_0)
        dV0dx = np.gradient(V_0, self.x)
        dal0dx = np.gradient(al_0, self.x)
        f_dal0dx = self.fnize(self.x, dal0dx)
        f_dV_0dx = self.fnize(self.x, dV0dx)
        def f_dal0dx_2(x):
            al_0 = self.f_al0(x)
            V_0 = self.f_V0(x)
            return al_0*(al_0 - 2/x*(x - V_0))*(x-V_0)/((x-V_0)**2 -1)

        def f_dV0dx_2(x):
            al_0 = self.f_al0(x)
            V_0 = self.f_V0(x)
            return (al_0*(x - V_0) - 2/x)*(x-V_0)/((x-V_0)**2 -1)

        self.f_dal0dx = f_dal0dx_2 if mode==1 else f_dal0dx
        self.f_dV0dx = f_dV0dx_2  if mode==1 else f_dV0dx

    def solve_Qualdrapolar(self, search_K=True):
        def f_QuadraPolar(x, vals): ## Eq 63
            al, V, W, Q, P = vals
            al_0    = self.f_al0(x)    #if x < 1 else 2/x**2
            V_0     = self.f_V0(x)     #if x < 1 else 0
            dal0dx = self.f_dal0dx(x) #if x < 1 else -4/x**3
            dV0dx  = self.f_dV0dx(x)  #if x < 1 else 0
            Psi = -x**2/3 - Q/x**3 - x**2*P
            dPsidx = -2/3*x + 3*Q/x**4 - 2*x*P
            m = x**2 * al_0 * (x - V_0)
            A = al/x**2 * (2*x*V_0 + x**2*dV0dx ) + V/x**2 * (x**2*dal0dx + 2*x*al_0) - 6*al_0*W/x
            B = -al/al_0**2*dal0dx  + ( 2 + dV0dx )*V + ( dPsidx + 2/3*(m/2)**4/x**3 )
            dal = 1/( (x-V_0)**2 -1 ) * ((x-V_0)*A + al_0*B)
            dV = 1/( (x-V_0)**2 -1 ) * ((x-V_0)*B + 1/al_0/2*A)
            dW = 1/x/(x-V_0) * ((2*x+V_0)*W + al/al_0 + (Psi+(m/2)**4/(3*x**2)))
            dQ =  0.2*x**4*al
            dP = -0.2*al/x
            return dal, dV, dW, dQ, dP

        def f_QuadraPolar_out(x, vals):
            al, V, W, Q, P = vals
            dal = 1/(x**2-1) * (-12/x**2*W + 2/x**2*( x*al + 2*V + 3*Q/x**4 - 2*x*P ))
            dV = 1/(x**2-1) * (x*(x*al + 2*V + 3*Q/x**4 - 2*x*P) -6/x*W)
            dW = 2/x*W + 0.5*al - Q/x**5 - P
            dQ = 0.2*x**4*al
            dP = -0.2*al/x
            return dal, dV, dW, dQ, dP

        def sol_Q(K):
            opt = {"method":{0:'RK45', 1:'BDF', 2:'Radau', 3:'DOP853', 4:"LSODA"}[1], "rtol":1e-8, "atol":1e-8}
            xini = self.x[0]
            y0 = (-2/7*K*xini**(-7), -K/2*xini**(-4), K/6*xini**(-4), K*(1+xini**(-2)/35), -2/245*K*xini**(-7))
            sol_o = integrate.solve_ivp(f_QuadraPolar_out, (self.xout[0], self.xout[-1]), y0, t_eval=self.xout, **opt)
            Delta_Q = - sol_o.y[1][-1]/3
            alin = sol_o.y[0][-1] + 2*Delta_Q
            Vin = sol_o.y[1][-1] + 2*Delta_Q
            Win = sol_o.y[2][-1]
            Qin = sol_o.y[3][-1]
            Pin = sol_o.y[4][-1]
            cond_outerEW = sol_o.y[0][-1] + 2*sol_o.y[1][-1] - 6*sol_o.y[2][-1] + 3*sol_o.y[3][-1] - 2*sol_o.y[4][-1]
            sol_i = integrate.solve_ivp(f_QuadraPolar, (self.xin[0], self.xin[-1]), (alin, Vin, Win, Qin, Pin), t_eval=self.xin, **opt)
            soly = np.hstack((sol_o.y, sol_i.y))
            solx = np.hstack((sol_o.t, sol_i.t))
            return solx, soly, Delta_Q

        j=0
        def loop_K(K):
            nonlocal j
            j += 1
            K = K[0]
            solx, soly, Delta_Q = sol_Q(K)
            #if plot:
            #    alQ = soly[0] - Delta_Q * solx * self.f_dal0dx(solx)
            #    myp.plot([["-alpha_Q", -alQ], ["-V_Q", -soly[1]], ["-W_Q", -soly[2]], ["Q", soly[3]], ["P", soly[4]]],
            #              f"TSC_solQ_{j:d}", x=solx, xlim=[1e-3, 1e1], ylim=[1e-8, 1e3], **self.figopt)
            return abs(soly[3][-1])

        K0 = -0.0015754247837703587
        if search_K:
            K = optimize.minimize(loop_K, x0=K0).x
        else:
            K = K0
        solx, soly, Delta_Q = sol_Q(K)
        self.Delta_Q = Delta_Q
        self.al_Q = soly[0] - Delta_Q * solx * self.f_dal0dx(solx)
        self.V_Q = soly[1] - Delta_Q * solx * self.f_dV0dx(solx)
        self.W_Q = soly[2]
        self.f_al_Q = self.fnize(self.x, self.al_Q)
        self.f_V_Q = self.fnize(self.x, self.V_Q)
        self.f_W_Q = self.fnize(self.x, self.W_Q)

    def solve_Monopolar(self):
        #def f_MonoPolar(x, vals, fun_al0=None, fun_V0=None ,fun_dal0dx=None, fun_dV0dx=None):
        def f_MonoPolar(x, vals):
            al, V, M = vals
            al_0 = self.f_al0(x)  if x < 1 else 2/x**2
            V_0     = self.f_V0(x)  if x < 1 else 0
            dal0dx = self.f_dal0dx(x) if x < 1 else -4/x**3
            dV0dx  = self.f_dV0dx(x) if x < 1 else 0
            dPhidx = x/6 + M/x**2
            m = x**2*al_0*(x-V_0)
            A = al/x**2 * (2*x*V_0 + x**2*dV0dx) + V/x**2*(2*x*al_0 + x**2*dal0dx)
            B = -al/al_0**2 * dal0dx + (2+dV0dx) * V + (dPhidx - 2/3*(m/2)**4/x**3)
            dal = ((x-V_0)*A + al_0*B)/((x-V_0)**2 -1)
            dV = ((x-V_0)*B + A/al_0)/((x-V_0)**2 -1)
            dM = (al - 0.5)*x**2
            return dal, dV, dM

        y0 = (1/2, 0, 0)
        solM = integrate.solve_ivp(f_MonoPolar, (self.x[0], self.x[-1]), y0, t_eval=self.x, method='BDF', rtol=1e-9, atol=1e-9)
        self.al_M = solM.y[0]
        self.V_M = solM.y[1]
        self.f_al_M = self.fnize(self.x, self.al_M)
        self.f_V_M = self.fnize(self.x, self.V_M)

   # @staticmethod
    def fnize(self, x, y, extrapolate=False, fill_value=None):
        fill_value = 'extrapolate' if extrapolate else fill_value
        f = interpolate.interp1d(x, y, fill_value=fill_value)
        ff = np.frompyfunc(f, 1, 1)
        return ff
        # return lambda x: ff(x).astype(np.float64)

    def fnize_loglog(self, x, y, extrapolate=False, fill_value=None):
        fill_value = 'extrapolate' if extrapolate else fill_value
        logx = np.log(x)
        logy = np.log(np.abs(y))
        logf = interpolate.interp1d(logx, logy, fill_value=fill_value)
        sgn = np.sign(y)
        fsgn = interpolate.interp1d(logx, sgn, fill_value=fill_value)
        def fnized(x):
            logx = np.log(x)
            sgn = np.sign(fsgn(logx))
            return sgn*np.exp(logf(logx))
        return fnized

    def calc_rho(self, r, theta):
        if r < self.r_crit :
            return 0
        x = r/self.cs/self.t
        P2 = 1 - 3/2*np.sin(theta)**2
        y = x*(1 + self.tau**2*self.Delta_Q*P2)
        return 1/(4*cst.pi*cst.G*self.t**2)*( self.f_al0(y) + self.tau**2*(self.f_al_M(y) + self.f_al_Q(y)*P2) )

    def calc_velocity(self, r, theta):
        if r < self.r_crit :
            return 0
        x = r/self.cs/self.t
        P2 = 1 - 3/2*np.sin(theta)**2
        y = x * (1 + self.tau**2*self.Delta_Q*P2)
        vr = self.cs * (self.f_V0(y) + self.tau**2*(self.f_V_M(y) + self.f_V_Q(y)*P2 ))
        vth = self.cs * self.tau**2*self.f_W_Q(y)*(-3*np.sin(theta)*np.cos(theta))
        vph = self.cs**2/(self.Omega*r) * self.f_Gamma_mid0(y) * np.sin(theta) * self.tau**2
        return vr, vth, vph

    def set_functions(self):
        frho_0 = np.frompyfunc(self.calc_rho, 2, 1)
        fvelo_0 = np.frompyfunc(self.calc_velocity, 2, 3)
        self.frho = lambda r, th : frho_0(r, th).astype(np.float64)
        self.fvelo = lambda r, th : [v.astype(np.float64) for v in fvelo_0(r, th)]

def plot_TSC_figs(cls):
    figopt = {"logx":True, "logy" :True, "save":True, "leg":True, "show":False}

    myp.plot([[r"$-V_{0}$", -cls.V_0], [r"$\alpha_{0}$", cls.al_0]], 'TSC_test_f0_out', x=cls.x,
             xl=r'log y', yl=r'log $\rho$[$\Omega^2/2\pi G$]', xlim=[1e-3, 1e1], ylim=[1e-7, 1e7], **figopt)

    m0 = 0.975
    rho = (m0/2/cls.x**3)**0.5/2/cls.tau**2
    # x = r/at = rOmg/a/tau = ksi/tau ==> ksi = x*tau
    myp.plot([["Shu's 0th order sol.", cls.al_0/2/cls.tau**2], ["Unpertubed state", cls.f_rho0(cls.x*cls.tau)], ["Inner expansion-wave sol.", rho]],
             'TSC_rho', x=cls.x, xl=r'log($y$)',yl=r'$\rho$ [$\Omega^2/(2\pi G)$]', ls=["-",":","--"], lw=[2,4,4], **figopt)

    myp.plot([["rho", cls.rho_eq], ["M", cls.M_eq]], "TSC_fig1", x=cls.ksi_eq,
             xl=r'r [a/$\Omega$]', yl=r'$\rho$ [$\Omega^2/2\pi G$]', xlim=[1e-2, 1e2], ylim=[1e-2, 1e6], **figopt)

    myp.plot(0.25*(cls.x**2*cls.al_0*(cls.x-cls.V_0))**2, 'TSC_fig2', x=cls.x, xl=r'log y', yl=r'$(m_0/2)^2$', xlim=[1e-3, 1e1], ylim=[1e-1, 1e2], **figopt)

    myp.plot([["alpha_0", cls.al_0], ["-alpha_Q", -cls.al_Q], ["alpha_M", cls.al_M]],
              f"TSC_fig3", x=cls.x, xlim=[1e-3, 1e1], ylim=[3e-6, 1e7], **figopt)

    myp.plot([["-V_0", -cls.V_0], ["V_M", cls.V_M]],
              f"TSC_fig4a", x=cls.x, xlim=[1e-3, 1e1], ylim=[3e-4, 1e3], **figopt)

    myp.plot([["V_Q", cls.V_Q], ["-V_Q", -cls.V_Q], ["-W_Q", -cls.W_Q]],
              f"TSC_fig4b", x=cls.x, xlim=[1e-3, 1e1], ylim=[1e-8, 1e3], **figopt)

if __name__=="__main__":
    sol = solve_TSC(tau=0.1)

