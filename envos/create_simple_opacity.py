import numpy as np
from config import radmc_dir
import nconst as nc

# this module will be merged into radmccontoroller


class kappa:
    def __init__(self, kappa0_micron, beta):
        self.N_lam = 100
        self.kappa0_micron = kappa0_micron
        self.beta = beta
        self.lam_micron = np.logspace(np.log10(0.1), np.log10(1e5), self.N_lam)
        self.kappa_abs = self.kappa_abs_func()
        self.kappa_sca = self.kappa_sca_func()
        self.table = np.array([self.lam_micron, self.kappa_abs, self.kappa_sca]).T

    def kappa_abs_func(self):
        return self.kappa0_micron * self.lam_micron ** self.beta

    def kappa_sca_func(self):
        return np.full_like(self.lam_micron, 0.0)

    def save(self):
        # print(f"{radmc_dir}/dustkappa_kappa0{self.kappa0_micron:.0e}_beta{self.beta}.inp")
        np.savetxt(
            f"{radmc_dir}/dustkappa_kappa0{self.kappa0_micron:.0e}_beta{self.beta}.inp",
            self.table,
            header=f"2\n{self.N_lam}\n",
            comments="",
        )


if __name__ == "__main__":
    for k0 in [1e4, 1e3]:
        for b in [0, -1, -2]:
            kappa(k0, b).save()
