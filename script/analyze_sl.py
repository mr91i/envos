import envos
import numpy as np

data = np.loadtxt("run/stream_r2424_th80.txt", unpack=True)
print(data)
data[0] /= envos.nc.year
data[1] /= envos.nc.au
data[2] /= envos.nc.au
data[3] /= 100
data[4] /= 100
data[5] /= 1e-18
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["lines.markersize"] = 1
mpl.rcParams["lines.marker"] = ","
mpl.rcParams["lines.linewidth"] = 0.2
mpl.rcParams["savefig.dpi"] = 500


def plot_dt():
    dt = data[0, 1:] - data[0, 0:-1]
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.1, 1e6)
    plt.ylim(0.1, 1e5)
    plt.xlabel("t [year]")
    plt.plot(data[0, :-1], dt, marker="o", markersize=1)
    plt.savefig("dt.png")
    plt.clf()


def plot_values():
    plt.plot(data[0], data[1], label="R [au]")
    plt.plot(data[0], data[2], label="z [au]")
    plt.plot(data[0], -data[3], label="vr [m/s]")
    plt.plot(data[0], -data[4], label="vt [m/s]")
    plt.plot(data[0], data[5], label=r"rho [10$^{-21}$ g/cm$^{3}$]")
    plt.plot(data[0], data[6], label="T [K]")
    plt.xscale("log")
    plt.yscale("log")
    # plt.xlim(1e3, 1e6)
    plt.ylim(0.1, 1e4)
    plt.xlabel("t [year]")
    plt.legend(loc="upper left")
    plt.savefig("sline.pdf")
    # plt.show()


plot_dt()
plot_values()
