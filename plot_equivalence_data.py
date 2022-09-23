from mpl import *
import matplotlib.pyplot as plt
import numpy as np

# Plot degree 1,2,3 in linear scales
linestyles = ("solid", "dashed", "dotted", "dashdot")
N = 1
for degree in range(1,4):
    lwfr = np.loadtxt("error_lwfr_"+str(degree)+".txt")
    lwfr_d1 = np.loadtxt("error_lwfr_d1_"+str(degree)+".txt")
    ader = np.loadtxt("error_ader_"+str(degree)+".txt")

    plt.figure()
    plt.plot(lwfr_d1[::N,0], lwfr_d1[::N,2], label = "LW-D1", ls = linestyles[0])
    plt.plot(lwfr[::N,0], lwfr[::N,2], label = "LW-D2", ls = linestyles[1])
    plt.plot(ader[::N,0], ader[::N,2], label = "ADER", ls = linestyles[2])
    plt.xlabel("$ t $")
    plt.ylabel("$L^2$ error")
    plt.legend()
    plt.grid(True)
    plt.savefig("error"+str(degree)+".pdf")

    lwfr = np.loadtxt("error_dirichlet_lwfr_"+str(degree)+".txt")
    lwfr_d1 = np.loadtxt("error_dirichlet_lwfr_d1_"+str(degree)+".txt")
    ader = np.loadtxt("error_dirichlet_ader_"+str(degree)+".txt")

    plt.figure()
    plt.plot(lwfr_d1[::N,0], lwfr_d1[::N,2], label = "LW-D1", ls = linestyles[0])
    plt.plot(lwfr[::N,0], lwfr[::N,2], label = "LW-D2", ls = linestyles[1])
    plt.plot(ader[::N,0], ader[::N,2], label = "ADER", ls = linestyles[2])
    plt.xlabel("$ t $")
    plt.ylabel("$L^2$ error")
    plt.legend()
    plt.grid(True)
    plt.savefig("error_dirichlet"+str(degree)+".pdf")

    plt.figure()
    plt.plot(lwfr_d1[::N,0], np.sqrt(lwfr_d1[::N,3]), label = "LW-D1", ls = linestyles[0])
    plt.plot(lwfr[::N,0], np.sqrt(lwfr[::N,3]), label = "LW-D2", ls = linestyles[1])
    plt.plot(ader[::N,0], np.sqrt(ader[::N,3]), label = "ADER", ls = linestyles[2])
    plt.xlabel("$ t $")
    plt.ylabel("$L^2$ norm")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_dirichlet"+str(degree)+".pdf")
