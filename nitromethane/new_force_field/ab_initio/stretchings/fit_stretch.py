import numpy as np
import matplotlib.pyplot as plt
import re
import os
import sys
from scipy.optimize import curve_fit

from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

reqs = {"1 2":1.5037393733,"2 3":1.2213234100,"2 4":1.2213236709,
"1 5":1.0902521836,"1 6":1.0863729466,"1 7":1.0863730719}

Dsrt = {"1 2":251.0419414,"2 3":390.3702976,"2 4":390.3702976,
"1 5":426.7713101,"1 6":426.7713101,"1 7":426.7713101}
Bsrt = {"1 2":2.005501,"2 3":2.459942,"2 4":2.459942,
"1 5":1.892486,"1 6":1.892486,"1 7":1.892486}
Rsrt = {"1 2":1.499574,"2 3":1.226747,"2 4":1.226747,
"1 5":1.09,"1 6":1.09,"1 7":1.09}

hartTocm = 219474.63
kjTohart = 1/2625.5002
kcalTocm = 349.7550112241469
kcalTokj = 4.184
kjTocm   = kcalTocm/kcalTokj

def get_stretch_coord(xyz,atom1,atom2):
    a = atom1-1
    b = atom2-1
    vdist = xyz[:,a]-xyz[:,b]
    return np.sqrt((vdist**2).sum())

def morse(x,D,beta,req):
    return D*((1.-np.exp(-beta*(x-req)))**2)


def main():
    if(len(sys.argv)!=5):
        print("Usage: confs energies atom1 atom2")
        quit()

    with open(sys.argv[1].strip(),"r") as f:
        Nconfs = int(f.readline().strip())
        Natoms = int(f.readline().strip())
        confs = np.zeros([3,Natoms,Nconfs])
        S = [0 for i in range(Natoms)]

        f.seek(0)
        f.readline()
        for conf in range(Nconfs):
            f.readline()
            f.readline()

            for i in range(Natoms):
                line = f.readline().strip().split()
                S[i] = line[0]
                confs[:,i,conf] = np.array(line[1:])

    a = int(sys.argv[3])
    b = int(sys.argv[4])
    req = reqs[f"{a} {b}"]

    dist_list = []
    E_list = []
    with open(sys.argv[2].strip(),"r") as f:
        f.readline()
        for conf in range(1,Nconfs+1):
            try:
                l = f.readline().strip().split()
                E_list.append(float(l[1])+244.9752116003)
                dist_list.append(get_stretch_coord(confs[:,:,conf-1],a,b))
            except IndexError:
                break

    dist_list = np.array(dist_list)
    E_list = np.array(E_list)

    param,cov = curve_fit(lambda x, a, b: morse(x,a,b,req),dist_list,E_list,p0=[500.,2.00])
    err = np.sqrt(np.diag(cov))

    R2 = 1. - (np.sum(E_list- morse(dist_list, *param,req))**2 / np.sum((E_list-np.mean(E_list))**2))

    dist_samples = np.linspace(dist_list.min()*0.9,dist_list.max()*1.1,1000)
    print("Optimized parameters:")
    print(f"D    = {param[0]}  {param[0]*hartTocm/kjTocm}")
    print(f"beta = {param[1]}")
    try:
        print("Errors:")
        print(f"D    = {round_to_1(err[0])}  {round_to_1(err[0]*hartTocm/kjTocm)}")
        print(f"beta = {round_to_1(err[1])}")
        print(f"R2 = {R2}")
    except OverflowError:
        print("Could not get errors")


    plt.figure()
    plt.xlabel(r"$r (\AA)$")
    plt.ylabel(r"$E (E_{H})$")
    plt.plot(dist_samples,morse(dist_samples,Dsrt[f"{a} {b}"]*kjTohart,Bsrt[f"{a} {b}"],Rsrt[f"{a} {b}"]),"g--")
    plt.plot(dist_samples,morse(dist_samples,*param,req),"b-")
    plt.plot(dist_list,E_list,"rx")
    plt.vlines(req,plt.ylim()[0],plt.ylim()[1],linestyles="dashed")
    plt.savefig("energies.eps",format="eps")
    plt.close()

if(__name__=="__main__"):
    main()
