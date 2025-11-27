import numpy as np
import matplotlib.pyplot as plt
import gamess_inp as gi
import re
import os
import sys


def main():
    os.system("mkdir -p input_files")
    os.system("mkdir -p output_files")
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

    iini = int(sys.argv[2])
    ifin = int(sys.argv[3])
    for conf in range(iini,ifin+1):
        gi.energy_run(S,confs[:,:,conf-1],f"input_files/nitro_{conf:05d}.inp")
        os.system(f"rm -fv /tmp/nitro_{conf:05d}*")
        os.system(f"/home/ajurado/Desktop/gamess/rungms input_files/nitro_{conf:05d}.inp 00 8 > output_files/nitro_{conf:05d}.log")

if(__name__=="__main__"):
    main()
