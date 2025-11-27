import numpy as np
import matplotlib.pyplot as plt
import gamess_inp as gi
import re
import os
import sys

def get_energy_from_gamess_output(file):
    E = [float(line.strip().split()[4]) for line in open(file.strip(), 'r') if re.search("FINAL", line)]
    if(len(E)>1):  print("WARNING: more than one 'FINAL' line encountered in file ",file)
    if(len(E)==0): 
        print("WARNING: no 'FINAL' line encountered in file ",file)
        return None
    return E[-1]

def main():

    energies = []


    for filename in sorted(os.listdir("output_files")):

        full_file = "output_files/"+filename
        E = get_energy_from_gamess_output(full_file)
        if(E != None):
            energies.append(E)
            print(f"Progress: {len(energies)} confs",end="\r")
        else:
            print(f"Some energies missing and/or still running",end="\r")
            break
    print("")
    Nconfs = len(energies)        
    print(f"Finished, got {Nconfs} energies")
    f = open("energies.dat","w")
    f.write(f"{Nconfs}\n")
    for conf in range(1,Nconfs+1):
        f.write(f"{conf:4d}  {energies[conf-1]:20.10f}\n")
        f.flush()

    plt.xlabel("Configuration Index")
    plt.ylabel(r"E (cm$^{-1}$)")
    plt.xlim([0,Nconfs])
    plt.ylim([0,10000.])
    # plt.xticks(np.arange(0,Nconfs+1,20))
    plt.xticks(np.linspace(0,Nconfs,11,dtype=np.int32))
    plt.plot((np.array(energies)+244.9752116003)*219474.6)
    plt.savefig("energies.eps",format="eps")
    plt.close()

    f.close()


if(__name__=="__main__"):
    main()
