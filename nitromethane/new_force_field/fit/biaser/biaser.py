import numpy as np
import sys


def morse(x,D,beta,req):
    return D*((1.-np.exp(-beta*(x-req)))**2)

def get_stretching_biases(path):
    with open(path.strip(),"r") as file: 
        #------------Stretches------------
        #Read number of stretches
        line = file.readline().strip().split()
        NS = int(line[0]) #Number of stretches
        S_atoms = np.zeros([NS,2],dtype=np.int32) #Pairs of atoms
        S_forms = []
        S_coefs = [] #Coefficients
        print(f"{NS} stretches")

        #Read stretch types and atoms
        for i in range(NS):
            line = file.readline().strip().split()
            S_atoms[i,:] = line[0:2]
            S_forms.append(line[2])
            S_coefs.append(line[3:])

    return NS,S_atoms,S_forms,S_coefs

def get_S_coords(xyz,S_atoms):
    NS = S_atoms.shape[0]
    S_coords = np.zeros([NS])
    for i in range(NS):
        a = S_atoms[i,0]-1
        b = S_atoms[i,1]-1
        vdist = xyz[:,a]-xyz[:,b]
        S_coords[i] = np.sqrt((vdist**2).sum())
    return S_coords



def main():
    if(len(sys.argv)!=4):
        print("Usage: confs energies bias")
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

    NS,S_atoms,S_forms,S_coefs = get_stretching_biases(sys.argv[3])

    E_list = []
    with open(sys.argv[2].strip(),"r") as f:
        f.readline()
        for conf in range(1,Nconfs+1):
            l = f.readline().strip().split()
            try:
                E_list.append(float(l[1]))
            except IndexError:
                break

    E_list = np.array(E_list)
    f = open("biasedenergies.dat","w")
    f.write(f"{E_list.shape[0]}\n")
    for conf in range(1,E_list.shape[0]+1):
        bias = 0.0
        S_coords = get_S_coords(confs[:,:,conf-1],S_atoms)
        for i in range(NS):
            bias += morse(S_coords[i],float(S_coefs[i][0]),float(S_coefs[i][1]),float(S_coefs[i][2]))
        f.write(f"{conf:4d}  {E_list[conf-1]-bias:20.10f}\n")
        f.flush()

    f.close()
if(__name__=="__main__"):
    main()