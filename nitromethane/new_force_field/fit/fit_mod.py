import numpy as np
from potentials import *

hartTocm = 219474.63
kcalTocm = 349.7550112241469
kcalTokj = 4.184
kjTocm   = kcalTocm/kcalTokj

def get_molecule(path):
    with open(path.strip(),"r") as file: 
        #Read number of atoms
        Natoms = int(file.readline().strip())
        #Prepare eq coordinates array
        xyz_eq = np.zeros([3,Natoms])

        file.readline() #Throw away empty line

        for i in range(Natoms):
            #This is a compact way of reading a line of coordinates
            xyz_eq[:,i] = np.array(file.readline().strip().split()[1:])
    return Natoms,xyz_eq

def get_confs(path,Natoms):
    with open(path.strip(),"r") as file:
        #Read number of confs
        Nconfs = int(file.readline().strip())
        #Prepare arrays
        energies = np.zeros([Nconfs])
        xyz = np.zeros([3,Natoms,Nconfs])

        #For all confs
        for conf in range(Nconfs):
            #Read energies
            energies[conf] = float(file.readline().strip().split()[2])

            #Read coords
            for i in range(Natoms):
                xyz[:,i,conf] = np.array(file.readline().strip().split()[1:])
    return Nconfs,energies,xyz

def get_confs_alt(path,epath,Natoms,e_eq = -244.9752116003):
    with open(epath.strip(),"r") as file:
        #Read number of confs
        Nconfs = int(file.readline().strip())
        #Prepare arrays
        energies = np.zeros([Nconfs])

        #For all confs
        for conf in range(Nconfs):
            #Read energies
            energies[conf] = float(file.readline().strip().split()[1])
    energies = (energies-e_eq)*hartTocm
    with open(path.strip(),"r") as file:
        #Prepare arrays
        xyz = np.zeros([3,Natoms,Nconfs])

        #For all confs
        for conf in range(Nconfs):
            file.readline()
            file.readline()
            #Read coords
            for i in range(Natoms):
                xyz[:,i,conf] = np.array(file.readline().strip().split()[1:])
    return Nconfs,energies,xyz

def get_potential_terms(path):
    with open(path.strip(),"r") as file: 
        #------------Stretches------------
        #Read number of stretches
        line = file.readline().strip().split()
        NS = int(line[0]) #Number of stretches
        NS_types = int(line[1]) #Number of stretch types
        S_types = np.zeros([NS],dtype=np.int) #Types of stretches
        S_atoms = np.zeros([NS,2],dtype=np.int) #Pairs of atoms
        print(f"{NS} stretches and {NS_types} types")

        #Read stretch types and atoms
        for i in range(NS):
            line = file.readline().strip().split()
            S_types[i] = line[0]
            S_atoms[i,:] = line[1:3]
        #Read stretch forms
        file.readline() #Throw description line
        S_forms = {}
        for i in range(NS_types):
            line = file.readline().strip().split()
            if(int(line[0]) != i+1):
                print("ERROR: stretch type missing or in incorrect order")
                exit()
            S_forms[int(line[0])] = line[1]
        #------------Bends------------
        #Read number of bends
        line = file.readline().strip().split()
        NB = int(line[0]) #Number of bends
        NB_types = int(line[1]) #Number of bend types
        B_types = np.zeros([NB],dtype=np.int) #Types of bends
        B_atoms = np.zeros([NB,3],dtype=np.int) #Trios of atoms
        print(f"{NB} bends and {NB_types} types")

        #Read bend types and atoms
        for i in range(NB):
            line = file.readline().strip().split()
            B_types[i] = line[0]
            B_atoms[i,:] = line[1:4]
        #Read bend forms
        file.readline() #Throw description line
        B_forms = {}
        for i in range(NB_types):
            line = file.readline().strip().split()
            if(int(line[0]) != i+1):
                print("ERROR: bend type missing or in incorrect order")
                exit()
            B_forms[int(line[0])] = line[1]
        #------------Dihedrals------------
        #Read number of dihedrals
        line = file.readline().strip().split()
        ND = int(line[0]) #Number of dihedrals
        ND_types = int(line[1]) #Number of dihedral types
        D_types = np.zeros([ND],dtype=np.int) #Types of dihedrals
        D_atoms = np.zeros([ND,4],dtype=np.int) #Quartetos of atoms
        print(f"{ND} dihedrals and {ND_types} types")

        #Read dihedral types and atoms
        for i in range(ND):
            line = file.readline().strip().split()
            D_types[i] = line[0]
            D_atoms[i,:] = line[1:5]
        #Read dihedral forms
        file.readline() #Throw description line
        D_forms = {}
        for i in range(ND_types):
            line = file.readline().strip().split()
            if(int(line[0]) != i+1):
                print("ERROR: dihedral type missing or in incorrect order")
                exit()
            D_forms[int(line[0])] = line[1]

    return NS,NS_types,S_types,S_atoms,S_forms,\
           NB,NB_types,B_types,B_atoms,B_forms,\
           ND,ND_types,D_types,D_atoms,D_forms

def get_internal_coords(xyz,S_atoms,B_atoms,D_atoms):
    S_coords = get_S_coords(xyz, S_atoms)
    B_coords = get_B_coords(xyz, B_atoms)
    D_coords = get_D_coords(xyz, D_atoms)
    return S_coords,B_coords,D_coords

def get_S_coords(xyz,S_atoms):
    NS = S_atoms.shape[0]
    S_coords = np.zeros([NS])
    for i in range(NS):
        a = S_atoms[i,0]-1
        b = S_atoms[i,1]-1
        vdist = xyz[:,a]-xyz[:,b]
        S_coords[i] = np.sqrt((vdist**2).sum())
    return S_coords

def get_B_coords(xyz,B_atoms):
    NB = B_atoms.shape[0]
    B_coords = np.zeros([NB])
    for i in range(NB):
        a = B_atoms[i,0]-1
        b = B_atoms[i,1]-1
        c = B_atoms[i,2]-1

        u1 = xyz[:,b]-xyz[:,a]
        u2 = xyz[:,c]-xyz[:,b]

        u1 = u1/np.sqrt((u1**2).sum())
        u2 = u2/np.sqrt((u2**2).sum())

        proj = np.dot(u1,u2)
        if(proj>=1.):
            B_coords[i] = 180.
        elif(proj<=-1.):
            B_coords[i] = 0.
        else:
            B_coords[i] = 180. - (180./np.pi)*np.arccos(proj)

    return B_coords

def get_D_coords(xyz,D_atoms):
    ND = D_atoms.shape[0]
    D_coords = np.zeros([ND])
    for i in range(ND):
        a = D_atoms[i,0]-1
        b = D_atoms[i,1]-1
        c = D_atoms[i,2]-1
        d = D_atoms[i,3]-1

        u12 = xyz[:,b]-xyz[:,a]
        u23 = xyz[:,c]-xyz[:,b]
        u32 = xyz[:,b]-xyz[:,c]
        u43 = xyz[:,c]-xyz[:,d]

        u12 = u12/np.sqrt((u12**2).sum())
        u23 = u23/np.sqrt((u23**2).sum())
        u32 = u32/np.sqrt((u32**2).sum())
        u43 = u43/np.sqrt((u43**2).sum())

        u1232 = np.cross(u12,u32)
        u1232 = u1232/np.sqrt((u1232**2).sum())
        u2343 = np.cross(u23,u43)
        u2343 = u2343/np.sqrt((u2343**2).sum())
            
        proj = np.dot(u1232,u2343)
        proj2 = np.dot(u1232,u43)
        if(proj>=1.):
            D_coords[i] = 0.
        elif(proj<=-1.):
            D_coords[i] = 180.*np.sign(-proj2)
        else:
            D_coords[i] = np.sign(-proj2)*(180./np.pi)*np.arccos(proj)

    return D_coords

def print_internal_coordinates(S_coords,B_coords,D_coords,S_atoms,B_atoms,D_atoms):
    print("Stretch coordinates:")
    for i in range(S_coords.size):
        print(f"{i+1:2d}   {S_coords[i]:20.10f}   {S_atoms[i,:]}")
    print("")

    print("Bend coordinates:")
    for i in range(B_coords.size):
        print(f"{i+1:2d}   {B_coords[i]*np.pi/180.:20.10f}   {B_coords[i]:20.10f}   {B_atoms[i,:]}")
    print("")

    print("Dihedral coordinates:")
    for i in range(D_coords.size):
        print(f"{i+1:2d}   {D_coords[i]*np.pi/180.:20.10f}   {D_coords[i]:20.10f}   {D_atoms[i,:]}")
    print("")
    return

def get_g(S_info,B_info,D_info):
    S_coords,S_eq,S_types,S_forms = S_info
    B_coords,B_eq,B_types,B_forms = B_info
    D_coords,D_eq,D_types,D_forms = D_info

    NS = S_types.size
    NB = B_types.size
    ND = D_types.size

    NS_types = len(S_forms)
    NB_types = len(B_forms)
    ND_types = len(D_forms)

    g = np.zeros([NS_types+NB_types+ND_types])

    #Evaluate stretches
    for s in range(NS):
        i_type = S_types[s] - 1
        # print("IN S",s,i_type)
        pot = eval(S_forms[S_types[s]])

        g[i_type] += pot(S_coords[s],S_eq[s])

    #Evaluate bends
    for b in range(NB):
        i_type = NS_types + B_types[b] - 1
        # print("IN B",b,i_type)
        pot = eval(B_forms[B_types[b]])

        g[i_type] += pot(B_coords[b]*(np.pi/180.),B_eq[b]*(np.pi/180.))

    #Evaluate dihedrals
    for d in range(ND):
        i_type = NS_types + NB_types + D_types[d] - 1
        # print("IN D",d,i_type)
        pot = eval(D_forms[D_types[d]])

        g[i_type] += pot(D_coords[d]*(np.pi/180.),D_eq[d]*(np.pi/180.))

    return g

def print_conf(xyz):
    l = ["C","C","C","C","C","C","H","H","H","H","N","O","O","N","H","H"]
    print(xyz.shape[1])
    print("")
    for i in range(xyz.shape[1]):
        print(l[i],xyz[0,i],xyz[1,i],xyz[2,i])
    return

def print_ff_file(S_info,B_info,D_info,S_coefs,B_coefs,D_coefs,file_name="outputs/PNA_ff.dat"):
    S_coords,S_atoms,S_eq,S_types,S_forms = S_info
    B_coords,B_atoms,B_eq,B_types,B_forms = B_info
    D_coords,D_atoms,D_eq,D_types,D_forms = D_info

    S_indexes = np.sort(np.unique(S_atoms, axis=0, return_index=True)[1])
    S_unique = S_atoms[S_indexes]

    S_unique_types = S_types[S_indexes]
    S_unique_types = np.unique(S_unique_types, return_inverse=True)[1] + 1

    B_indexes = np.sort(np.unique(B_atoms, axis=0, return_index=True)[1])
    B_unique = B_atoms[B_indexes]

    B_unique_types = B_types[B_indexes]
    B_unique_types = np.unique(B_unique_types, return_inverse=True)[1] + 1

    D_indexes = np.sort(np.unique(D_atoms, axis=0, return_index=True)[1])
    D_unique = D_atoms[D_indexes]

    D_unique_types = D_types[D_indexes]
    D_unique_types = np.unique(D_unique_types, return_inverse=True)[1] + 1

    D = S_coefs[0]
    a = S_coefs[1]

    with open(file_name.strip(),"w") as f:
        f.write(f"{S_unique_types.shape[0]}\n")
        f.write("CM\n")
        for i,pair in enumerate(S_unique):
            k = S_unique_types[i]-1
            f.write(f"{pair[0]}  {pair[1]}  MORSE  {D[k]}  {a[k]}  {S_eq[S_indexes][i]}\n")

        f.write("\n")
        f.write(f"{B_unique_types.shape[0]}\n")
        f.write("CM\n")
        for i,pair in enumerate(B_unique):
            k = B_unique_types[i]-1
            f.write(f"{pair[0]}  {pair[1]}  {pair[2]}  {B_forms[k+1].upper()}  {B_coefs[k]}  {B_eq[B_indexes][i]}\n")

        f.write("\n")
        f.write(f"{D_unique_types.shape[0]}\n")
        f.write("CM\n")
        for i,pair in enumerate(D_unique):
            k = D_unique_types[i]-1
            f.write(f"{pair[0]}  {pair[1]}  {pair[2]}  {pair[3]}  {D_forms[k+1].upper()}  {D_coefs[k]}  {D_eq[D_indexes][i]}\n")



    return
