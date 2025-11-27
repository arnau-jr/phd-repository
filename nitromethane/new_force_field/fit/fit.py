import numpy as np
import matplotlib.pyplot as plt
import fit_mod as fm
import sys

def main():
    #Initialize molecule and configurations
    if(len(sys.argv)==4):
        Natoms,xyz_eq = fm.get_molecule(sys.argv[1])
        Nconfs,energies,xyz = fm.get_confs(sys.argv[2],Natoms)
    elif(len(sys.argv)==5):
        Natoms,xyz_eq = fm.get_molecule(sys.argv[1])
        Nconfs,energies,xyz = fm.get_confs_alt(sys.argv[2],sys.argv[4],Natoms)
    else:
        print("Usage: xyz_eq confs pot_terms\n or xyz_eq confs pot_terms ener")
        quit()

    #Remove high energy configurations
    Elim = 10000.
    # Elim = 5000.
    # Emin = 2000.
    Emin = 0.
    mask = np.where(np.logical_or(energies>Elim,energies<Emin),False,True)
    Nconfs = mask.sum()
    energies = energies[mask]
    xyz = xyz[:,:,mask]
    print(f"Initialized molecule of {Natoms} atoms, got {Nconfs} configurations.")

    #Initialize force fit potential terms
    NS,NS_types,S_types,S_atoms,S_forms,\
    NB,NB_types,B_types,B_atoms,B_forms,\
    ND,ND_types,D_types,D_atoms,D_forms = fm.get_potential_terms(sys.argv[3])

    #Get equilibrium coordinates and print them
    S_eq,B_eq,D_eq = fm.get_internal_coords(xyz_eq,S_atoms,B_atoms,D_atoms)
    fm.print_internal_coordinates(S_eq,B_eq,D_eq,S_atoms,B_atoms,D_atoms)
    
    #Start building the matrix of data
    g_matrix = np.zeros([Nconfs,NS_types+NB_types+ND_types])
    for conf in range(Nconfs):
        print(f"Progress: {conf} of {Nconfs} confs",end="\r")
        S_coords,B_coords,D_coords = fm.get_internal_coords(xyz[:,:,conf],S_atoms,B_atoms,D_atoms)

        S_info = [S_coords,S_eq,S_types,S_forms]
        B_info = [B_coords,B_eq,B_types,B_forms]
        D_info = [D_coords,D_eq,D_types,D_forms]

        g_matrix[conf,:] = fm.get_g(S_info,B_info,D_info)
    print("")
    #Perform least squares fit
    sss = np.matmul(np.transpose(g_matrix),g_matrix)
    sssi = np.linalg.inv(sss)
    vtemp = np.matmul(np.transpose(g_matrix),energies)
    coef = np.matmul(sssi,vtemp)
    diff = np.matmul(g_matrix,coef)

    #Present obtained coefficients
    print("Stretches: Type, Form, Coef")
    B_coefs = []
    D_coefs = []
    for s in range(NS_types):
        i_type = s
        print(f"{s+1:2d}  {S_forms[s+1]:15s}  {coef[i_type]:15.2f}  {coef[i_type]/fm.kcalTocm:20.10f}  {coef[i_type]/fm.kjTocm:20.10f}")
    print("Bends: Type, Form, Coef")
    for b in range(NB_types):
        i_type = NS_types+b
        print(f"{b+1:2d}  {B_forms[b+1]:15s}  {coef[i_type]:15.2f}  {coef[i_type]/fm.kcalTocm:20.10f}  {coef[i_type]/fm.kjTocm:20.10f}")
        B_coefs.append(coef[i_type])
    print("Dihedrals: Type, Form, Coef")
    for d in range(ND_types):
        i_type = NS_types+NB_types+d
        print(f"{d+1:2d}  {D_forms[d+1]:15s}  {coef[i_type]:15.2f}  {coef[i_type]/fm.kcalTocm:20.10f}  {coef[i_type]/fm.kjTocm:20.10f}")
        D_coefs.append(coef[i_type])
    print("")

    #Compute relative mean error and R^2
    corr_matrix = np.corrcoef(energies, diff)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    print(f"Mean relative error: {(np.abs(diff-energies)/energies).sum()/Nconfs * 100}")
    print(f"R^2: {R_sq}")

    #Convert to morse and show params
    print("")
    print("Morse coefs:")
    print("Type          D (cm-1)      a (A-1)            D (kcal/mol)            D (kj/mol)")
    k = 1
    D = []
    a = []
    for i in range(0,NS_types,3):
        k2 = coef[i]
        k3 = coef[i+1]
        alpha = -k3/k2
        print(f"{k:2d}     {k2/alpha**2:15.2f}  {alpha:14.10f}  {k2/alpha**2/fm.kcalTocm:20.10f}  {k2/alpha**2/fm.kjTocm:20.10f}")
        D.append(k2/alpha**2)
        a.append(alpha)
        k += 1

    print("")
    print("Morse coefs (alt):")
    print("Type          D (cm-1)      a (A-1)            D (kcal/mol)            D (kj/mol)")
    k = 1
    for i in range(0,NS_types,3):
        k2 = coef[i]
        k3 = coef[i+1]
        k4 = coef[i+2]
        alpha = np.sqrt(12.*(k4/k2)/7.)
        print(f"{k:2d}     {k2/alpha**2:15.2f}  {alpha:14.10f}  {k2/alpha**2/fm.kcalTocm:20.10f}  {k2/alpha**2/fm.kjTocm:20.10f}")
        k += 1

    #Plot fitted vs actual
    plt.xlabel("Configuration index")
    plt.ylabel(r"E (cm$^{-1}$)")
    # plt.xlim([130,140])
    plt.xlim([0,Nconfs])
    # plt.ylim([0,10000.])
    plt.plot(energies,label="Ab initio data")
    plt.plot(diff,label="Fit")
    plt.legend()
    plt.savefig("comparison.eps",format="eps")
    plt.close()

    S_info = [S_coords,S_atoms,S_eq,S_types,S_forms]
    B_info = [B_coords,B_atoms,B_eq,B_types,B_forms]
    D_info = [D_coords,D_atoms,D_eq,D_types,D_forms]
    fm.print_ff_file(S_info,B_info,D_info,[D,a],B_coefs,D_coefs)

    with open('ener_comp.dat','w') as f:
        for i in range(energies.shape[0]):
            f.write(f'{energies[i]} {diff[i]}\n')

if(__name__=="__main__"):
    main()
