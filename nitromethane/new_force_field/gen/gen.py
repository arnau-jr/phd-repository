import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.spatial.transform import Rotation

def get_scan_terms(path):
    count = 0
    with open(path.strip(),"r") as file: 
        #------------Stretches------------
        #Read number of stretches
        line = file.readline().strip().split()
        NS = int(line[0]) #Number of stretches
        S_atoms = np.zeros([NS,2],dtype=np.int32) #Pairs of atoms
        S_data = np.zeros([NS,3]) #Scanning specifications, f_min, f_max, df
        S_clusters = [] #List of cluster lists
        print(f"{NS} stretches")

        for i in range(NS):
            line = file.readline().strip().split()
            S_atoms[i,:] = line[0:2]
            S_data[i,:] = line[2:5]

            n_cluster = int(line[5])
            cluster = []
            for iclust in range(n_cluster):
                cluster.append(int(line[6+iclust]))
            S_clusters.append(cluster)

            print(S_atoms[i,:],S_data[i,:],[count,count+int((S_data[i,1]-S_data[i,0])/S_data[i,2])-1]) 
            count += int((S_data[i,1]-S_data[i,0])/S_data[i,2])

        #------------Bends------------
        #Read number of bends
        line = file.readline().strip().split()
        NB = int(line[0]) #Number of bends
        B_atoms = np.zeros([NB,3],dtype=np.int32) #Trios of atoms
        B_data = np.zeros([NB,3]) #Scanning specifications, delta_a_neg, delta_a_pos, df
        B_clusters = []
        print(f"{NB} bends")

        for i in range(NB):
            line = file.readline().strip().split()
            B_atoms[i,:] = line[0:3]
            B_data[i,:] = line[3:6]

            n_cluster1 = int(line[6])
            cluster1 = []
            for iclust in range(n_cluster1):
                cluster1.append(int(line[7+iclust]))
            n_cluster2 = int(line[7+n_cluster1])
            cluster2 =[]
            for iclust in range(n_cluster2):
                cluster2.append(int(line[7+n_cluster1+1+iclust]))
            B_clusters.append([cluster1,cluster2])

            print(B_atoms[i,:],B_data[i,:],[count,count+int((B_data[i,1]-B_data[i,0])/B_data[i,2])-1])
            count += int((B_data[i,1]-B_data[i,0])/B_data[i,2])

        #------------Dihedrals------------
        #Read number of dihedrals
        line = file.readline().strip().split()
        ND = int(line[0]) #Number of bends
        D_atoms = np.zeros([ND,4],dtype=np.int32) #Quartets of atoms
        D_data = np.zeros([ND,3]) #Scanning specifications, delta_a_neg, delta_a_pos, df
        print(f"{ND} dihedrals")

        for i in range(ND):
            line = file.readline().strip().split()
            D_atoms[i,:] = line[0:4]
            D_data[i,:] = line[4:7]
            print(D_atoms[i,:],D_data[i,:],[count,count+int((D_data[i,1]-D_data[i,0])/D_data[i,2])-1]) 
            count += int((D_data[i,1]-D_data[i,0])/D_data[i,2])

    return NS,S_atoms,S_data,S_clusters,\
           NB,B_atoms,B_data,B_clusters,\
           ND,D_atoms,D_data

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

def stretch_bond(a,b,xyz,f,cluster=[]):
    #Save relative atom positions to move groups together
    if(len(cluster)>0):
        rel_vecs = np.zeros([3,len(cluster)])
        for i,atom in enumerate(cluster):
            rel_vecs[:,i] = xyz[:,b-1] - xyz[:,atom-1] 

    #f is the stretching factor of the bond e.g. f=2 is increases the bond to be double length
    #while f=1 is unchanged. f<1 shortens the bond and f>1 makes it longer.
    d_vec = xyz[:,b-1] - xyz[:,a-1] 

    xyz_mod = np.copy(xyz)
    xyz_mod[:,b-1] = xyz[:,a-1] + f*d_vec

    #Restore relative positions
    if(len(cluster)>0):
        for i,atom in enumerate(cluster):
            rel_vec = rel_vecs[:,i]
            xyz_mod[:,atom-1] = xyz_mod[:,b-1] - rel_vec
    return xyz_mod

def bend_angle(a,b,c,xyz,f,cluster1=[],cluster2=[]):
    #Save relative atom positions to move groups together
    if(len(cluster1)>0):
        rel_vecs1 = np.zeros([3,len(cluster1)])
        for i,atom in enumerate(cluster1):
            rel_vecs1[:,i] = xyz[:,a-1] - xyz[:,atom-1]
    if(len(cluster2)>0):
        rel_vecs2 = np.zeros([3,len(cluster2)])
        for i,atom in enumerate(cluster2):
            rel_vecs2[:,i] = xyz[:,c-1] - xyz[:,atom-1] 

    #f is the amount of degrees that the angle closes (negative) or opens (positive)
    u1 = xyz[:,a-1]-xyz[:,b-1]
    u2 = xyz[:,c-1]-xyz[:,b-1]
    u1mod = np.sqrt((u1**2).sum())
    u2mod = np.sqrt((u2**2).sum())
    u1 = u1/u1mod
    u2 = u2/u2mod

    u_rot = np.cross(u1,u2)
    u_rot = u_rot/np.sqrt((u_rot**2).sum())

    f *= (np.pi/180.)
    u_rotpos =  f/2.*u_rot
    u_rotneg = -f/2.*u_rot
    rot_pos = Rotation.from_rotvec(u_rotpos)
    rot_neg = Rotation.from_rotvec(u_rotneg)

    u1_rot = u1mod*rot_neg.apply(u1)
    u2_rot = u2mod*rot_pos.apply(u2)

    xyz_mod = np.copy(xyz)
    xyz_mod[:,a-1] = xyz[:,b-1] + u1_rot
    xyz_mod[:,c-1] = xyz[:,b-1] + u2_rot

    #Restore relative positions
    if(len(cluster1)>0):
        for i,atom in enumerate(cluster1):
            rel_vec = rel_vecs1[:,i]
            xyz_mod[:,atom-1] = xyz_mod[:,a-1] - rel_vec
    if(len(cluster2)>0):
        for i,atom in enumerate(cluster2):
            rel_vec = rel_vecs2[:,i]
            xyz_mod[:,atom-1] = xyz_mod[:,c-1] - rel_vec
    return xyz_mod

def torsion_dihedral(a,b,c,d,xyz,f):
    #f is the amount of degrees that the dihedral increases (positive) or decreases (negative)
    u1 = xyz[:,d-1]-xyz[:,c-1]
    u1mod = np.sqrt((u1**2).sum())
    u1 = u1/u1mod

    u_rot = xyz[:,c-1]-xyz[:,b-1]
    u_rot = u_rot/np.sqrt((u_rot**2).sum())

    f *= (np.pi/180.)
    u_rotpos = f*u_rot
    rot_pos= Rotation.from_rotvec(u_rotpos)

    u1_rot = u1mod*rot_pos.apply(u1)

    xyz_mod = np.copy(xyz)
    xyz_mod[:,d-1] = xyz[:,c-1] + u1_rot
    return xyz_mod


def print_conf(xyz,file=None):
    # l = ["C","C","C","C","C","C","H","H","H","H","N","O","O","N","H","H"] #PNA
    # l = ["C","C","C","C","C","C","O","O","N","N","C","H","H","H","C","H","H","H","H","H","H","H"] #DMPNA
    l = ["C","N","O","O","H","H","H"] #Nitromethane
    if(file is None):
        print(xyz.shape[1])
        print("")
        for i in range(xyz.shape[1]):
            print(l[i],xyz[0,i],xyz[1,i],xyz[2,i])
    else:
        file.write(f"{xyz.shape[1]}\n")
        file.write("\n")
        for i in range(xyz.shape[1]):
            file.write(f"{l[i]}  {xyz[0,i]} {xyz[1,i]} {xyz[2,i]}\n")
    return

def main():
    if(len(sys.argv)==1):
        print("Usage: mol scanterms (outfile)")
        quit()
    elif(len(sys.argv)>3):
        file = open(sys.argv[3].strip(),"w")
    else:
        file = open("confs.xyz","w")

    #Initialize molecule
    Natoms,xyz_eq = get_molecule(sys.argv[1])
    NS,S_atoms,S_data,S_clusters,\
    NB,B_atoms,B_data,B_clusters,\
    ND,D_atoms,D_data  = get_scan_terms(sys.argv[2])

    increment = 0.1
    total_confs = 0

    for S in range(NS):
        a = S_atoms[S,0]
        b = S_atoms[S,1]

        fmin = S_data[S,0]
        fmax = S_data[S,1]
        increment = S_data[S,2]

        cluster = S_clusters[S]

        Nsteps = int(np.rint((1.-fmin)/increment))
        for i in range(Nsteps):
            # f = 1. - (i+1)*increment
            f = 1.-Nsteps*increment + i*increment
            xyz = stretch_bond(a,b,xyz_eq,f,cluster)
            print_conf(xyz,file)
            total_confs += 1

        Nsteps = int(np.rint((fmax-1.)/increment))
        for i in range(Nsteps):
            f = 1. + (i+1)*increment
            xyz = stretch_bond(a,b,xyz_eq,f,cluster)
            print_conf(xyz,file)
            total_confs += 1
    print(f"Generated {total_confs}[{0},{total_confs}] stretching configurations")
    oldconfs = total_confs + 1
    for B in range(NB):
        a = B_atoms[B,0]
        b = B_atoms[B,1]
        c = B_atoms[B,2]

        delta_a_neg = B_data[B,0]
        delta_a_pos = B_data[B,1]
        increment = B_data[B,2]

        cluster1 = B_clusters[B][0]
        cluster2 = B_clusters[B][1]

        Nsteps = int(np.abs(delta_a_neg)/increment)
        for i in range(Nsteps):
            # f = -(i+1)*increment
            f = -(Nsteps)*increment + i*increment
            xyz = bend_angle(a,b,c,xyz_eq,f,cluster1,cluster2)
            print_conf(xyz,file)
            total_confs += 1

        Nsteps = int(delta_a_pos/increment)
        for i in range(Nsteps):
            f = (i+1)*increment
            xyz = bend_angle(a,b,c,xyz_eq,f,cluster1,cluster2)
            print_conf(xyz,file)
            total_confs += 1
    print(f"Generated {total_confs-oldconfs}[{oldconfs},{total_confs}] bending configurations")
    oldconfs = total_confs + 1
    for D in range(ND):
        a = D_atoms[D,0]
        b = D_atoms[D,1]
        c = D_atoms[D,2]
        d = D_atoms[D,3]

        delta_a_neg = D_data[D,0]
        delta_a_pos = D_data[D,1]
        increment = D_data[D,2]

        Nsteps = int(np.abs(delta_a_neg)/increment)
        for i in range(Nsteps):
            # f = -(i+1)*increment
            f = -(Nsteps)*increment + i*increment
            xyz = torsion_dihedral(a,b,c,d,xyz_eq,f)
            print_conf(xyz,file)
            total_confs += 1

        Nsteps = int(delta_a_pos/increment)
        for i in range(Nsteps):
            f = (i+1)*increment
            xyz = torsion_dihedral(a,b,c,d,xyz_eq,f)
            print_conf(xyz,file)
            total_confs += 1
    print(f"Generated {total_confs-oldconfs}[{oldconfs},{total_confs}] torsion configurations")

    file.close()
    print(f"Generated {total_confs} configurations")
if(__name__=="__main__"):
    main()
