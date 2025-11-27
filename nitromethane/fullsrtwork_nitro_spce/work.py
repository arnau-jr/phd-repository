#!/usr/bin/python3


import sys
import os
import numpy as np
import ctypes

sys.path.insert(1, '/users/ajurado/phd/codes/aux_vib')
import auxvibmod as av

argv = sys.argv
if len(argv) != 4:
  print("Syntax: simple.py in.lammps NM EXENERGY")
  sys.exit()

infile = sys.argv[1]

from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
master = me == 0
nprocs = MPI.COMM_WORLD.Get_size()

from lammps import lammps,LMP_STYLE_LOCAL,LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,LMP_TYPE_ARRAY
lmp = lammps(name='mpi',cmdargs=["-pk","omp","2","-sf", "omp","-screen","none"])




def write_conf(lmp,x_array,f,strings=None):
    natoms = lmp.get_natoms()
    if(strings==None): strings = natoms*["Ar"]
    f.write(f'{natoms} \n')
    f.write("\n")
    for i in range(natoms):
        f.write(f"{strings[i]} {x_array[0,i]} {x_array[1,i]} {x_array[2,i]}\n")
    return 0

def unpack(a,format=''):
    return " ".join([f"{x:21.14E}" for x in a])

def unpack_int(a,format=''):
    return " ".join([f"{x:d}" for x in a])

def create_group_mask(ids,group):
    premask = np.isin(ids,group)
    return np.logical_xor(premask[:,0],premask[:,1])

def gather_compute(computeid,lmp,comm,vec_size,datatype=np.float64):
    array = lmp.numpy.extract_compute(computeid,LMP_STYLE_LOCAL,LMP_TYPE_ARRAY).astype(datatype,copy=False)
    farray = array.flatten()
    array_sizes = np.array(comm.gather(farray.shape[0],root=0))
    if(master):
        recvbf = np.empty(array_sizes.sum(),dtype=datatype)
    else:
        recvbf = None
    comm.Gatherv(farray,(recvbf,array_sizes),root=0)
    if(master):
        return recvbf.reshape((int(array_sizes.sum()/vec_size),vec_size))
    else:
        return None

def compute_coords_nitro(xnitro,vnitro):
    n = 7
    #Eq coord and mass hard coded to spce water
    xyz_eq = np.zeros([3,n],order='F',dtype=np.float64)
    xyz_eq[:,0] = -0.1256938291E-03, 0.3834994319E-08, 0.6781222655E-09
    xyz_eq[:,1] =  0.1499448306E+01,-0.2609289635E-07, 0.1055414928E-09
    xyz_eq[:,2] =  0.2057376327E+01, 0.1828650844E-07, 0.1092531247E+01
    xyz_eq[:,3] =  0.2057376326E+01, 0.2911719356E-08,-0.1092531247E+01
    xyz_eq[:,4] = -0.3289321224E+00, 0.1039223913E+01, 0.7974184136E-07
    xyz_eq[:,5] = -0.3289321540E+00,-0.5196118723E+00,-0.8999943382E+00
    xyz_eq[:,6] = -0.3289321533E+00,-0.5196120090E+00, 0.8999942609E+00
    mass = np.array([12.01,14.01,15.99491502,15.99491502,1.00782522,1.00782522,1.00782522],order='F',dtype=np.float64)
    
    xcom = np.zeros(3,dtype=np.float64)
    vcom = np.zeros(3,dtype=np.float64)
    # vrot = np.zeros([vsol.shape[0],3],dtype=np.float64)

    xyz = np.zeros([3,n],order='F',dtype=np.float64)
    vel = np.zeros([3,n],order='F',dtype=np.float64)

    for j in range(0,n):
        for k in range(3):
            xyz[k,j] = np.float64(xnitro[j,k])
            vel[k,j] = np.float64(vnitro[j,k])

    out = av.auxvib.vibrational_analysis(xyz_eq,xyz,vel,mass,24.8,n=n)
    #out[0] -> xcom shape 3
    #out[3] -> vcom shape 3
    #out[6] -> vrot shape 3,n
    for j in range(0,n):
        xcom = np.float64(out[0]) #Assign xcom
        vcom = np.float64(out[3]) #Assign vcom
        # vrot = np.float64(out[3]) #Assign vcom
        # vrot[n*mol+j,:] = np.float64(out[6][:,j]) #Assign its corresponding vrot to every atom
    return xcom,vcom

def compute_coords_solvent(xsol,vsol):
    n = 3
    if(xsol.shape[0]%n != 0): print(f'Number of solvent atoms not divisible by {n}')
    nmol = int(xsol.shape[0]/n)
    #Eq coord and mass hard coded to spce water
    xyz_eq = np.zeros([3,n],order='F',dtype=np.float64)
    xyz_eq[:,0] = 0.0000000000000E+00,-0.5275758213025E-01,-0.3730608705389E-01
    xyz_eq[:,1] = 0.0000000000000E+00,-0.5275758213025E-01, 0.9626939129461E+00
    xyz_eq[:,2] = 0.0000000000000E+00, 0.8900585606015E+00,-0.3706193346221E+00
    mass = np.array([15.9994,1.008,1.008],order='F',dtype=np.float64)
    
    xcom = np.zeros([vsol.shape[0],3],dtype=np.float64)
    vcom = np.zeros([vsol.shape[0],3],dtype=np.float64)
    vrot = np.zeros([vsol.shape[0],3],dtype=np.float64)

    xyz = np.zeros([3,n],order='F',dtype=np.float64)
    vel = np.zeros([3,n],order='F',dtype=np.float64)

    for mol in range(0,nmol):
        for j in range(0,n):
            xyz[0,j] = np.float64(xsol[n*mol+j,0])
            xyz[1,j] = np.float64(xsol[n*mol+j,1])
            xyz[2,j] = np.float64(xsol[n*mol+j,2])
            vel[0,j] = np.float64(vsol[n*mol+j,0])
            vel[1,j] = np.float64(vsol[n*mol+j,1])
            vel[2,j] = np.float64(vsol[n*mol+j,2])
            
        # xyz = xsol[n*mol:n*mol+n,:].reshape([3,n],order='F')
        # vel = vsol[n*mol:n*mol+n,:].reshape([3,n],order='F')

        out = av.auxvib.vibrational_analysis(xyz_eq,xyz,vel,mass,24.8,n=n)
        #out[0] -> xcom shape 3
        #out[3] -> vcom shape 3
        #out[6] -> vrot shape 3,n
        for j in range(0,n):
            xcom[n*mol+j,:] = np.float64(out[0]) #Assign the same xcom to every atom
            vcom[n*mol+j,:] = np.float64(out[3]) #Assign the same vcom to every atom
            vrot[n*mol+j,:] = np.float64(out[6][:,j]) #Assign its corresponding vrot to every atom
    return xcom,vcom,vrot

def compute_power_per_central_atom(ids,forces,energies,x,v,group,reference_vector,Npro=48,dpro=0.5):
    P = np.zeros((ids.shape[0])) #Power associated to each individual interaction

    xcm,vcm,vrot = compute_coords_solvent(x[:-len(group),:],v[:-len(group),:])
    cxcm,cvcm = compute_coords_nitro(x[-len(group):,:],v[-len(group):,:])

    firstcol = np.isin(ids,group)[:,0] #Check if the first id corresponds to a central atom
    P_indexes = np.where(firstcol,ids[:,1]-1,ids[:,0]-1) #If the first atom is central, get the second index (solvent), if not then the first index of the pair is solvent
    P_central_indexes = np.where(firstcol,ids[:,0]-1,ids[:,1]-1) #If the first atom is central, get its index, if not then take the second index (which is central because of the masking)
    P_signs = np.where(firstcol,-1,+1) #Negative if first atom of pair is central, positive otherwise

    F = -P_signs[:,np.newaxis]*forces
    Fj  = np.empty((len(group),3)) #Force to central atom j

    P  = P_signs*np.sum(forces*v[P_indexes,:],axis=1) #Pk = fj->i * vi with k = interaction involving central atom j and solvent atom i
    Pproj = P_signs*np.sum(forces*reference_vector,axis=1)*np.sum(v[P_indexes,:]*reference_vector,axis=1) #Same as P but projected onto reference_vector
    PT = P_signs*np.sum(forces*vcm[P_indexes,:],axis=1) #Same but for vcm, translational power
    PR = P_signs*np.sum(forces*vrot[P_indexes,:],axis=1) #Same but for vrot, rotational power

    Pj  = np.empty((len(group))) #Power to solvent created by central atom j
    Pjt = np.empty((len(group))) #"" translational
    Pjr = np.empty((len(group))) #"" rotational

    #Front/back division
    Pjf = np.empty((len(group))) #"" front
    Pjb = np.empty((len(group))) #"" back

    distances = xcm[P_indexes,:]-cxcm[:]
    distances = distances-24.8*np.rint(distances/24.8)
    front_contributions = np.sum(distances*reference_vector,axis=1) > 0.
    back_contributions = np.logical_not(front_contributions)

    #Mapping division
    proj = np.sum(distances*reference_vector,axis=1)
    dist = np.sqrt(np.sum(distances**2,axis=1))
    phi = np.arccos(proj/np.sqrt(np.sum(reference_vector**2))/dist)
    rho = dist*np.sin(phi)
    z   = dist*np.cos(phi)

    Pmap = np.zeros((Npro,Npro,len(group)))
    Pprojmap = np.zeros((Npro,Npro,len(group)))
    irho = np.floor(rho/dpro + Npro/2.).astype(int)
    iz   = np.floor(z/dpro + Npro/2.).astype(int)
    counter = 0
    for j in range(P.shape[0]):
        if(irho[j]<Npro/2): print(irho[j],rho[j])
        if(irho[j]<Npro-1 and iz[j]<Npro-1):
            counter += 1
            iatom = P_central_indexes[j]-(group[0]-1)
            Pmap[irho[j],iz[j],iatom] += P[j]
            Pprojmap[irho[j],iz[j],iatom] += Pproj[j]

    Uf = np.sum(energies[front_contributions])
    Ub = np.sum(energies[back_contributions])

    counter = 0
    for j in range(len(group)):
        mask = np.where(P_central_indexes==group[j]-1,True,False) #Mask all interactions that involve central atom j
        counter += np.sum(mask)

        Fj[j] = np.sum(F[mask],axis=0) #Sum all forces to atom j

        Pj[j] = np.sum(P[mask]) #Sum all interactions that involve central atom j
        Pjt[j] = np.sum(PT[mask]) # "" trans
        Pjr[j] = np.sum(PR[mask]) # "" rot
        Pjf[j] = np.sum(P[np.logical_and(front_contributions,mask)]) #Sum all interactions that involve central atom j and are front
        Pjb[j] = np.sum(P[np.logical_and(back_contributions,mask)]) #Sum all interactions that involve central atom j and are back
    if(counter!=P.shape[0]): print('Some interaction unaccounted for in mask')
    return Pj,Pjt,Pjr,Pjf,Pjb,Fj,Uf,Ub,Pmap,Pprojmap

def ComputePowerPerCentralAtomToCentral(ids,forces,x,v,group,reference_vector):
    P = np.zeros((ids.shape[0])) #Power associated to each individual interaction

    xcm,vcm,vrot = compute_coords_solvent(x[:-len(group),:],v[:-len(group),:])
    cxcm,cvcm = compute_coords_nitro(x[-len(group):,:],v[-len(group):,:])

    firstcol = np.isin(ids,group)[:,0] #Check if the first id corresponds to a central atom
    P_indexes = np.where(firstcol,ids[:,1]-1,ids[:,0]-1) #If the first atom is central, get the second index (solvent), if not then the first index of the pair is solvent
    P_central_indexes = np.where(firstcol,ids[:,0]-1,ids[:,1]-1) #If the first atom is central, get its index, if not then take the second index (which is central because of the masking)
    P_signs = np.where(firstcol,+1,-1) #Positive if first atom of pair is central, negative otherwise

    P  = P_signs*np.sum(forces*v[P_central_indexes,:],axis=1) #Pk = fj->i * vi with k = interaction involving central atom j and solvent atom i
    Pproj = P_signs*np.sum(forces*reference_vector,axis=1)*np.sum(v[P_central_indexes,:]*reference_vector,axis=1) #Same as P but projected onto reference_vector
    PT = P_signs*np.sum(forces*cvcm[:],axis=1) #Same but for vcm, translational power
    # PR = P_signs*np.sum(forces*cvrot[:],axis=1) #Same but for vrot, rotational power

    Pj  = np.empty((len(group))) #Power to solvent created by central atom j
    Pjt = np.empty((len(group))) #"" translational
    Pjr = np.empty((len(group))) #"" rotational

    #Front/back division
    Pjf = np.empty((len(group))) #"" front
    Pjb = np.empty((len(group))) #"" back

    distances = xcm[P_indexes,:]-cxcm[:]
    distances = distances-24.8*np.rint(distances/24.8)
    front_contributions = np.sum(distances*reference_vector,axis=1) > 0.
    back_contributions = np.logical_not(front_contributions)

    #Mapping division
    proj = np.sum(distances*reference_vector,axis=1)
    dist = np.sqrt(np.sum(distances**2,axis=1))
    phi = np.arccos(proj/np.sqrt(np.sum(reference_vector**2))/dist)
    rho = dist*np.sin(phi)
    z   = dist*np.cos(phi)

    Pmap = np.zeros((Npro,Npro,len(group)))
    Pprojmap = np.zeros((Npro,Npro,len(group)))
    irho = np.floor(rho/dpro + Npro/2.).astype(int)
    iz   = np.floor(z/dpro + Npro/2.).astype(int)
    counter = 0
    for j in range(P.shape[0]):
        if(irho[j]<Npro/2): print(irho[j],rho[j])
        if(irho[j]<Npro-1 and iz[j]<Npro-1):
            counter += 1
            iatom = P_central_indexes[j]-(group[0]-1)
            Pmap[irho[j],iz[j],iatom] += P[j]
            Pprojmap[irho[j],iz[j],iatom] += Pproj[j]

    counter = 0
    for j in range(len(group)):
        mask = np.where(P_central_indexes==group[j]-1,True,False) #Mask all interactions that involve central atom j
        counter += np.sum(mask)

        Pj[j] = np.sum(P[mask]) #Sum all interactions that involve central atom j
        Pjt[j] = np.sum(PT[mask]) # "" trans
        # Pjr[j] = np.sum(PR[mask]) # "" rot
        Pjf[j] = np.sum(P[np.logical_and(front_contributions,mask)]) #Sum all interactions that involve central atom j and are front
        Pjb[j] = np.sum(P[np.logical_and(back_contributions,mask)]) #Sum all interactions that involve central atom j and are back
    if(counter!=P.shape[0]): print('Some interaction unaccounted for in mask')
    return Pj,Pjt,Pjr,Pjf,Pjb,Pmap,Pprojmap


#Excitation paths
exbin='/users/ajurado/phd/common_files/excitations/nitro_excitation.x'
exeq='/users/ajurado/phd/common_files/molecules/nitro_eq.xyz'
exff='/users/ajurado/phd/common_files/force_fields/fortran_ff/nitro_ff.dat'

Nt = 200000+1 #10 ps
step = 100 #Every 5 fs
dt = 0.05 #fs

Cid = 1537
Nid = 1538
natoms_central = 7
group = list(range(Cid,Cid + natoms_central))

dpro = 0.5 #Angstroms
Npro = 48 #12 total Angstroms

P_to_solvent = np.empty((Nt+1,natoms_central))
P_front = np.empty((Nt+1,natoms_central))
P_back = np.empty((Nt+1,natoms_central))
P_trans = np.empty((Nt+1,natoms_central))
P_rot = np.empty((Nt+1,natoms_central))

P_map      = np.empty((2,Npro,Npro,natoms_central)) #Only store previous and current power
Pproj_map  = np.empty((2,Npro,Npro,natoms_central))
CP_map     = np.empty((2,Npro,Npro,natoms_central))
CPproj_map = np.empty((2,Npro,Npro,natoms_central))

work_to_solvent = np.zeros(natoms_central)
work_front = np.zeros(natoms_central)
work_back = np.zeros(natoms_central)
work_trans = np.zeros(natoms_central)
work_rot = np.zeros(natoms_central)

work_map = np.zeros((Npro,Npro,natoms_central))
workproj_map = np.zeros((Npro,Npro,natoms_central))
Cwork_map = np.zeros((Npro,Npro,natoms_central))
Cworkproj_map = np.zeros((Npro,Npro,natoms_central))

CP_to_solvent = np.empty((Nt+1,natoms_central))
CP_front = np.empty((Nt+1,natoms_central))
CP_back = np.empty((Nt+1,natoms_central))
CP_trans = np.empty((Nt+1,natoms_central))
CP_rot = np.empty((Nt+1,natoms_central))

Cwork_to_solvent = np.zeros(natoms_central)
Cwork_front = np.zeros(natoms_central)
Cwork_back = np.zeros(natoms_central)
Cwork_trans = np.zeros(natoms_central)
Cwork_rot = np.zeros(natoms_central)

#Prepare file output
try:
    os.mkdir("results")
except FileExistsError:
    pass
CM      =  open("results/CM.dat","w") #Central molecule info
FP      =  open("results/FP.dat","w") #Projected forces per central atom
VP      =  open("results/VP.dat","w") #Projected velocities per central atom
P       =   open("results/P.dat","w") #Total power per central atom
W       =   open("results/W.dat","w") #Total work per central atom
WM      =   open("results/WM.dat","w") #Total work map
WprojM  =   open("results/WprojM.dat","w") #Total projected work map
CWM     =   open("results/CWM.dat","w") #Total work to central map
CWprojM =   open("results/CWprojM.dat","w") #Total projected work to central map
PF      =  open("results/PF.dat","w") #Total front power per central atom
WF      =  open("results/WF.dat","w") #Total front work per central atom
PB      =  open("results/PB.dat","w") #Total back power per central atom
WB      =  open("results/WB.dat","w") #Total back work per central atom
PT      =  open("results/PT.dat","w") #Total translational power per central atom
WT      =  open("results/WT.dat","w") #Total translational work per central atom
PR      =  open("results/PR.dat","w") #Total rotational power per central atom
WR      =  open("results/WR.dat","w") #Total rotational work per central atom
CP      =  open("results/CP.dat","w") #Total power per central atom to central
CW      =  open("results/CW.dat","w") #Total work per central atom to central
CPF     = open("results/CPF.dat","w") #Total front power per central atom to central
CWF     = open("results/CWF.dat","w") #Total front work per central atom to central
CPB     = open("results/CPB.dat","w") #Total back power per central atom to central
CWB     = open("results/CWB.dat","w") #Total back work per central atom to central
CPT     = open("results/CPT.dat","w") #Total translational power per central atom to central
CWT     = open("results/CWT.dat","w") #Total translational work per central atom to central
CPR     = open("results/CPR.dat","w") #Total rotational power per central atom to central
CWR     = open("results/CWR.dat","w") #Total rotational work per central atom to central
U       = open("results/U.dat","w") #Potential energies
SC      =  open("results/SC.dat","w") #Sanity checks


#Read lammps initialization file
lmp.file(infile)

types_dict = {1:"O",2:"H",3:"C",4:"N",5:"O",6:"H"}
natoms = lmp.get_natoms()
types = np.array(lmp.gather_atoms("type",0,1))
type_strings = [types_dict[i] for i in types]


lmp.command("compute ids all property/local patom1 patom2")
lmp.command("compute fdist all pair/local fx fy fz eng")
lmp.command("compute commom central_atoms momentum")
lmp.command("run 0")

ids = gather_compute("ids",lmp,MPI.COMM_WORLD,2,np.int32)
fdist = gather_compute("fdist",lmp,MPI.COMM_WORLD,4)
commom = lmp.numpy.extract_compute("commom",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR).astype(np.float64,copy=False)

x = lmp.gather_atoms("x",1,3)
x_array = np.array(x,dtype=np.float64).reshape([natoms,3])
v = lmp.gather_atoms("v",1,3)
v_array = np.array(v,dtype=np.float64).reshape([natoms,3])

if(master):
    #Separate forces from distances (pair/local fix)
    forces = fdist[:,:3]
    energies = fdist[:,3]

    #Get the indexes, ids, forces and distances of the central atoms' interactions
    mask = create_group_mask(ids,group)
    masked_ids = ids[mask,:]
    masked_forces = forces[mask,:]
    masked_energies = energies[mask]

    #Define vector for front/back separation
    CNvec = x_array[Nid-1,:]-x_array[Cid-1,:]
    CNvec = CNvec/np.sqrt(np.sum(CNvec**2))

    projcommom = np.sum(commom*CNvec)
    # projvels   = v_array[-len(group):,:]*np.tile(CNvec,len(group))
    projvels   = np.sum(v_array[-len(group):,:]*CNvec,axis=1)

    #Compute measurements
    power,power_trans,power_rot,power_front,power_back,force,U_front,U_back,power_map,powerproj_map = compute_power_per_central_atom(masked_ids,masked_forces,masked_energies,x_array,v_array,group,CNvec,Npro,dpro)
    cpower,cpower_trans,cpower_rot,cpower_front,cpower_back,Cpower_map,Cpowerproj_map = ComputePowerPerCentralAtomToCentral(masked_ids,masked_forces,x_array,v_array,group,CNvec)

    projforce = np.sum(force*CNvec,axis=1)

    P_to_solvent[0,:] = power
    P_trans[0,:] = power_trans
    P_rot[0,:] = power_rot
    P_front[0,:] = power_front
    P_back[0,:] = power_back

    P_map[0,:,:,:] = power_map
    Pproj_map[0,:,:,:] = powerproj_map
    CP_map[0,:,:,:] = Cpower_map
    CPproj_map[0,:,:,:] = Cpowerproj_map
    #Both current and old power are the same for the first step
    P_map[1,:,:,:] = power_map
    Pproj_map[1,:,:,:] = powerproj_map
    CP_map[1,:,:,:] = Cpower_map
    CPproj_map[1,:,:,:] = Cpowerproj_map

    work_to_solvent = P_to_solvent[0,:]*dt/2.
    work_trans = P_trans[0,:]*dt/2.
    work_rot = P_rot[0,:]*dt/2.
    work_front = P_front[0,:]*dt/2.
    work_back = P_back[0,:]*dt/2.

    work_map = P_map[0,:,:,:]*dt/2.
    workproj_map = Pproj_map[0,:,:,:]*dt/2.
    Cwork_map = CP_map[0,:,:,:]*dt/2.
    Cworkproj_map = CPproj_map[0,:,:,:]*dt/2.


    CP_to_solvent[0,:] = cpower
    CP_trans[0,:] = cpower_trans
    CP_rot[0,:] = cpower_rot
    CP_front[0,:] = cpower_front
    CP_back[0,:] = cpower_back

    Cwork_to_solvent = CP_to_solvent[0,:]*dt/2.
    Cwork_trans = CP_trans[0,:]*dt/2.
    Cwork_rot = CP_rot[0,:]*dt/2.
    Cwork_front = CP_front[0,:]*dt/2.
    Cwork_back = CP_back[0,:]*dt/2.
    

    #Write to file
    CM.write(f"{0*dt:14.7E} {np.sum(projforce):21.14E} {projcommom:21.14E} {np.sum(projvels):21.14E}\n")
    FP.write(f"{0*dt:14.7E} {unpack(projforce)}\n")
    VP.write(f"{0*dt:14.7E} {unpack(projvels)}\n")
    P.write(f"{0*dt:14.7E} {unpack(P_to_solvent[0,:])}\n")
    W.write(f"{0*dt:14.7E} {unpack(P_to_solvent[0,:]*dt/2.)}\n")
    PT.write(f"{0*dt:14.7E} {unpack(P_trans[0,:])}\n")
    WT.write(f"{0*dt:14.7E} {unpack(P_trans[0,:]*dt/2.)}\n")
    PR.write(f"{0*dt:14.7E} {unpack(P_rot[0,:])}\n")
    WR.write(f"{0*dt:14.7E} {unpack(P_rot[0,:]*dt/2.)}\n")
    PF.write(f"{0*dt:14.7E} {unpack(P_front[0,:])}\n")
    WF.write(f"{0*dt:14.7E} {unpack(P_front[0,:]*dt/2.)}\n")
    PB.write(f"{0*dt:14.7E} {unpack(P_back[0,:])}\n")
    WB.write(f"{0*dt:14.7E} {unpack(P_back[0,:]*dt/2.)}\n")
    CP.write(f"{0*dt:14.7E} {unpack(CP_to_solvent[0,:])}\n")
    CW.write(f"{0*dt:14.7E} {unpack(CP_to_solvent[0,:]*dt/2.)}\n")
    CPT.write(f"{0*dt:14.7E} {unpack(CP_trans[0,:])}\n")
    CWT.write(f"{0*dt:14.7E} {unpack(CP_trans[0,:]*dt/2.)}\n")
    CPR.write(f"{0*dt:14.7E} {unpack(CP_rot[0,:])}\n")
    CWR.write(f"{0*dt:14.7E} {unpack(CP_rot[0,:]*dt/2.)}\n")
    CPF.write(f"{0*dt:14.7E} {unpack(CP_front[0,:])}\n")
    CWF.write(f"{0*dt:14.7E} {unpack(CP_front[0,:]*dt/2.)}\n")
    CPB.write(f"{0*dt:14.7E} {unpack(CP_back[0,:])}\n")
    CWB.write(f"{0*dt:14.7E} {unpack(CP_back[0,:]*dt/2.)}\n")
    U.write(f"{0*dt:14.7E} {U_front:21.14E} {U_back:21.14E} {U_front+U_back:21.14E}\n")
    SC.write(f"{0*dt:14.7E}"\
    f" {np.sum(P_trans[0,:]+P_rot[0,:])-np.sum(P_to_solvent[0,:]):21.14E}"\
    f" {np.sum(P_front[0,:]+P_back[0,:])-np.sum(P_to_solvent[0,:]):21.14E}"\
    f" {np.sum(P_trans[0,:]*dt/2.+P_rot[0,:]*dt/2.)-np.sum(P_to_solvent*dt/2.):21.14E}"\
    f" {np.sum(P_front[0,:]*dt/2.+P_back[0,:]*dt/2.)-np.sum(P_to_solvent*dt/2.):21.14E}"\
    )

for i in range(1,Nt):
    if(i==1):
        lmp.command("run 1")
    elif(i==20000+1): #At 1.0 ps
        #Excitation
        lmp.command('write_dump central_atoms custom temp.dat element x y z vx vy vz modify sort id element O H C N O H')
        if(master): os.system(f'{exbin} {exeq} {exff} {sys.argv[2]} {sys.argv[3]} temp.dat new.dat')
        MPI.COMM_WORLD.barrier()
        with open('new.dat','r') as f:
            for a in group:
                l = f.readline()
                l = l.strip().split()
                print(l)
                lmp.command(f'set atom {a} vx {float(l[3])} vy {float(l[4])} vz {float(l[5])}')
        lmp.command("run 0")
        lmp.command("run 1")
    else:
        lmp.command("run 1 pre no post no")
    x = lmp.gather_atoms("x",1,3)
    x_array = np.array(x,dtype=np.float64).reshape([natoms,3])
    v = lmp.gather_atoms("v",1,3)
    v_array = np.array(v,dtype=np.float64).reshape([natoms,3])

    ids = gather_compute("ids",lmp,MPI.COMM_WORLD,2,np.int32)
    fdist = gather_compute("fdist",lmp,MPI.COMM_WORLD,4)
    commom = lmp.numpy.extract_compute("commom",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR).astype(np.float64,copy=False)

    if(master):
        #Separate forces from distances (pair/local fix)
        forces = fdist[:,:3]
        energies = fdist[:,3]

        #Get the indexes, ids, forces and distances of the central atoms' interactions
        mask = create_group_mask(ids,group)
        masked_ids = ids[mask,:]
        masked_forces = forces[mask,:]
        masked_energies = energies[mask]

        #Define vector for front/back separation
        CNvec = x_array[Nid-1,:]-x_array[Cid-1,:]
        CNvec = CNvec/np.sqrt(np.sum(CNvec**2))

        projcommom = np.sum(commom*CNvec)
        # projvels   = v_array[-len(group):,:]*np.tile(CNvec,len(group))
        projvels   = np.sum(v_array[-len(group):,:]*CNvec,axis=1)

        #Store the previous power in the other 'slot'
        P_map[0,:,:,:]      = P_map[1,:,:,:]     
        Pproj_map[0,:,:,:]  = Pproj_map[1,:,:,:] 
        CP_map[0,:,:,:]     = CP_map[1,:,:,:]    
        CPproj_map[0,:,:,:] = CPproj_map[1,:,:,:]

        #Compute measurements
        power,power_trans,power_rot,power_front,power_back,force,U_front,U_back,power_map,powerproj_map = compute_power_per_central_atom(masked_ids,masked_forces,masked_energies,x_array,v_array,group,CNvec,Npro,dpro)
        cpower,cpower_trans,cpower_rot,cpower_front,cpower_back,Cpower_map,Cpowerproj_map = ComputePowerPerCentralAtomToCentral(masked_ids,masked_forces,x_array,v_array,group,CNvec)

        projforce = np.sum(force*CNvec,axis=1)

        P_to_solvent[i,:] = power
        P_trans[i,:] = power_trans
        P_rot[i,:] = power_rot
        P_front[i,:] = power_front
        P_back[i,:] = power_back

        P_map[1,:,:,:]      = power_map
        Pproj_map[1,:,:,:]  = powerproj_map
        CP_map[1,:,:,:]     = Cpower_map
        CPproj_map[1,:,:,:] = Cpowerproj_map

        CP_to_solvent[i,:] = cpower
        CP_trans[i,:] = cpower_trans
        CP_rot[i,:] = cpower_rot
        CP_front[i,:] = cpower_front
        CP_back[i,:] = cpower_back
        if(i>1):
            work_to_solvent += P_to_solvent[i-1,:]*dt/2. + P_to_solvent[i,:]*dt/2.
            work_trans += P_trans[i-1,:]*dt/2. + P_trans[i,:]*dt/2.
            work_rot += P_rot[i-1,:]*dt/2. + P_rot[i,:]*dt/2.
            work_front += P_front[i-1,:]*dt/2. + P_front[i,:]*dt/2.
            work_back += P_back[i-1,:]*dt/2. + P_back[i,:]*dt/2.

            work_map += P_map[0,:,:,:]*dt/2. + P_map[1,:,:,:]*dt/2.
            workproj_map += Pproj_map[0,:,:,:]*dt/2. + Pproj_map[1,:,:,:]*dt/2.
            Cwork_map += CP_map[0,:,:,:]*dt/2. + CP_map[1,:,:,:]*dt/2.
            Cworkproj_map += CPproj_map[0,:,:,:]*dt/2. + CPproj_map[1,:,:,:]*dt/2.

            Cwork_to_solvent += CP_to_solvent[i-1,:]*dt/2. + CP_to_solvent[i,:]*dt/2.
            Cwork_trans += CP_trans[i-1,:]*dt/2. + CP_trans[i,:]*dt/2.
            Cwork_rot += CP_rot[i-1,:]*dt/2. + CP_rot[i,:]*dt/2.
            Cwork_front += CP_front[i-1,:]*dt/2. + CP_front[i,:]*dt/2.
            Cwork_back += CP_back[i-1,:]*dt/2. + CP_back[i,:]*dt/2.
        else:
            work_to_solvent += (P_to_solvent[1,:])*dt/2.
            work_trans += (P_trans[1,:])*dt/2.
            work_rot += (P_rot[1,:])*dt/2.
            work_front += (P_front[1,:])*dt/2.
            work_back += (P_back[1,:])*dt/2.

            work_map += (P_map[1,:,:,:])*dt/2.
            workproj_map += (Pproj_map[1,:,:,:])*dt/2.
            Cwork_map += (CP_map[1,:,:,:])*dt/2.
            Cworkproj_map += (CPproj_map[1,:,:,:])*dt/2.

            Cwork_to_solvent += (CP_to_solvent[1,:])*dt/2.
            Cwork_trans += (CP_trans[1,:])*dt/2.
            Cwork_rot += (CP_rot[1,:])*dt/2.
            Cwork_front += (CP_front[1,:])*dt/2.
            Cwork_back += (CP_back[1,:])*dt/2.

    if(i%step==0 and master):
        if(i<int((Nt-1)*0.99)): lmp.command('log log.lammps') 
        # print(f' {i/Nt*100.:5.1f}% ['+ int(i/(Nt-1)*50)*u'\u2586' +(50-int(i/(Nt-1)*50))*' '+']',end='\r')

        CM.write(f"{i*dt:14.7E} {np.sum(projforce):21.14E} {projcommom:21.14E} {np.sum(projvels):21.14E}\n")
        FP.write(f"{i*dt:14.7E} {unpack(projforce)}\n")
        VP.write(f"{i*dt:14.7E} {unpack(projvels)}\n")
        P.write(f"{i*dt:14.7E} {unpack(P_to_solvent[i,:])}\n")
        W.write(f"{i*dt:14.7E} {unpack(work_to_solvent)}\n")
        PT.write(f"{i*dt:14.7E} {unpack(P_trans[i,:])}\n")
        WT.write(f"{i*dt:14.7E} {unpack(work_trans)}\n")
        PR.write(f"{i*dt:14.7E} {unpack(P_rot[i,:])}\n")
        WR.write(f"{i*dt:14.7E} {unpack(work_rot)}\n")
        PF.write(f"{i*dt:14.7E} {unpack(P_front[i,:])}\n")
        WF.write(f"{i*dt:14.7E} {unpack(work_front)}\n")
        PB.write(f"{i*dt:14.7E} {unpack(P_back[i,:])}\n")
        WB.write(f"{i*dt:14.7E} {unpack(work_back)}\n")
        CP.write(f"{i*dt:14.7E} {unpack(CP_to_solvent[i,:])}\n")
        CW.write(f"{i*dt:14.7E} {unpack(Cwork_to_solvent)}\n")
        CPT.write(f"{i*dt:14.7E} {unpack(CP_trans[i,:])}\n")
        CWT.write(f"{i*dt:14.7E} {unpack(Cwork_trans)}\n")
        CPR.write(f"{i*dt:14.7E} {unpack(CP_rot[i,:])}\n")
        CWR.write(f"{i*dt:14.7E} {unpack(Cwork_rot)}\n")
        CPF.write(f"{i*dt:14.7E} {unpack(CP_front[i,:])}\n")
        CWF.write(f"{i*dt:14.7E} {unpack(Cwork_front)}\n")
        CPB.write(f"{i*dt:14.7E} {unpack(CP_back[i,:])}\n")
        CWB.write(f"{i*dt:14.7E} {unpack(Cwork_back)}\n")
        U.write(f"{i*dt:14.7E} {U_front:21.14E} {U_back:21.14E} {U_front+U_back:21.14E}\n")
        SC.write(f"{i*dt:14.7E}"\
        f" {np.sum(P_trans[i,:]+P_rot[i,:])-np.sum(P_to_solvent[i,:]):21.14E}"\
        f" {np.sum(P_front[i,:]+P_back[i,:])-np.sum(P_to_solvent[i,:]):21.14E}"\
        f" {np.sum(work_trans+work_rot)-np.sum(work_to_solvent):21.14E}"\
        f" {np.sum(work_front+work_back)-np.sum(work_to_solvent):21.14E}"\
        )

for irho in range(Npro):
    for iz in range(Npro):
        WM.write(f"{(irho-Npro/2.)*dpro:14.7E} {(iz-Npro/2.)*dpro:14.7E} {unpack(work_map[irho,iz,:])}\n")
        WprojM.write(f"{(irho-Npro/2.)*dpro:14.7E} {(iz-Npro/2.)*dpro:14.7E} {unpack(workproj_map[irho,iz,:])}\n")
        CWM.write(f"{(irho-Npro/2.)*dpro:14.7E} {(iz-Npro/2.)*dpro:14.7E} {unpack(Cwork_map[irho,iz,:])}\n")
        CWprojM.write(f"{(irho-Npro/2.)*dpro:14.7E} {(iz-Npro/2.)*dpro:14.7E} {unpack(Cworkproj_map[irho,iz,:])}\n")

print(60*" ",end='\r')
CM.close()
FP.close()
VP.close()
P.close()
W.close()
WM.close()
WprojM.close()
CWM.close()
CWprojM.close()
PT.close()
WT.close()
PR.close()
WR.close()
PF.close()
WF.close()
PB.close()
WB.close()
CP.close()
CW.close()
CPT.close()
CWT.close()
CPR.close()
CWR.close()
CPF.close()
CWF.close()
CPB.close()
CWB.close()
U.close()
SC.close()

