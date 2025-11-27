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
        #out[3] -> comv shape 3,n
        #out[6] -> vrot shape 3,n
        for j in range(0,n):
            xcom[n*mol+j,:] = np.float64(out[0]) #Assign the same xcom to every atom
            vcom[n*mol+j,:] = np.float64(out[3]) #Assign the same vcom to every atom
            # comv[n*mol+j,:] = np.float64(out[4][:,j]) #Assign its corresponding comv to every atom
            vrot[n*mol+j,:] = np.float64(out[6][:,j]) #Assign its corresponding vrot to every atom
    return xcom,vcom,vrot

def compute_power_per_central_atom(ids,forces,energies,x,v,group,reference_vector):

    xcm,vcm,vrot = compute_coords_solvent(x[:-len(group),:],v[:-len(group),:])
    cxcm,cvcm = compute_coords_nitro(x[-len(group):,:],v[-len(group):,:])

    firstcol = np.isin(ids,group)[:,0] #Check if the first id corresponds to a central atom
    P_indexes = np.where(firstcol,ids[:,1]-1,ids[:,0]-1) #If the first atom is central, get the second index (solvent), if not then the first index of the pair is solvent
    P_central_indexes = np.where(firstcol,ids[:,0]-1,ids[:,1]-1) #If the first atom is central, get its index, if not then take the second index (which is central because of the masking)
    P_signs = np.where(firstcol,-1,+1) #Negative if first atom of pair is central, positive otherwise

    F = -P_signs[:,np.newaxis]*forces
    Fj  = np.empty((len(group),3)) #Force to central atom j

    distances = xcm-cxcm
    distances = distances-24.8*np.rint(distances/24.8)
    front_contributions = np.sum(distances*reference_vector,axis=1) > 0. #Indexed to water atoms
    back_contributions = np.logical_not(front_contributions)

    vwater = v[:-len(group),:]  
    # mass = 2*1.008+15.9994 #g/mol
    mass = np.tile(np.array([15.9994,1.008,1.008]),int(vwater.shape[0]/3))
    kB = 8.31446261815324 #m2 s-2 kg mol-1
    
    vfront,mfront = vwater[front_contributions,:],mass[front_contributions]
    vback ,mback = vwater[back_contributions,:],mass[back_contributions]
    Tf  = (1e7/kB)*np.sum(mfront*np.sum(vfront**2,axis=1))/(2*vfront.shape[0]) #Rigid water should have 6 degrees of freedom, so a total of 2*Natom
    Tb  = (1e7/kB)*np.sum(mback*np.sum(vback**2,axis=1))/(2*vback.shape[0])
    Nf = int(vfront.shape[0]/3)
    Nb = int(vback.shape[0]/3)

    counter = 0
    for j in range(len(group)):
        mask = np.where(P_central_indexes==group[j]-1,True,False) #Mask all interactions that involve central atom j
        counter += np.sum(mask)

        Fj[j] = np.sum(F[mask],axis=0) #Sum all forces to atom j

    return Fj,Tf,Tb,Nf,Nb

def compute_temp_gradients(x,v,group,reference_vector,L_box=24.8):
    vwater = v[:-len(group),:]
    mass = np.tile(np.array([15.9994,1.008,1.008]),int(vwater.shape[0]/3))
    kB = 8.31446261815324 #m2 s-2 kg mol-1

    xcm,vcm,vrot = compute_coords_solvent(x[:-len(group),:],v[:-len(group),:])      
    cxcm,cvcm = compute_coords_nitro(x[-len(group):,:],v[-len(group):,:])

    #Temperature on the front
    distances3 = xcm-x[-5,:] #Distances between O3 and water CoM
    distances3 = distances3-L_box*np.rint(distances3/L_box)
    distances4 = xcm-x[-4,:] #Distances between O4 and water CoM
    distances4 = distances4-L_box*np.rint(distances4/L_box)
    distancesN = xcm-x[-6,:] #Distances between N and water CoM
    distancesN = distancesN-L_box*np.rint(distancesN/L_box)

    # front_contributions = np.logical_and(np.logical_or(np.sum(distances3**2,axis=1)<2.34**2,np.sum(distances4**2,axis=1)<2.34**2),np.sum(distancesN*reference_vector,axis=1)>0.) #First O-O RDF minimum
    front_contributions = np.logical_or(np.sum(distances3**2,axis=1)<5.6125**2,np.sum(distances4**2,axis=1)<5.6125**2) #Second O-O RDF maximum
    vfront,mfront = vwater[front_contributions,:],mass[front_contributions]
    Tf1  = (1e7/kB)*np.sum(mfront*np.sum(vfront**2,axis=1))/(2*vfront.shape[0]) #Rigid water should have 6 degrees of freedom, so a total of 2*Natom
    Nf1 = int(vfront.shape[0]/3)

    # front_contributions = np.logical_and(np.logical_or(np.sum(distances3**2,axis=1)<3.16**2,np.sum(distances4**2,axis=1)<3.16**2),np.sum(distancesN*reference_vector,axis=1)>0.) #Second O-O RDF minimum
    front_contributions = np.logical_or(np.sum(distances3**2,axis=1)<6.9125**2,np.sum(distances4**2,axis=1)<6.9125**2) #Second O-O RDF minimum
    vfront,mfront = vwater[front_contributions,:],mass[front_contributions]
    Tf2  = (1e7/kB)*np.sum(mfront*np.sum(vfront**2,axis=1))/(2*vfront.shape[0])
    Nf2 = int(vfront.shape[0]/3)

    front_contributions = np.logical_and(np.logical_or(np.sum(distances3**2,axis=1)<10.0**2,np.sum(distances4**2,axis=1)<10.0**2),np.sum(distancesN*reference_vector,axis=1)>0.) #Third O-O RDF minimum
    vfront,mfront = vwater[front_contributions,:],mass[front_contributions]
    Tf3  = (1e7/kB)*np.sum(mfront*np.sum(vfront**2,axis=1))/(2*vfront.shape[0])
    Nf3 = int(vfront.shape[0]/3)

    #Temperature on the back
    distances = xcm-x[-7,:] #Distances between C and water CoM
    distances = distances-L_box*np.rint(distances/L_box)

    # back_contributions = np.logical_and(np.sum(distances**2,axis=1)<3.86**2,np.sum(distances*reference_vector,axis=1)<0.) #First C-O RDF minimum
    back_contributions = np.sum(distances**2,axis=1)<5.7125**2 #First C-O RDF minimum
    vback ,mback = vwater[back_contributions,:],mass[back_contributions]
    Tb1  = (1e7/kB)*np.sum(mback*np.sum(vback**2,axis=1))/(2*vback.shape[0])
    Nb1 = int(vback.shape[0]/3)

    back_contributions = np.logical_and(np.sum(distances**2,axis=1)<10.0**2,np.sum(distances*reference_vector,axis=1)<0.) #Second C-O RDF minimum 
    vback ,mback = vwater[back_contributions,:],mass[back_contributions]
    Tb2  = (1e7/kB)*np.sum(mback*np.sum(vback**2,axis=1))/(2*vback.shape[0])
    Nb2 = int(vback.shape[0]/3)

    return Tf1,Tf2,Tf3,Tb1,Tb2,Nf1,Nf2,Nf3,Nb1,Nb2
#Excitation paths
exbin='/users/ajurado/phd/common_files/excitations/nitro_excitation.x'
exeq='/users/ajurado/phd/common_files/molecules/nitro_eq.xyz'
exff='/users/ajurado/phd/common_files/force_fields/fortran_ff/nitro_ff.dat'

Nt = 60000+1 #3 ps 
step = 10 #Every 0.5 fs
dt = 0.05 #fs

Cid = 1537
Nid = 1538
natoms_central = 7
group = list(range(Cid,Cid + natoms_central))

#Prepare file output
try:
    os.mkdir("results")
except FileExistsError:
    pass
CM =  open("results/CM.dat","w") #Central molecule info
T   = open("results/T.dat","w") #Temperatures of solvent
TN   = open("results/TN.dat","w") #Water numbers used for temperatures
FP =  open("results/FP.dat","w") #Projected forces per central atom
VP =  open("results/VP.dat","w") #Projected velocities per central atom

#Read lammps initialization file
lmp.file(infile)

types_dict = {1:"O",2:"H",3:"C",4:"N",5:"O",6:"H"}
natoms = lmp.get_natoms()
types = np.array(lmp.gather_atoms("type",0,1))
type_strings = [types_dict[i] for i in types]


lmp.command("compute ids all property/local patom1 patom2")
lmp.command("compute fdist all pair/local fx fy fz eng")
lmp.command("compute commom central_atoms momentum")
lmp.command("compute cforce central_atoms group/group solvent_atoms pair yes kspace yes")
lmp.command("run 0")

ids = gather_compute("ids",lmp,MPI.COMM_WORLD,2,np.int32)
fdist = gather_compute("fdist",lmp,MPI.COMM_WORLD,4)
commom = lmp.numpy.extract_compute("commom",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR).astype(np.float64,copy=False)
ftot = lmp.numpy.extract_compute("cforce",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR).astype(np.float64,copy=False)

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
    projvels   = np.sum(v_array[-len(group):,:]*CNvec,axis=1)

    #Compute measurements
    force,T_front,T_back,Nf,Nb = compute_power_per_central_atom(masked_ids,masked_forces,masked_energies,x_array,v_array,group,CNvec)
    Tf1,Tf2,Tf3,Tb1,Tb2,Nf1,Nf2,Nf3,Nb1,Nb2 = compute_temp_gradients(x_array,v_array,group,CNvec,L_box=24.8)

    projforce = np.sum(force*CNvec,axis=1)
    projftot = np.sum(ftot*CNvec)

    #Write to file
    CM.write(f"{0*dt:14.7E} {np.sum(projforce):21.14E} {projcommom:21.14E} {unpack(np.sum(force,axis=0))} {projftot:21.14E} {unpack(ftot)} {unpack(commom)}\n")
    T.write(f"{0*dt:14.7E} {Tf1:21.14E} {Tf2:21.14E} {Tf3:21.14E} {T_front:21.14E} {Tb1:21.14E} {Tb2:21.14E} {T_back:21.14E}\n")
    TN.write(f"{0*dt:14.7E} {Nf1} {Nf2} {Nf3} {Nf} {Nb1} {Nb2} {Nb}\n")
    FP.write(f"{0*dt:14.7E} {unpack(projforce)}\n")
    VP.write(f"{0*dt:14.7E} {unpack(projvels)}\n")

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
                # print(l)
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
    ftot = lmp.numpy.extract_compute("cforce",LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR).astype(np.float64,copy=False)

    if(i%step==0 and master):
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
        projvels   = np.sum(v_array[-len(group):,:]*CNvec,axis=1)


        #Compute measurements
        force,T_front,T_back,Nf,Nb = compute_power_per_central_atom(masked_ids,masked_forces,masked_energies,x_array,v_array,group,CNvec)
        Tf1,Tf2,Tf3,Tb1,Tb2,Nf1,Nf2,Nf3,Nb1,Nb2 = compute_temp_gradients(x_array,v_array,group,CNvec,L_box=24.8)

        projforce = np.sum(force*CNvec,axis=1)
        projftot = np.sum(ftot*CNvec)

        #Output
        if(i<int((Nt-1)*0.99)): lmp.command('log log.lammps') 

        CM.write(f"{i*dt:14.7E} {np.sum(projforce):21.14E} {projcommom:21.14E} {unpack(np.sum(force,axis=0))} {projftot:21.14E} {unpack(ftot)} {unpack(commom)}\n")
        T.write(f"{i*dt:14.7E} {Tf1:21.14E} {Tf2:21.14E} {Tf3:21.14E} {T_front:21.14E} {Tb1:21.14E} {Tb2:21.14E} {T_back:21.14E}\n")
        TN.write(f"{i*dt:14.7E} {Nf1} {Nf2} {Nf3} {Nf} {Nb1} {Nb2} {Nb}\n")
        FP.write(f"{i*dt:14.7E} {unpack(projforce)}\n")
        VP.write(f"{i*dt:14.7E} {unpack(projvels)}\n")

# print(60*" ",end='\r')
CM.close()
T.close()
TN.close()
FP.close()
VP.close()
