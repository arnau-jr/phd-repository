#!/usr/bin/python3


import sys
import os
import numpy as np
import ctypes


argv = sys.argv
if len(argv) != 4:
  print("Syntax: controller.py in.lammps NM EXENERGY")
  sys.exit()

infile = sys.argv[1]


from mpi4py import MPI
me = MPI.COMM_WORLD.Get_rank()
master = me == 0
nprocs = MPI.COMM_WORLD.Get_size()


from lammps import lammps,LMP_STYLE_LOCAL,LMP_STYLE_GLOBAL,LMP_TYPE_VECTOR,LMP_TYPE_ARRAY
lmp = lammps(name='mpi',cmdargs=["-pk","omp","2","-sf", "omp","-screen","none"])


#Excitation paths
exbin='/users/ajurado/phd/detailed_nitro_spce/excitation.x'
exeq='/users/ajurado/phd/common_files/molecules/nitro_eq.xyz'
exff='/users/ajurado/phd/common_files/force_fields/fortran_ff/nitro_ff.dat'

dt = 0.05 #fs
Tpreexcitation = 1.0 #ps
StepsPreExcitation = (Tpreexcitation*1000.0)/dt
Tpostexcitation = 4.0 #ps
StepsPostExcitation = (Tpostexcitation*1000.0)/dt


#Read lammps initialization file
lmp.file(infile)

Cid = 1537
natoms_central = 7
group = list(range(Cid,Cid + natoms_central))


lmp.command(f"run {int(StepsPreExcitation)}")
#Excitation
lmp.command('write_dump central_atoms custom temp.dat element x y z vx vy vz modify sort id element O H C N O H')
if(master): os.system(f'{exbin} {exeq} {exff} {sys.argv[2]} {sys.argv[3]} temp.dat new.dat')
MPI.COMM_WORLD.barrier()
with open('new.dat','r') as f:
    for a in group:
        l = f.readline()
        l = l.strip().split()
        lmp.command(f'set atom {a} vx {float(l[3])} vy {float(l[4])} vz {float(l[5])}')
    
lmp.command(f"run {int(StepsPostExcitation)}")
