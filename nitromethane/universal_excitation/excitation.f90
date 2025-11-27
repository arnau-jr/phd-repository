use molecule
use force_field
use vibration
implicit none
integer   :: i,j,nm_mode,time_index
character :: mol_filename*90,ff_filename*90,nmarg*90,exctenergyarg*90
character :: input_filename*90,output_filename*90
character :: filename*90,dummy*90

real*8,allocatable :: xyz_old(:,:),vel_old(:,:),xyz_cm_og(:,:)
real*8             :: E0,K0,Ecm0,Erot0,Evib0,Ecor0
real*8             :: E,K,Ecm,Erot,Evib,Ecor
real*8,allocatable :: KNM(:),UNM(:),KNM0(:),UNM0(:)

real*8             :: exctenergy
real*8             :: rot_mat(3,3),cm_pos_og(3),cm_vel_og(3)

real*8             :: L_box(3)

call get_command_argument(1,mol_filename)
call get_command_argument(2,ff_filename)
call get_command_argument(3,nmarg)
call get_command_argument(4,exctenergyarg)
call get_command_argument(5,input_filename)
call get_command_argument(6,output_filename)
read(nmarg,*)nm_mode
read(exctenergyarg,*)exctenergy

open(1,file=mol_filename)
open(2,file=ff_filename)
call init_molecule(1)
call init_forcefield(2,Natoms,xyz_mol)
close(1)
close(2)

call comp_normal_modes()

allocate(xyz_old(3,Natoms),vel_old(3,Natoms),xyz_cm_og(3,Natoms))
allocate(KNM(3*Natoms),UNM(3*Natoms),KNM0(3*Natoms),UNM0(3*Natoms))

!Read instanteous configuration from file
open(1,file=input_filename)
!Read dummy lines
read(1,*)
read(1,*)time_index
call srand(984561+time_index)
do j=1,3
      read(1,*)
end do
read(1,*)dummy,L_box(1)
read(1,*)dummy,L_box(2)
read(1,*)dummy,L_box(3)
L_box = L_box*2.d0
read(1,*)


do j=1,Natoms
      read(1,*)dummy,xyz_mol(:,j),vel_mol(:,j)
end do
vel_mol=vel_mol*100.d0
close(1)

!Stablish eckart frame, the resulting rotation matrix brings us from the instanteous
!frame to the equilibrium frame. We also get cm_pos which is the original CoM position
call get_eckart_frame_with_pbc(L_box)
call update_cm_vel()

rot_mat=U_eckart
cm_pos_og=cm_pos
xyz_cm_og=xyz_cm
cm_vel_og=cm_vel


!Compute everything for instantaneous state
E = comp_energy(Natoms,xyz_mol)
K = comp_kinetic_energy(vel_mol)

call get_eckart_frame_with_pbc(L_box)

call kinetic_energy_analysis(Ecm,Erot,Evib,Ecor)
call comp_normal_energy(KNM,UNM)

E0    = E
K0    = K
Ecm0  = Ecm
Erot0 = Erot
Evib0 = Evib
Ecor0 = Ecor

KNM0  = KNM
UNM0  = UNM

print*,"Timestep",time_index
print*,"Initial energies (input conf):"
print*,"U   ",E
print*,"K   ",K
print*,"Ecm ",Ecm
print*,"Erot",Erot
print*,"Evib",Evib
print*,"Ecor",Ecor
print*,"------------------------------"
print*,"NM | Initial NM energies: K, U"
do i=1,3*Natoms
      print*,i,"|",KNM(i),UNM(i)
end do

! Stablish the in CoM frame coordinates as the new coordinates
! all excitations will be done as if we were in the equilibrium frame (but with displaced coordinates)
do j=1,Natoms
      xyz_mol(:,j) = matmul(rot_mat,xyz_cm(:,j))
      vel_mol(:,j) = matmul(rot_mat,vel_cm(:,j))
end do

!We need to recompute Evib (and we need the eckart frame for that)
!so the second excitation is accurate
call get_eckart_frame_with_pbc(L_box)
call kinetic_energy_analysis(Ecm,Erot,Evib,Ecor)
!Excite
if(nm_mode==0) then !Global excitation is taken as mode 0
      !This line has to be reworked from the module side, right now it's specific for nitromethane
      call excite_normal_modes_micro(exctenergy,15,(/1,2,3,4,5,6,7,8,9,10,11,12,13,14,15/))
else
      call excite_normal_mode(exctenergy,nm_mode,0.d0,1.d0)
end if

!Correct coordinates by rotating them and displacing them, so they agree with the original system
do j=1,Natoms
      xyz_mol(:,j) = matmul(transpose(rot_mat),xyz_mol(:,j)) + cm_pos_og(:)
      xyz_cm(:,j) = matmul(transpose(rot_mat),xyz_cm(:,j))
      vel_mol(:,j) = matmul(transpose(rot_mat),vel_mol(:,j)) + cm_vel_og(:)
end do

!Compute everything
E = comp_energy(Natoms,xyz_mol)

 K = comp_kinetic_energy(vel_mol)

call get_eckart_frame_with_pbc(L_box)
call kinetic_energy_analysis(Ecm,Erot,Evib,Ecor)
call comp_normal_energy(KNM,UNM)

!Correct for pbcs
do j=1,Natoms
      xyz_mol(:,j) = xyz_mol(:,j) - L_box*nint(xyz_mol(:,j)/L_box)
end do

open(3,file=output_filename)
do j=1,Natoms
      write(3,"(6(E20.10,2X))")xyz_mol(:,j),vel_mol(:,j)/100.d0 !A/fs
end do


print*,"Final energies | Difference:"
print*,"U   ",E,"|",E-E0
print*,"K   ",K,"|",K-K0
print*,"Ecm ",Ecm,"|",Ecm-Ecm0
print*,"Erot",Erot,"|",Erot-Erot0
print*,"Evib",Evib,"|",Evib-Evib0
print*,"Ecor",Ecor,"|",Ecor-Ecor0
print*,"------------------------------"
print*,"NM | Final NM energies: K, U, K difference"
do i=1,3*Natoms
      print*,i,"|",KNM(i),UNM(i),KNM(i)-KNM0(i)
end do
end
