use molecule
use force_field
use vibration
implicit none
integer :: i,nm
character :: mol_filename*90,ff_filename*90,output_filename*90,fmtlabel*90


call get_command_argument(1,mol_filename)
call get_command_argument(2,ff_filename)


open(1,file=mol_filename)
open(2,file=ff_filename)
call init_molecule(1)
call init_forcefield(2,Natoms,xyz_mol)
close(1)
close(2)

call comp_normal_modes()


do nm=1,3*Natoms-6
      if(nm<10) then
            write(fmtlabel,"(A,I1,A)")"(A,I",1,",A)"
      elseif(nm>9 .and. nm<100) then
            write(fmtlabel,"(A,I1,A)")"(A,I",2,",A)"
      elseif(nm>99) then
            write(fmtlabel,"(A,I1,A)")"(A,I",3,",A)"
      endif

      write(output_filename,fmtlabel)"results/trajectory",nm,".xyz"
      open(3,file=output_filename)

      xyz_mol = xyz_eq
      do i=1,100

            call excite_normal_mode(0.01d0,nm,1.d0,0.d0,+1)

            call write_conf(xyz_mol,3)

      end do
      do i=1,200

            call excite_normal_mode(0.01d0,nm,1.d0,0.d0,-1)

            call write_conf(xyz_mol,3)

      end do
      close(3)
end do
end