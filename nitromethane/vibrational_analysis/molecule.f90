! molecule module 
! September 2021
! Author: Arnau Jurado Romero

module molecule
      implicit none 

      !Constants and parameters
      real*8 :: atomic_mass(18)

      !Atom information
      integer               :: Natoms
      character,allocatable :: S_mol(:)*2 !Atomic symbols of molecule
      integer,allocatable   :: Z_mol(:) !Atomic numbers of molecule
      real*8,allocatable    :: M_mol(:) !Atomic masses of molecule
      
      !Coordinates
      real*8,allocatable :: xyz_mol(:,:) !Instantaneous coordinates
      real*8,allocatable :: xyz_cm(:,:) !Instantaneous coordinates in CoM frame
      real*8             :: cm_pos(3)
      real*8,allocatable :: xyz_eq(:,:) !Equilibrium coordinates in CoM frame
      real*8             :: eq_cm_pos(3)
      real*8,allocatable :: xyz_eckart(:,:) !Equilibirum coordinates in the Eckart frame
      real*8             :: U_eckart(3,3) !Rotation matrix that transforms from the eckart frame to the original equilibrium frame

      !Velocities
      real*8,allocatable :: vel_mol(:,:) !Instantaneous velocities
      real*8,allocatable :: vel_cm(:,:) !Instantaneous velocities in CoM frame
      real*8             :: cm_vel(3),omega_mol(3)
      real*8,allocatable :: vel_vib(:,:) !Vibrational velocities


      !Format labels for outputting normal mode energies

      character :: normal_format_label*90
      contains

      subroutine parse_atomic_symbol()
            implicit none
            integer :: i
            do i=1,Natoms
                  if(S_mol(i)=="H") then
                        Z_mol(i) = 1
                  elseif(S_mol(i)=="C") then
                        Z_mol(i) = 6
                  elseif(S_mol(i)=="N") then
                        Z_mol(i) = 7
                  elseif(S_mol(i)=="O") then
                        Z_mol(i) = 8
                  elseif(S_mol(i)=="Ar" .or.S_mol(i)=="AR" .or. S_mol(i)=="ar") then
                        Z_mol(i) = 18
                  else
                        write(*,*)"Parsing: atom not recognized"
                  endif
                  M_mol(i) = atomic_mass(Z_mol(i))
            enddo
      end subroutine parse_atomic_symbol

      subroutine update_cm_coords()
            implicit none
            real*8              :: total_mass
            integer             :: i

            total_mass = 0.d0
            cm_pos = 0.d0
            xyz_cm = 0.d0

            do i=1,Natoms
                  total_mass = total_mass + M_mol(i)
                  cm_pos(:) = cm_pos(:) + M_mol(i)*xyz_mol(:,i)
            end do
            cm_pos = cm_pos/total_mass
            do i=1,Natoms
                  xyz_cm(:,i) = xyz_mol(:,i) - cm_pos(:)
            end do
      end subroutine update_cm_coords

      subroutine update_cm_coords_with_pbc(L_box)
            implicit none
            real*8,intent(in) :: L_box(3)
            real*8            :: total_mass
            real*8            :: distv(3)
            real*8            :: xyz_aux(3,Natoms)
            integer           :: i

            total_mass = 0.d0
            cm_pos = 0.d0
            xyz_cm = 0.d0

            xyz_aux(:,1) = xyz_mol(:,1)
            do i=2,Natoms
                  distv = xyz_mol(:,1) - xyz_mol(:,i)
                  distv = distv-L_box*nint(distv/L_box)
                  xyz_aux(:,i) = xyz_mol(:,1) - distv
            end do

            do i=1,Natoms
                  total_mass = total_mass + M_mol(i)
                  cm_pos(:) = cm_pos(:) + M_mol(i)*xyz_aux(:,i)
            end do
            cm_pos = cm_pos/total_mass
            do i=1,Natoms
                  xyz_cm(:,i) = xyz_aux(:,i) - cm_pos(:)
            end do
      end subroutine update_cm_coords_with_pbc


      subroutine init_molecule(unit,unit_mass)
            implicit none
            integer,intent(in) :: unit
            integer,optional   :: unit_mass
            integer            :: i,dummy

            !If mass file present override default masses
            if(present(unit_mass)) then
                  do i=1,18 !This has to be modified when adding new elements
                        read(unit_mass,*)dummy,atomic_mass(i)
                  end do
            else !Else use default masses
                  !Default masses right now are for nitromethane's SRT force field
                  atomic_mass = (/1.00782522d0,0.d0,0.d0,0.d0,0.d0,12.01d0,14.01d0,15.99491502d0,0.d0,0.d0,&
                                                0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,39.95d0/)
            end if

            !Read number of atoms from file
            read(unit,*)Natoms
            read(unit,*)
            !Allocate quantities
            allocate(S_mol(Natoms),M_mol(Natoms),Z_mol(Natoms))
            allocate(xyz_mol(3,Natoms),xyz_cm(3,Natoms),xyz_eq(3,Natoms),xyz_eckart(3,Natoms))
            allocate(vel_mol(3,Natoms),vel_cm(3,Natoms),vel_vib(3,Natoms))
            vel_mol = 0.d0
            vel_cm = 0.d0
            vel_vib = 0.d0

            !Read the atoms themselves
            do i = 1,Natoms
                  read(unit,*)S_mol(i),xyz_mol(:,i)
            enddo

            !Get masses from symbols
            call parse_atomic_symbol()

            !Obtain equilibrium CoM coordinates
            call update_cm_coords()
            xyz_eq = xyz_cm
            eq_cm_pos = cm_pos

            write(normal_format_label,"(A,I2,A)")"(E20.13,2X,",3*Natoms,"(E20.13,2X))"
      end subroutine init_molecule


      subroutine write_conf(r,port)
            implicit none
            integer,intent(in) :: port
            real*8,intent(in)  :: r(3,Natoms)
            integer            :: i

            write(port,"(I5)")Natoms
            write(port,*)""
            do i=1,Natoms
                  write(port,"(A,2X,E20.13,2X,E20.13,2X,E20.13)")S_mol(i),r(:,i)
            enddo
      end subroutine write_conf


end module molecule

