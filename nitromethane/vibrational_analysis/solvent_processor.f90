! solvent_processor module 
! October 2021
! Author: Arnau Jurado Romero

module solvent_processor
      use OMP_LIB
      implicit none 

      !Constants and parameters
      real*8,parameter :: solvent_atomic_mass(18) = (/1.00782522d0,0.d0,0.d0,0.d0,0.d0,12.01d0,14.01d0,15.99491502d0,0.d0,0.d0,&
                                                0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,0.d0,39.95d0/)!TBD
      real*8,parameter :: solvent_kcal_to_kj = 4.184d0
      real*8,parameter :: electrostatic_constant = 1389.3502 !kJ/mol Angs

      !Atom information
      integer               :: Nmols,Natoms_per_mol,solvent_Natoms
      character,allocatable :: solvent_S_mol(:)*2 !Atomic symbols of atoms
      integer,allocatable   :: solvent_Z_mol(:) !Atomic numbers of atoms
      real*8,allocatable    :: solvent_M_mol(:) !Atomic masses of atoms
      
      !Coordinates (coord,atom,molecule) or (coord,molecule) or (coord,coord,molecule)
      real*8,allocatable :: solvent_xyz_mol(:,:,:) !Instantaneous coordinates
      real*8,allocatable :: solvent_xyz_cm(:,:,:) !Instantaneous coordinates in CoM frame
      real*8,allocatable :: solvent_cm_pos(:,:)
      real*8,allocatable :: solvent_xyz_eq(:,:) !Equilibrium coordinates in CoM frame (coord,atom)
      real*8             :: solvent_eq_cm_pos(3)
      real*8,allocatable :: solvent_xyz_eckart(:,:,:) !Equilibirum coordinates in the Eckart frame
      real*8,allocatable :: solvent_U_eckart(:,:,:) !Rotation matrix that transforms from the eckart frame to the original equilibrium frame
      real*8             :: L_box(3) !Size of the solvent box

      !Velocities (same as coordinates)
      real*8,allocatable :: solvent_vel_mol(:,:,:) !Instantaneous velocities
      real*8,allocatable :: solvent_vel_cm(:,:,:) !Instantaneous velocities in CoM frame
      real*8,allocatable :: solvent_cm_vel(:,:),solvent_omega_mol(:,:)
      real*8,allocatable :: solvent_vel_vib(:,:,:) !Vibrational velocities

      !Forces acting on the solvent
      integer            :: Natoms_central
      character          :: pair_pot_type*90
      real*8,allocatable :: pair_coefs(:,:,:) !(coef,solvent types,central types)
      real*8             :: pair_cut,long_cut
      real*8,allocatable :: q_solvent(:),q_central(:)
      real*8,allocatable :: solvent_F(:,:,:,:),solvent_U(:,:,:) 
      !(i,j,k,l) component i that atom j of the central molecule does on atom k of molecule l
      !(j,k,l) interaction energy of atom j of the central molecule and atom k of molecule l
      real*8,allocatable :: solvent_PT(:),solvent_PR(:)

      real*8,allocatable :: solvent_Fcentral(:,:,:,:)
      contains

      subroutine solvent_parse_atomic_symbol()
            implicit none
            integer :: i
            do i=1,Natoms_per_mol
                  if(solvent_S_mol(i)=="H") then
                        solvent_Z_mol(i) = 1
                  elseif(solvent_S_mol(i)=="C") then
                        solvent_Z_mol(i) = 6
                  elseif(solvent_S_mol(i)=="N") then
                        solvent_Z_mol(i) = 7
                  elseif(solvent_S_mol(i)=="O") then
                        solvent_Z_mol(i) = 8
                  elseif(Solvent_S_mol(i)=="Ar" .or.Solvent_S_mol(i)=="AR" .or. Solvent_S_mol(i)=="ar") then
                        solvent_Z_mol(i) = 18
                  else
                        write(*,*)"Parsing: atom not recognized"
                  endif
                  solvent_M_mol(i) = solvent_atomic_mass(solvent_Z_mol(i))
            enddo
      end subroutine solvent_parse_atomic_symbol

      subroutine solvent_update_cm_coords()
            implicit none
            real*8              :: total_mass
            real*8              :: distv(3)
            real*8              :: xyz_aux(3,Natoms_per_mol)
            integer             :: mol,i

            total_mass = 0.d0
            solvent_cm_pos = 0.d0
            solvent_xyz_cm = 0.d0
            !$OMP PARALLEL DO REDUCTION(+:solvent_cm_pos) PRIVATE(total_mass,distv,xyz_aux,i)
            do mol=1,Nmols
                  xyz_aux(:,1) = solvent_xyz_mol(:,1,mol)
                  do i=2,Natoms_per_mol
                        distv = solvent_xyz_mol(:,1,mol) - solvent_xyz_mol(:,i,mol)
                        distv = distv-L_box*nint(distv/L_box)
                        xyz_aux(:,i) = solvent_xyz_mol(:,1,mol) - distv
                  end do

                  total_mass = 0.d0
                  do i=1,Natoms_per_mol
                        total_mass = total_mass + solvent_M_mol(i)
                        solvent_cm_pos(:,mol) = solvent_cm_pos(:,mol) + &
                        solvent_M_mol(i)*xyz_aux(:,i)
                  end do
                  solvent_cm_pos(:,mol) = solvent_cm_pos(:,mol)/total_mass
                  do i=1,Natoms_per_mol
                        solvent_xyz_cm(:,i,mol) = xyz_aux(:,i) - solvent_cm_pos(:,mol)
                  end do
            end do
            !$OMP END PARALLEL DO
      end subroutine solvent_update_cm_coords

      function solvent_comp_kinetic_energy() result(K)
            implicit none
            real*8 :: K
            integer :: mol,i
            K = 0.d0
            do mol=1,Nmols
                  do i=1,Natoms_per_mol
                        K = K + 0.5d0*solvent_M_mol(i)*sum(solvent_vel_mol(:,i,mol)**2)
                  end do
            end do
      end function solvent_comp_kinetic_energy

      subroutine solvent_update_cm_vel()
            implicit none
            real*8  :: total_mass
            integer :: mol,i
            solvent_cm_vel = 0.d0
            !$OMP PARALLEL DO REDUCTION(+:solvent_cm_vel) PRIVATE(total_mass)
            do mol=1,Nmols
                  total_mass=0.d0
                  do i=1,Natoms_per_mol
                        total_mass = total_mass + solvent_M_mol(i)
                        solvent_cm_vel(:,mol) = solvent_cm_vel(:,mol) + &
                        solvent_M_mol(i)*solvent_vel_mol(:,i,mol)
                  end do
                  solvent_cm_vel(:,mol) = solvent_cm_vel(:,mol)/total_mass
                  do i=1,Natoms_per_mol
                        solvent_vel_cm(:,i,mol) = solvent_vel_mol(:,i,mol) - solvent_cm_vel(:,mol)
                  end do
            end do
            !$OMP END PARALLEL DO
      end subroutine solvent_update_cm_vel


      subroutine init_solvent(Nmolecules,Nat,Ncentral,uniteq)
            implicit none
            integer,intent(in) :: Nmolecules,Nat,Ncentral
            integer,intent(in) :: uniteq
            integer            :: i

            Nmols = Nmolecules
            Natoms_central = Ncentral
            Natoms_per_mol = Nat
            solvent_Natoms = Natoms_per_mol*Nmols
            L_box = 0.d0
            !Allocate quantities
            allocate(solvent_S_mol(Natoms_per_mol),solvent_M_mol(Natoms_per_mol),solvent_Z_mol(Natoms_per_mol))
            allocate(solvent_xyz_mol(3,Natoms_per_mol,Nmols),solvent_xyz_cm(3,Natoms_per_mol,Nmols))
            allocate(solvent_xyz_eq(3,Natoms_per_mol),solvent_xyz_eckart(3,Natoms_per_mol,Nmols),solvent_cm_pos(3,Nmols))
            allocate(solvent_vel_mol(3,Natoms_per_mol,Nmols),solvent_vel_cm(3,Natoms_per_mol,Nmols))
            allocate(solvent_vel_vib(3,Natoms_per_mol,Nmols),solvent_cm_vel(3,Nmols))
            allocate(solvent_U_eckart(3,3,Nmols),solvent_omega_mol(3,Nmols))
            allocate(solvent_F(3,Natoms_central,Natoms_per_mol,Nmols),solvent_U(Natoms_central,Natoms_per_mol,Nmols))
            allocate(solvent_PT(Nmols),solvent_PR(Nmols))
            allocate(solvent_Fcentral(3,Natoms_central,Natoms_per_mol,Nmols))

            solvent_xyz_mol = 0.d0

            !Get equilibrium coordinates
            read(uniteq,*)
            read(uniteq,*)
            do i=1,Natoms_per_mol
                  read(uniteq,*)solvent_S_mol(i),solvent_xyz_eq(:,i)
            end do
            solvent_eq_cm_pos = 0.d0
            
            solvent_vel_cm = 0.d0
            solvent_vel_vib = 0.d0

            !Get masses from symbols
            call solvent_parse_atomic_symbol()
            !Obtain CoM coordinates
            call solvent_update_cm_coords()
            call solvent_update_cm_vel()
      end subroutine init_solvent

      function solvent_build_pseudo_inertia_moment(mol) result(I)
            implicit none
            integer :: mol
            real*8  :: I(3,3)
            real*8  :: delta_term
            integer :: j,a,b

            delta_term = 0.d0
            do j=1,Natoms_per_mol
                  delta_term = delta_term + solvent_M_mol(j)*&
                  sum(solvent_xyz_eckart(:,j,mol)*solvent_xyz_cm(:,j,mol))
            end do

            I = 0.d0
            do a=1,Natoms_per_mol
                  do b=1,Natoms_per_mol
                        if(a==b) I(a,b) = I(a,b) + delta_term
                        do j=1,Natoms_per_mol
                              I(a,b) = I(a,b) - solvent_M_mol(j)*&
                              solvent_xyz_cm(a,j,mol)*solvent_xyz_eckart(b,j,mol)
                        end do
                  end do
            end do
      end function solvent_build_pseudo_inertia_moment

      function solvent_invert_matrix(A) result(Ainv)
            implicit none
            real*8 :: A(3,3),Ainv(3,3)
            real*8 :: det
            det = (A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
                  - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
                  + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))
            det = 1.d0/det

            Ainv(1,1) = +det * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
            Ainv(2,1) = -det * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
            Ainv(3,1) = +det * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
            Ainv(1,2) = -det * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
            Ainv(2,2) = +det * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
            Ainv(3,2) = -det * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
            Ainv(1,3) = +det * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
            Ainv(2,3) = -det * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
            Ainv(3,3) = +det * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
      end function solvent_invert_matrix

      function solvent_cross_product(v1,v2) result(v3)
            implicit none
            real*8 :: v1(3),v2(3)
            real*8 :: v3(3)

            v3(1) =   v1(2)*v2(3) - v1(3)*v2(2)
            v3(2) = - v1(1)*v2(3) + v1(3)*v2(1)
            v3(3) =   v1(1)*v2(2) - v1(2)*v2(1)
      end function solvent_cross_product

      function solvent_comp_pseudo_angular_moment(mol) result(l)
            implicit none
            integer :: mol
            real*8  :: l(3)
            integer :: i
            l = 0.d0
            do i=1,Natoms_per_mol
                  l = l + solvent_M_mol(i)*&
                  solvent_cross_product(solvent_xyz_eckart(:,i,mol),solvent_vel_mol(:,i,mol))
            end do
      end function solvent_comp_pseudo_angular_moment

      subroutine solvent_comp_angular_vel()
            implicit none
            real*8  :: Iab(3,3),Iab_inv(3,3),l(3)
            integer :: mol
            do mol=1,Nmols
                  Iab = solvent_build_pseudo_inertia_moment(mol)
                  Iab_inv = solvent_invert_matrix(Iab)
            
                  l = solvent_comp_pseudo_angular_moment(mol)
                  
                  solvent_omega_mol(:,mol) = matmul(Iab_inv,l)
            end do
      end subroutine solvent_comp_angular_vel

      function solvent_build_eckart_matrix(mol) result(EM)
            implicit none
            integer :: mol
            real*8  :: xpa,xma,ypa,yma,zpa,zma
            real*8  :: EM(4,4)
            integer :: a
            EM = 0.d0
            do a=1,Natoms_per_mol
                  xpa = solvent_xyz_eq(1,a) + solvent_xyz_cm(1,a,mol)
                  xma = solvent_xyz_eq(1,a) - solvent_xyz_cm(1,a,mol)

                  ypa = solvent_xyz_eq(2,a) + solvent_xyz_cm(2,a,mol)
                  yma = solvent_xyz_eq(2,a) - solvent_xyz_cm(2,a,mol)

                  zpa = solvent_xyz_eq(3,a) + solvent_xyz_cm(3,a,mol)
                  zma = solvent_xyz_eq(3,a) - solvent_xyz_cm(3,a,mol)

                  !Diagonal
                  EM(1,1) = EM(1,1) + solvent_M_mol(a)*(xma**2 + yma**2 + zma**2)
                  EM(2,2) = EM(2,2) + solvent_M_mol(a)*(xma**2 + ypa**2 + zpa**2)
                  EM(3,3) = EM(3,3) + solvent_M_mol(a)*(xpa**2 + yma**2 + zpa**2)
                  EM(4,4) = EM(4,4) + solvent_M_mol(a)*(xpa**2 + ypa**2 + zma**2)

                  !Rest of 1st row
                  EM(1,2) = EM(1,2) + solvent_M_mol(a)*(ypa*zma - yma*zpa)
                  EM(1,3) = EM(1,3) + solvent_M_mol(a)*(xma*zpa - xpa*zma)
                  EM(1,4) = EM(1,4) + solvent_M_mol(a)*(xpa*yma - xma*ypa)

                  !Rest of 2nd row
                  EM(2,3) = EM(2,3) + solvent_M_mol(a)*(xma*yma - xpa*ypa)
                  EM(2,4) = EM(2,4) + solvent_M_mol(a)*(xma*zma - xpa*zpa)

                  !Rest of 3rd row
                  EM(3,4) = EM(3,4) + solvent_M_mol(a)*(yma*zma - ypa*zpa)
            end do

            !Symmetrize
            EM(2,1) = EM(1,2)
            EM(3,1) = EM(1,3)
            EM(4,1) = EM(1,4)

            EM(3,2) = EM(2,3)
            EM(4,2) = EM(2,4)

            EM(4,3) = EM(3,4)
      end function solvent_build_eckart_matrix

      function solvent_build_direction_cosine_matrix(V) result (U)
            implicit none
            real*8 :: V(4)
            real*8 :: U(3,3)
            real*8 :: q0,q1,q2,q3
            q0 = V(1)
            q1 = V(2)
            q2 = V(3)
            q3 = V(4)

            !1st row
            U(1,1) = q0**2 + q1**2 - q2**2 - q3**2
            U(1,2) = 2.d0*(q1*q2 + q0*q3)
            U(1,3) = 2.d0*(q1*q3 - q0*q2)
            
            !2nd row
            U(2,1) = 2.d0*(q1*q2 - q0*q3)
            U(2,2) = q0**2 - q1**2 + q2**2 - q3**2
            U(2,3) = 2.d0*(q2*q3 + q0*q1)

            !3rd row
            U(3,1) = 2.d0*(q1*q3 + q0*q2)
            U(3,2) = 2.d0*(q2*q3 - q0*q1)
            U(3,3) = q0**2 - q1**2 - q2**2 + q3**2
      end function solvent_build_direction_cosine_matrix

      subroutine solvent_check_eckart_conditions()
            implicit none
            real*8           :: tra_cond(3),rot_cond(3),comb_cond(3),disp(3,Natoms_per_mol)
            real*8,parameter :: eps=1.d-10
            integer          :: i,mol
            !$OMP PARALLEL DO PRIVATE(disp,tra_cond,rot_cond,comb_cond,i)
            do mol=1,Nmols
                  disp = solvent_xyz_cm(:,:,mol) - solvent_xyz_eckart(:,:,mol)
                  tra_cond = 0.d0
                  rot_cond = 0.d0
                  comb_cond = 0.d0
                  do i=1,Natoms_per_mol
                        tra_cond  = tra_cond  + solvent_M_mol(i)*disp(:,i)
                        rot_cond  = rot_cond  + solvent_M_mol(i)*&
                        solvent_cross_product(solvent_xyz_eckart(:,i,mol),disp(:,i))
                        comb_cond = comb_cond + solvent_M_mol(i)*&
                        solvent_cross_product(solvent_xyz_eckart(:,i,mol),solvent_xyz_cm(:,i,mol))
                  end do

                  if(any(tra_cond > eps) .or. any(rot_cond > eps) .or. any(comb_cond > eps)) then
                        write(0,*)"Eckart conditions not satisfied"
                        write(0,*)"Eckart Conditons:"
                        write(0,"(A,2X,3(E16.8,2X))")"Translational:",tra_cond
                        write(0,"(A,2X,3(E16.8,2X))")"Rotational:   ",rot_cond
                        write(0,"(A,2X,3(E16.8,2X))")"Combined:     ",comb_cond
                        write(0,*)""
                  end if
            end do
            !$OMP END PARALLEL DO
      end subroutine solvent_check_eckart_conditions


      subroutine solvent_get_eckart_frame()
            implicit none
            real*8             :: EM(4,4),U(3,3)
            real*8             :: work(100),d(4)
            integer            :: i,mol,ierror

            call solvent_update_cm_coords()
            !$OMP PARALLEL DO PRIVATE(EM,U,work,d,i,ierror)
            do mol=1,Nmols
                  EM = solvent_build_eckart_matrix(mol)

                  call dsyev("V","U",4,EM,4,d,work,100,ierror)

                  U = solvent_build_direction_cosine_matrix(EM(:,1))

                  do i=1,Natoms_per_mol
                        solvent_xyz_eckart(:,i,mol) = matmul(transpose(U),solvent_xyz_eq(:,i))
                  end do
                  solvent_U_eckart(:,:,mol) = U
            end do
            !$OMP END PARALLEL DO
            call solvent_check_eckart_conditions()
      end subroutine solvent_get_eckart_frame


      subroutine init_solvent_forcefield(unit)
            implicit none
            integer,intent(in) :: unit
            character          :: dummy*90,units*90,pot_type*90
            real*8             :: aux1,aux2,aux3
            integer            :: i,a,b,dummy2

            read(unit,*)dummy,a
            read(unit,*)dummy,b

            if(Natoms_central /= a) then
                  write(0,*)"Number of central molecule atoms do not match, not initialized correctly, aborting..."
                  stop
            end if

            if(Natoms_per_mol /= b) then
                  write(0,*)"Atom types do not match atoms per molecule, not initialized correctly, aborting..."
                  stop
            end if

            !Read pair info
            read(unit,*)pot_type
            read(unit,*)units
            read(unit,*)pair_cut
            if(pot_type=="LJ") then
                  allocate(pair_coefs(2,Natoms_per_mol,Natoms_central))
            elseif(pot_type=="BU") then
                  allocate(pair_coefs(3,Natoms_per_mol,Natoms_central))
            elseif(pot_type=="H") then
                  allocate(pair_coefs(2,Natoms_per_mol,Natoms_central))
            else
                  write(0,*)"Pair potential not supported, aborting..."
                  stop
            end if

            pair_pot_type = pot_type
            !Read pair coefs
            do i=1,Natoms_central*Natoms_per_mol
                  if(pot_type=="LJ") then
                        read(unit,*)a,b,aux1,aux2
                        pair_coefs(:,a,b) = (/aux1,aux2/)
                        !Correct units if necessary
                        if(units=="KC") then
                              pair_coefs(1,a,b) = pair_coefs(1,a,b)*solvent_kcal_to_kj
                        end if
                  end if
                  if(pot_type=="BU") then
                        read(unit,*)a,b,aux1,aux2,aux3
                        pair_coefs(:,a,b) = (/aux1,aux2,aux3/)
                        !Correct units if necessary
                        if(units=="KC") then
                              pair_coefs(1,a,b) = pair_coefs(1,a,b)*solvent_kcal_to_kj
                              pair_coefs(3,a,b) = pair_coefs(3,a,b)*solvent_kcal_to_kj
                        end if
                  end if
                  if(pot_type=="H") then
                        read(unit,*)a,b,aux1,aux2
                        pair_coefs(:,a,b) = (/aux1,aux2/)
                        !Correct units if necessary
                        if(units=="KC") then
                              pair_coefs(1,a,b) = pair_coefs(1,a,b)*solvent_kcal_to_kj
                        end if
                  end if
            end do
            read(unit,*,end=10) !Check if end file, blank line should be there if continuing

            read(unit,*)dummy,long_cut
            read(unit,*)dummy
            allocate(q_solvent(Natoms_per_mol))
            do i=1,Natoms_per_mol
                  read(unit,*)dummy2, q_solvent(i)
            end do

            read(unit,*,end=10)

            read(unit,*)dummy
            allocate(q_central(Natoms_central))
            do i=1,Natoms_central
                  read(unit,*)dummy2, q_central(i)
            end do

            10 continue
      end subroutine init_solvent_forcefield

      subroutine comp_forces_on_solvent(xyz_central)
            implicit none
            real*8,intent(in)  :: xyz_central(3,Natoms_central)
            real*8             :: distv(3),dist
            real*8             :: epsilon,sigma
            real*8             :: Abu,bbu,Cbu
            real*8             :: k,r0,r0v(3)
            integer            :: i,j,mol

            
            solvent_F = 0.d0
            solvent_U = 0.d0
            !$OMP PARALLEL DO REDUCTION(+:solvent_F,solvent_U) PRIVATE(distv,dist,epsilon,sigma,Abu,bbu,Cbu,k,r0,r0v,i,j,mol)
            do mol=1,Nmols
                  do i=1,Natoms_per_mol
                        do j=1,Natoms_central
                              distv = solvent_xyz_mol(:,i,mol)-xyz_central(:,j)
                              distv = distv-L_box*nint(distv/L_box)
                              dist = sqrt(sum(distv**2))
                              if(any(dist>L_box)) write(0,*)"WARNING: distance greater than box length"

                              !Coulomb part
                              if(dist<long_cut) then
                                    solvent_F(:,j,i,mol) = solvent_F(:,j,i,mol) &
                                    + distv*electrostatic_constant*q_solvent(i)*q_central(j)/dist**3

                                    solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                    + electrostatic_constant*q_solvent(i)*q_central(j)/dist
                              end if
                              !Pair part
                              if(dist<pair_cut) then
                                    if(pair_pot_type=="LJ") then
                                          epsilon = pair_coefs(1,i,j)
                                          sigma = pair_coefs(2,i,j)
                                          solvent_F(:,j,i,mol) = solvent_F(:,j,i,mol) &
                                          + epsilon*((48.d0*sigma**12)/dist**14 - (24.d0*sigma**6)/dist**8)*distv

                                          solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                          + 4.d0*epsilon*((sigma/dist)**12 - (sigma/dist)**6)
                                    else if(pair_pot_type=="BU") then
                                          Abu = pair_coefs(1,i,j)
                                          bbu = pair_coefs(2,i,j)
                                          Cbu = pair_coefs(3,i,j)
                                          solvent_F(:,j,i,mol) = solvent_F(:,j,i,mol) &
                                          - (6.d0*Cbu/dist**8 - Abu*bbu*exp(-bbu*dist)/dist)*distv

                                          solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                          + Abu*exp(-bbu*dist) - Cbu/dist**6
                                    else if(pair_pot_type=="H") then
                                          k = pair_coefs(1,i,j)
                                          r0 = pair_coefs(2,i,j)
                                          r0v = (/r0,0.d0,0.d0/)
                                          solvent_F(1,j,i,mol) = solvent_F(1,j,i,mol) &
                                          + k*(dist - r0)

                                          solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                          + 0.5d0*k*(dist-r0)**2
                                    end if
                              end if    
                        end do
                  end do
            end do
            !$OMP END PARALLEL DO
      end subroutine comp_forces_on_solvent

      subroutine comp_forces_on_central(xyz_central)
            implicit none
            real*8,intent(in)  :: xyz_central(3,Natoms_central)
            real*8             :: distv(3),dist
            real*8             :: epsilon,sigma
            real*8             :: Abu,bbu,Cbu
            integer            :: i,j,mol

            
            solvent_Fcentral = 0.d0
            ! solvent_U = 0.d0
            do mol=1,Nmols
                  do i=1,Natoms_per_mol
                        do j=1,Natoms_central
                              distv = xyz_central(:,j)-solvent_xyz_mol(:,i,mol)
                              distv = distv-L_box*nint(distv/L_box)
                              dist = sqrt(sum(distv**2))

                              !Coulomb part
                              solvent_Fcentral(:,j,i,mol) = solvent_Fcentral(:,j,i,mol) &
                              + distv*electrostatic_constant*q_solvent(i)*q_central(j)/dist**3

                              ! solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                              ! + electrostatic_constant*q_solvent(i)*q_central(j)/dist

                              !Pair part
                              if(dist<pair_cut) then
                                    if(pair_pot_type=="LJ") then
                                          epsilon = pair_coefs(1,i,j)
                                          sigma = pair_coefs(2,i,j)
                                          solvent_Fcentral(:,j,i,mol) = solvent_Fcentral(:,j,i,mol) &
                                          + epsilon*((48.d0*sigma**12)/dist**14 - (24.d0*sigma**6)/dist**8)*distv

                                          ! solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                          ! + 4.d0*epsilon*(sigma**12/dist**12 - sigma**6/dist**6)
                                    else if(pair_pot_type=="BU") then
                                          Abu = pair_coefs(1,i,j)
                                          bbu = pair_coefs(2,i,j)
                                          Cbu = pair_coefs(3,i,j)
                                          solvent_Fcentral(:,j,i,mol) = solvent_Fcentral(:,j,i,mol) &
                                          - (6.d0*Cbu/dist**8 - Abu*bbu*exp(-bbu*dist)/dist)*distv

                                          ! solvent_U(j,i,mol) = solvent_U(j,i,mol) &
                                          ! + Abu*exp(-bbu*dist) - Cbu/dist**6
                                    end if
                              end if    
                        end do
                  end do
            end do
      end subroutine comp_forces_on_central

      subroutine comp_power_on_solvent(xyz_central)
            implicit none
            real*8,intent(in) :: xyz_central(3,Natoms_central)
            integer :: i,j,mol
            real*8  :: total_F(3),vrot(3),distv(3)

            call solvent_update_cm_vel()
            if(Natoms_per_mol>1)call solvent_get_eckart_frame()
            if(Natoms_per_mol>1)call solvent_comp_angular_vel()

            solvent_PT = 0.d0
            solvent_PR = 0.d0
            !$OMP PARALLEL DO REDUCTION(+:solvent_PT,solvent_PR) PRIVATE(total_F,distv,vrot,i,j)
            do mol=1,Nmols
                  do j=1,Natoms_central
                        total_F = sum(solvent_F(:,j,:,mol),2)

                        solvent_PT(mol) = solvent_PT(mol) + sum(total_F*solvent_cm_vel(:,mol))

                        do i=1,Natoms_per_mol

                              distv = solvent_xyz_mol(:,i,mol)-xyz_central(:,j)
                              distv = distv-L_box*nint(distv/L_box)

                              vrot = solvent_cross_product(solvent_omega_mol(:,mol),solvent_xyz_cm(:,i,mol))
                              solvent_PR(mol) = solvent_PR(mol) + sum(vrot*solvent_F(:,j,i,mol))
                        end do
                  end do
            end do
            !$OMP END PARALLEL DO
      end subroutine comp_power_on_solvent


      subroutine solvent_write_conf(r,port)
            implicit none
            integer,intent(in) :: port
            real*8,intent(in)  :: r(3,Natoms_per_mol,Nmols)
            integer            :: i,mol

            write(port,"(I5)")Natoms_per_mol*Nmols
            write(port,*)""
            do mol=1,Nmols
                  do i=1,Natoms_per_mol
                        write(port,"(A,2X,E20.10,2X,E20.10,2X,E20.10)")solvent_S_mol(i),r(:,i,mol)
                  end do
            enddo
      end subroutine solvent_write_conf

      subroutine solvent_write_conf_with_central(r,rcentral,S_central,port)
            implicit none
            integer,intent(in)   :: port
            real*8,intent(in)    :: r(3,Natoms_per_mol,Nmols),rcentral(3,Natoms_central)
            character,intent(in) :: S_central(Natoms_central)*2
            integer              :: i,mol

            write(port,"(I5)")Natoms_per_mol*Nmols+Natoms_central
            write(port,*)""
            do mol=1,Nmols
                  do i=1,Natoms_per_mol
                        write(port,"(A,2X,E20.10,2X,E20.10,2X,E20.10)")solvent_S_mol(i),r(:,i,mol)
                  end do
            enddo
            do i=1,Natoms_central
                  write(port,"(A,2X,E20.10,2X,E20.10,2X,E20.10)")S_central(i),rcentral(:,i)
            end do
      end subroutine solvent_write_conf_with_central



end module solvent_processor

