module auxvib
      implicit none

      contains
      subroutine vibrational_analysis(N,xyz_eqin,xyz,vel,mass,L_box,cm_pos,xyz_cm,xyz_eckart,cm_vel,vel_cm,omega,vel_rot,vel_vib)
            implicit none
            integer,intent(in) :: N
            real*8,intent(in)  :: xyz_eqin(3,N),xyz(3,N),vel(3,N),mass(N),L_box
            real*8,intent(out) :: cm_pos(3),xyz_cm(3,N),xyz_eckart(3,N)
            real*8,intent(out) :: cm_vel(3),vel_cm(3,N),omega(3),vel_rot(3,N),vel_vib(3,N)

            real*8             :: xyz_eq(3,N)
            !Eckart matrix and diagonalization
            real*8             :: EM(4,4),d(4),U(3,3)
            real*8,allocatable :: work(:)
            integer            :: Nwork

            integer            :: i
            !Compute CoM of input eq coordinates (just in case)
            call compute_cm_coords(N,xyz_eqin,vel,mass, cm_pos,xyz_eq,cm_vel,vel_cm,L_box)
            !Compute CoM coordinates and velocities and build eckart matrix
            call compute_cm_coords(N,xyz,vel,mass, cm_pos,xyz_cm,cm_vel,vel_cm,L_box)
            EM = build_eckart_matrix(N,xyz_eq,xyz_cm,mass)

            !Diagonalize eckart matrix
            allocate(work(1))
            call dsyev("V","U",4,EM,4,d,work,-1,i)
            if(i/=0) then
                  write(0,*)"Something went wrong when determining Nwork, aborting..."
                  stop
            end if
            Nwork = int(work(1))
            deallocate(work)
            allocate(work(Nwork))
            call dsyev("V","U",4,EM,4,d,work,Nwork,i)
            !Build eckart rotation matrix
            U = build_direction_cosine_matrix(EM(:,1))
            do i=1,N
                  xyz_eckart(:,i) = matmul(transpose(U),xyz_eq(:,i)) !Eq coordinates to Eckart frame
            end do
            call check_eckart_conditions(N,xyz_cm,xyz_eckart,mass)

            omega = comp_angular_vel(N,xyz_cm,xyz_eckart,vel,mass)
            do i=1,N
                  vel_rot(:,i) = cross_product(omega,xyz_cm(:,i))
                  vel_vib(:,i) = vel(:,i) - cm_vel - vel_rot(:,i)
            end do
      end subroutine vibrational_analysis

      subroutine compute_cm_coords(N,xyz,vel,mass,cm_pos,xyz_cm,cm_vel,vel_cm,L_box)
            implicit none
            integer,intent(in) :: N
            real*8,intent(in)  :: xyz(3,N),vel(3,N),mass(N),L_box
            real*8,intent(out) :: cm_pos(3),xyz_cm(3,N)
            real*8,intent(out) :: cm_vel(3),vel_cm(3,N)

            real*8             :: distv(3)
            real*8             :: xyz_aux(3,N)
            integer            :: i

            xyz_cm = 0.d0
            cm_pos = 0.d0
            cm_vel = 0.d0

            xyz_aux(:,1) = xyz(:,1)
            do i=2,N
                  distv = xyz(:,1) - xyz(:,i)
                  distv = distv-L_box*nint(distv/L_box)
                  xyz_aux(:,i) = xyz(:,1) - distv
            end do

            do i=1,N
                  cm_pos = cm_pos + mass(i)*xyz_aux(:,i)
                  cm_vel = cm_vel + mass(i)*vel(:,i)
            end do
            cm_pos = cm_pos/sum(mass)
            cm_vel = cm_vel/sum(mass)
            do i=1,N
                  xyz_cm(:,i) = xyz_aux(:,i) - cm_pos
                  vel_cm(:,i) = vel(:,i) - cm_vel
            end do
      end subroutine compute_cm_coords

      function build_eckart_matrix(N,xyz_eq,xyz_cm,mass) result(EM)
            implicit none
            integer :: N
            real*8  :: xyz_eq(3,N),xyz_cm(3,N),mass(N)

            real*8  :: xpa,xma,ypa,yma,zpa,zma
            real*8  :: EM(4,4)
            integer :: a
            EM = 0.d0
            do a=1,N
                  xpa = xyz_eq(1,a) + xyz_cm(1,a)
                  xma = xyz_eq(1,a) - xyz_cm(1,a)

                  ypa = xyz_eq(2,a) + xyz_cm(2,a)
                  yma = xyz_eq(2,a) - xyz_cm(2,a)

                  zpa = xyz_eq(3,a) + xyz_cm(3,a)
                  zma = xyz_eq(3,a) - xyz_cm(3,a)

                  !Diagonal
                  EM(1,1) = EM(1,1) + mass(a)*(xma**2 + yma**2 + zma**2)
                  EM(2,2) = EM(2,2) + mass(a)*(xma**2 + ypa**2 + zpa**2)
                  EM(3,3) = EM(3,3) + mass(a)*(xpa**2 + yma**2 + zpa**2)
                  EM(4,4) = EM(4,4) + mass(a)*(xpa**2 + ypa**2 + zma**2)

                  !Rest of 1st row
                  EM(1,2) = EM(1,2) + mass(a)*(ypa*zma - yma*zpa)
                  EM(1,3) = EM(1,3) + mass(a)*(xma*zpa - xpa*zma)
                  EM(1,4) = EM(1,4) + mass(a)*(xpa*yma - xma*ypa)

                  !Rest of 2nd row
                  EM(2,3) = EM(2,3) + mass(a)*(xma*yma - xpa*ypa)
                  EM(2,4) = EM(2,4) + mass(a)*(xma*zma - xpa*zpa)

                  !Rest of 3rd row
                  EM(3,4) = EM(3,4) + mass(a)*(yma*zma - ypa*zpa)
            end do

            !Symmetrize
            EM(2,1) = EM(1,2)
            EM(3,1) = EM(1,3)
            EM(4,1) = EM(1,4)

            EM(3,2) = EM(2,3)
            EM(4,2) = EM(2,4)

            EM(4,3) = EM(3,4)
      end function build_eckart_matrix

      function build_direction_cosine_matrix(V) result (U)
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
      end function build_direction_cosine_matrix

      function comp_angular_vel(N,xyz_cm,xyz_eckart,vel,mass) result(omega)
            implicit none
            integer :: N
            real*8  :: xyz_cm(3,N),xyz_eckart(3,N),vel(3,N),mass(N)
            real*8  :: omega(3)
            real*8  :: Iab(3,3),Iab_inv(3,3),l(3)

            Iab = build_pseudo_inertia_moment(N,xyz_cm,xyz_eckart,mass)
            Iab_inv = invert_matrix(Iab)

            l = comp_pseudo_angular_moment(N,xyz_eckart,vel,mass)

            omega = matmul(Iab_inv,l)
      end function comp_angular_vel

      function build_pseudo_inertia_moment(N,xyz_cm,xyz_eckart,mass) result(I)
            implicit none
            integer :: N
            real*8  :: xyz_cm(3,N),xyz_eckart(3,N),mass(N)
            real*8  :: I(3,3)
            real*8  :: delta_term
            integer :: j,a,b

            delta_term = 0.d0
            do j=1,N
                  delta_term = delta_term + mass(j)*sum(xyz_eckart(:,j)*xyz_cm(:,j))
            end do

            I = 0.d0
            do a=1,3
                  do b=1,3
                        if(a==b) I(a,b) = I(a,b) + delta_term
                        do j=1,N
                              I(a,b) = I(a,b) - mass(j)*xyz_cm(a,j)*xyz_eckart(b,j)
                        end do
                  end do
            end do
      end function build_pseudo_inertia_moment

      function invert_matrix(A) result(Ainv)
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
      end function invert_matrix

      function comp_pseudo_angular_moment(N,xyz_eckart,vel,mass) result(l)
            implicit none
            integer :: N
            real*8  :: xyz_eckart(3,N),vel(3,N),mass(N)
            real*8  :: l(3)
            integer :: i
            l = 0.d0
            do i=1,N
                  l = l + mass(i)*cross_product(xyz_eckart(:,i),vel(:,i))
            end do
      end function comp_pseudo_angular_moment


      function unit_cross(u1,u2) result(u3)
            implicit none
            real*8 :: u1(3),u2(3)
            real*8 :: u3(3)

            u3(1) = u1(2)*u2(3) - u1(3)*u2(2)
            u3(2) = - u1(1)*u2(3) + u1(3)*u2(1)
            u3(3) = u1(1)*u2(2) - u1(2)*u2(1)

            u3 = u3/sqrt(sum(u3**2))
      end function unit_cross

      function cross_product(v1,v2) result(v3)
            implicit none
            real*8 :: v1(3),v2(3)
            real*8 :: v3(3)

            v3(1) =   v1(2)*v2(3) - v1(3)*v2(2)
            v3(2) = - v1(1)*v2(3) + v1(3)*v2(1)
            v3(3) =   v1(1)*v2(2) - v1(2)*v2(1)
      end function cross_product

      function comp_norm(u) result(a)
            implicit none
            real*8 :: u(:),a
            a = sqrt(sum(u**2))
      end function comp_norm

      subroutine check_eckart_conditions(N,xyz_cm,xyz_eckart,mass)
            implicit none
            integer :: N
            real*8  :: xyz_cm(3,N),xyz_eckart(3,N),mass(N)
            real*8           :: tra_cond(3),rot_cond(3),comb_cond(3),disp(3,N)
            real*8,parameter :: eps=1.d-10
            integer          :: i

            disp = xyz_cm - xyz_eckart
            tra_cond = 0.d0
            rot_cond = 0.d0
            comb_cond = 0.d0
            do i=1,N
                  tra_cond  = tra_cond  + mass(i)*disp(:,i)
                  rot_cond  = rot_cond  + mass(i)*cross_product(xyz_eckart(:,i),disp(:,i))
                  comb_cond = comb_cond + mass(i)*cross_product(xyz_eckart(:,i),xyz_cm(:,i))
            end do

            if(any(tra_cond > eps) .or. any(rot_cond > eps) .or. any(comb_cond > eps)) then
                  write(0,*)"Eckart conditions not satisfied"
                  write(0,*)"Eckart Conditons:"
                  write(0,"(A,2X,3(E16.8,2X))")"Translational:",tra_cond
                  write(0,"(A,2X,3(E16.8,2X))")"Rotational:   ",rot_cond
                  write(0,"(A,2X,3(E16.8,2X))")"Combined:     ",comb_cond
                  write(0,*)""
            end if
      end subroutine check_eckart_conditions
end module auxvib
