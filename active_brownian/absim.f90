program ABSIM
use mtfort90
implicit none
integer          :: i,j,k,sample
real*8,parameter :: PI = 4.d0*atan(1.d0)

!Parameters
integer   :: D,Nt,nmeas,Nsamples,time(8),seed,Npar
integer   :: nseed
integer,allocatable :: seeds(:)
real*8    :: deltat,DR,DT,par(10)
character :: v_mode*90

real*8    :: aux_random1,rnd1,rnd2
!Used for random sequences
integer,allocatable :: aux_random_sequence(:)
integer   :: seqindex
logical   :: sorted_flag

!Trajectory
real*8              :: t
real*8,allocatable  :: xyz0(:),xyz(:),vel(:)
real*8              :: phi,theta
real*8,allocatable  :: orientation(:),orientationi(:)

!Measurements
real*8,allocatable  :: MSD(:,:),MSD2(:,:)

!Time lags
real*8,allocatable :: aux_lags(:)
!MSD
real*8,allocatable :: cm_positions(:,:),MSD_lags(:,:),MSD_lags2(:,:)
!MASD
real*8,allocatable :: orientations(:,:),MASD_lags(:),MASD_lags2(:)

real*8,allocatable  :: vmean(:,:),vmean2(:,:)
real*8,allocatable  :: vmeanabs(:,:),vmeanabs2(:,:)

real*8,allocatable  :: phimean(:),phimean2(:)
real*8,allocatable  :: thetamean(:),thetamean2(:)

real*8              :: globalvmean,globalvmean2
real*8              :: globalvnoprojmean,globalvnoprojmean2

!Distributions
integer,parameter   :: Nbins = 50
real*8,parameter    :: dphi = (2.d0*PI)/Nbins,dtheta = 2.d0/Nbins,dangdisp = PI/Nbins
real*8              :: phidist(Nbins,4),thetadist(Nbins,4),angdispdist(Nbins,4)
real*8              :: angdisp

!Performance
real*8              :: auxperf,global_start,global_stop,integration_time,acu_time,stat_time,start_time,stop_time

character :: input_filename*90,output_path*90

namelist /input/ D,deltat,Nt,nmeas,Nsamples,DR,DT,Npar,v_mode,par,seed,output_path

call get_command_argument(1,input_filename)
open(1,file=input_filename)

read(unit=1, nml=input)
close(1)

call execute_command_line("mkdir -p "//trim(output_path))

allocate(xyz0(D),xyz(D),vel(D))
allocate(orientation(D),orientationi(D))

allocate(MSD((Nt/nmeas)+1,D+1),MSD2((Nt/nmeas)+1,D+1))
allocate(aux_lags((Nt/nmeas)))
allocate(cm_positions(Nt+1,D),MSD_lags((Nt/nmeas),D+1),MSD_lags2((Nt/nmeas),D+1))
allocate(orientations(Nt+1,D),MASD_lags((Nt/nmeas)),MASD_lags2((Nt/nmeas)))
allocate(vmean((Nt/nmeas)+1,D+1),vmean2((Nt/nmeas)+1,D+1))
allocate(vmeanabs((Nt/nmeas)+1,D+1),vmeanabs2((Nt/nmeas)+1,D+1))
allocate(phimean((Nt/nmeas)+1),phimean2((Nt/nmeas)+1))
allocate(thetamean((Nt/nmeas)+1),thetamean2((Nt/nmeas)+1))

!Using mtfort90
if(seed==0) then
      call date_and_time(values=time)
      seed = time(5)*3600+time(6)*60+time(7)
end if
call sgrnd(seed) !Initialize RNG.

!Using intrinsic
! call random_seed()
! call random_seed(size = nseed)
! allocate(seeds(nseed))
! call random_seed(get=seeds)

MSD         = 0.d0
MSD2        = 0.d0
MSD_lags    = 0.d0
MSD_lags2   = 0.d0
MASD_lags   = 0.d0
MASD_lags2  = 0.d0
vmean       = 0.d0
vmean2      = 0.d0
vmeanabs    = 0.d0
vmeanabs2   = 0.d0
phimean     = 0.d0
phimean2    = 0.d0
thetamean   = 0.d0
thetamean2  = 0.d0

globalvmean        = 0.d0
globalvmean2       = 0.d0
globalvnoprojmean  = 0.d0
globalvnoprojmean2 = 0.d0

phidist   = 0.d0
thetadist = 0.d0

integration_time   = 0.d0
acu_time           = 0.d0
stat_time          = 0.d0

!Write trajectory to file if single sample
if(Nsamples==1) open(201,file=trim(output_path)//"traj.xyz")

call cpu_time(global_start)
do sample=1,Nsamples
      !Progress bar
      call cpu_time(global_stop)
      auxperf = int((Nsamples-sample-1)*(global_stop-global_start)/sample)
      write (*,"(A,I10,A,I10,A,I4,A,I2.2)",advance="no") "Sample",sample," of ",Nsamples," | "&
      ,int(auxperf/60),":",nint(auxperf-int(auxperf/60)*60)
      if (sample.le.Nsamples) call execute_command_line('echo "\033[A"')

      !Initialize time, position and velocities
      t    = 0.d0
      xyz0 = 0.d0
      xyz  = 0.d0
      vel  = 0.d0

      !Auxiliary random variables
      aux_random1 = grnd()
      ! call random_number(aux_random1)

      if(v_mode=="EXPPOISSON") then
            seqindex = 0
            !Generate random sequence
            if(.not. allocated(aux_random_sequence)) allocate(aux_random_sequence(int(par(3))))
            do j=1,int(par(3))
                  aux_random_sequence(j) = floor((Nt+1)*grnd())
            end do
            !Sort random sequence
            sorted_flag = .false.
            do while(.not. sorted_flag)
                  do j=1,int(par(3))-1
                        if(aux_random_sequence(j)>aux_random_sequence(j+1)) then
                              k = aux_random_sequence(j+1)
                              aux_random_sequence(j+1) = aux_random_sequence(j)
                              aux_random_sequence(j) = k
                        end if
                  end do
                  check: do j=1,int(par(3))-1
                        if(aux_random_sequence(j)<=aux_random_sequence(j+1)) then
                              sorted_flag = .true.
                        else
                              sorted_flag = .false.
                              exit check
                        end if
                  end do check
            end do
      elseif(v_mode=="EXPPMRND") then
            if(.not. allocated(aux_random_sequence)) allocate(aux_random_sequence(int(Nt/par(3))))
            do i=1,int(Nt/par(3))
                  if(grnd()>0.5d0) then
                        aux_random_sequence(i) = +1.d0
                  else 
                        aux_random_sequence(i) = -1.d0
                  end if
            end do
      end if

      !Initialize orientation
      select case(D)
            case(1)
                  phi  = PI/2.d0
                  theta  = PI
                  orientation = 0.d0
            case(2)
                  phi  = 2.d0*PI*(grnd()-0.5d0)
                  ! call random_number(rnd1)
                  ! phi  = 2.d0*PI*(rnd1-0.5d0)
                  theta  = PI
                  orientation = (/cos(phi),sin(phi)/)
            case(3)
                  orientation = (/gaussian_number(),gaussian_number(),gaussian_number()/)
                  orientation = orientation/sqrt(sum(orientation**2))
                  call get_phi_theta_from_n(orientation,phi,theta)
                  ! if(phi>= PI) phi = phi - 2.d0*PI
                  ! if(phi<=-PI) phi = phi + 2.d0*PI
                  ! if(theta>=   PI) theta = theta - 2.d0*PI
                  ! if(theta<= 0.d0) theta = theta + 2.d0*PI
      end select
      orientationi = orientation

      i = 0
      !Initialize observables
      MSD(1,1:D)    = MSD(1,1:D)  +    (xyz)**2
      MSD(1,D+1)    = MSD(1,D+1)  + sum(xyz**2)
      MSD2(1,1:D)   = MSD2(1,1:D) +    (xyz)**4
      MSD2(1,D+1)   = MSD2(1,D+1) + sum(xyz**2)**2

      vmean(1,1:D)  = vmean(1,1:D)  + vel
      vmean(1,D+1)  = vmean(1,D+1)    + sum(vel*orientation)
      vmean2(1,1:D) = vmean2(1,1:D) + vel**2
      vmean2(1,D+1) = vmean2(1,D+1)   + sum(vel*orientation)**2

      vmeanabs(1,1:D)  = vmeanabs(1,1:D)  + abs(vel)
      vmeanabs(1,D+1)  = vmeanabs(1,D+1)    + abs(sum(vel*orientation))
      vmeanabs2(1,1:D) = vmeanabs2(1,1:D) + vel**2
      vmeanabs2(1,D+1) = vmeanabs2(1,D+1)   + sum(vel*orientation)**2

      phimean(1)    = phimean(1)  + phi
      phimean2(1)   = phimean2(1) + phi**2
      thetamean(1)  = thetamean(1)  + theta
      thetamean2(1) = thetamean2(1) + theta**2

      !Origins for lags
      cm_positions(1,:) = xyz
      orientations(1,:) = orientation

      !Distributions initial time
      phidist(floor((phi+PI)/dphi)+1,1) = phidist(floor((phi+PI)/dphi)+1,1) + 1
      thetadist(floor((cos(theta)+1.d0)/dtheta)+1,1) = thetadist(floor((cos(theta)+1.d0)/dtheta)+1,1) + 1
      angdisp = sum(orientation*orientationi)
      if(abs(angdisp-1.d0)<10.d0*epsilon(angdisp)) angdisp = sign(1.d0,angdisp)
      angdispdist(floor(acos(angdisp)/dangdisp)+1,1) = &
      angdispdist(floor(acos(angdisp)/dangdisp)+1,1) + 1
      !Distributions all time
      phidist(floor((phi+PI)/dphi)+1,4) = phidist(floor((phi+PI)/dphi)+1,4) + 1
      thetadist(floor((cos(theta)+1.d0)/dtheta)+1,4) = thetadist(floor((cos(theta)+1.d0)/dtheta)+1,4) + 1
      angdispdist(floor(acos(angdisp)/dangdisp)+1,4) = &
      angdispdist(floor(acos(angdisp)/dangdisp)+1,4) + 1

      do i = 1,Nt
            call cpu_time(start_time)
            if(Nsamples==1) write(201,"(2(E20.13,2X))")xyz
            !Integrate
            select case(v_mode)
                  case("CONSTANT") 
                        call integration_step(t,deltat,D,xyz,phi,theta,constant_speed,DR,DT)
                  case("PW")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_pw,DR,DT)
                  case("PERIODIC")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_gaussian,DR,DT)
                  case("LINEXP")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_linexp,DR,DT)
                  case("EXP")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_exp,DR,DT)
                  case("EXPPM")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_exp_pm,DR,DT)
                  case("EXPPMRND")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_exp_pmrnd,DR,DT)
                  case("EXPRND")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_exp_rnd,DR,DT)
                  case("EXPPOISSON")
                        call integration_step(t,deltat,D,xyz,phi,theta,periodic_exp_poisson,DR,DT)
                  case("SINE")
                        call integration_step(t,deltat,D,xyz,phi,theta,sine,DR,DT)
                  case("ABSSINE")
                        call integration_step(t,deltat,D,xyz,phi,theta,abssine,DR,DT)
                  case("SINERND")
                        call integration_step(t,deltat,D,xyz,phi,theta,sine_rnd,DR,DT)
                  case("HARM")
                        call integration_step(t,deltat,D,xyz,phi,theta,harm,DR,DT)
            end select
            !Roll back angles and recompute orientation
            select case(D)
                  case(1)
                        orientation = 0.d0
                  case(2)
                        if(phi>= PI) phi = phi - 2.d0*PI
                        if(phi<=-PI) phi = phi + 2.d0*PI
                        orientation = (/cos(phi),sin(phi)/)
                  case(3)
                        ! if(phi>= PI) phi = phi - 2.d0*PI
                        ! if(phi<=-PI) phi = phi + 2.d0*PI
                        ! if(theta>=   PI) theta = theta - 2.d0*PI
                        ! if(theta<= 0.d0) theta = theta + 2.d0*PI
                        orientation = (/cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)/)
                        orientation = orientation/sqrt(sum(orientation**2))
            end select


            t = i*deltat
            vel = (xyz-xyz0)/deltat
            xyz0 = xyz
            call cpu_time(stop_time)
            integration_time = integration_time + stop_time-start_time
            call cpu_time(start_time)

            !Origins for lags
            cm_positions(i+1,:) = xyz
            orientations(i+1,:) = orientation
            if(mod(i,nmeas)==0) then
                  MSD((i/nmeas)+1,1:D)  = MSD((i/nmeas)+1,1:D)  +    (xyz)**2
                  MSD((i/nmeas)+1,D+1)  = MSD((i/nmeas)+1,D+1)  + sum(xyz**2)
                  MSD2((i/nmeas)+1,1:D) = MSD2((i/nmeas)+1,1:D) +    (xyz)**4
                  MSD2((i/nmeas)+1,D+1) = MSD2((i/nmeas)+1,D+1) + sum(xyz**2)**2
            
                  vmean((i/nmeas)+1,1:D)   = vmean((i/nmeas)+1,1:D)  + vel
                  vmean((i/nmeas)+1,D+1)   = vmean((i/nmeas)+1,D+1)    + sum(vel*orientation)
                  vmean2((i/nmeas)+1,1:D)  = vmean2((i/nmeas)+1,1:D) + vel**2
                  vmean2((i/nmeas)+1,D+1)  = vmean2((i/nmeas)+1,D+1)   + sum(vel*orientation)**2

                  vmeanabs((i/nmeas)+1,1:D)  = vmeanabs((i/nmeas)+1,1:D)  + abs(vel)
                  vmeanabs((i/nmeas)+1,D+1)  = vmeanabs((i/nmeas)+1,D+1)    + abs(sum(vel*orientation))
                  vmeanabs2((i/nmeas)+1,1:D) = vmeanabs2((i/nmeas)+1,1:D) + vel**2
                  vmeanabs2((i/nmeas)+1,D+1) = vmeanabs2((i/nmeas)+1,D+1)   + sum(vel*orientation)**2
            
                  phimean((i/nmeas)+1)     = phimean((i/nmeas)+1)  + phi
                  phimean2((i/nmeas)+1)    = phimean2((i/nmeas)+1) + phi**2

                  thetamean((i/nmeas)+1)     = thetamean((i/nmeas)+1)  + theta
                  thetamean2((i/nmeas)+1)    = thetamean2((i/nmeas)+1) + theta**2

                  !Distributions at t=totaltime/2
                  if(i==int(Nt/2)) then
                        phidist(floor((phi+PI)/dphi)+1,2) = phidist(floor((phi+PI)/dphi)+1,2) + 1
                        thetadist(floor((cos(theta)+1.d0)/dtheta)+1,2) = thetadist(floor((cos(theta)+1.d0)/dtheta)+1,2) + 1
                        angdisp = sum(orientation*orientationi)
                        if(abs(angdisp-1.d0)<2.d0*epsilon(angdisp)) angdisp = sign(1.d0,angdisp)
                        angdispdist(floor(acos(angdisp)/dangdisp)+1,2) = &
                        angdispdist(floor(acos(angdisp)/dangdisp)+1,2) + 1
                  else if(i==Nt) then !Distributions at t=totaltime
                        phidist(floor((phi+PI)/dphi)+1,3) = phidist(floor((phi+PI)/dphi)+1,3) + 1
                        thetadist(floor((cos(theta)+1.d0)/dtheta)+1,3) = thetadist(floor((cos(theta)+1.d0)/dtheta)+1,3) + 1
                        angdisp = sum(orientation*orientationi)
                        if(abs(angdisp-1.d0)<2.d0*epsilon(angdisp)) angdisp = sign(1.d0,angdisp)
                        angdispdist(floor(acos(angdisp)/dangdisp)+1,3) = &
                        angdispdist(floor(acos(angdisp)/dangdisp)+1,3) + 1
                  end if
                  !Distributions all time
                  phidist(floor((phi+PI)/dphi)+1,4) = phidist(floor((phi+PI)/dphi)+1,4) + 1
                  thetadist(floor((cos(theta)+1.d0)/dtheta)+1,4) = thetadist(floor((cos(theta)+1.d0)/dtheta)+1,4) + 1
                  angdispdist(floor(acos(angdisp)/dangdisp)+1,4) = &
                  angdispdist(floor(acos(angdisp)/dangdisp)+1,4) + 1

                  globalvmean = globalvmean + sum(vel*orientation)
                  globalvmean2 = globalvmean2 + sum(vel*orientation)**2
      
                  globalvnoprojmean = globalvnoprojmean + sqrt(sum(vel*vel))
                  globalvnoprojmean2 = globalvnoprojmean2 + sum(vel*vel)
            end if
            call cpu_time(stop_time)
            acu_time = acu_time + stop_time-start_time
      end do
      if(Nsamples==1) write(201,"(2(E20.13,2X))")xyz
      if(Nsamples==1) close(201)
      call cpu_time(start_time)
      !do i=1,D
      !      call MSD_with_lags(1,Nt+1,cm_positions(:,i),nmeas,aux_lags)
      !      MSD_lags(:,i) = MSD_lags(:,i) + aux_lags
      !      MSD_lags2(:,i) = MSD_lags2(:,i) + aux_lags**2
      !end do
      call MSD_with_lags(D,Nt+1,cm_positions(:,:),nmeas,aux_lags)
      MSD_lags(:,D+1) = MSD_lags(:,D+1) + aux_lags
      MSD_lags2(:,D+1) = MSD_lags2(:,D+1) + aux_lags**2

      call MASD_with_lags(D,Nt+1,orientations(:,:),nmeas,aux_lags)
      MASD_lags  = MASD_lags  + aux_lags
      MASD_lags2 = MASD_lags2 + aux_lags**2

      call cpu_time(stop_time)
      acu_time = acu_time + stop_time-start_time
end do

call cpu_time(start_time)
do i=0,(Nt/nmeas)
      do j=1,D+1
            call naive_sampling_acu(Nsamples,MSD(i+1,j),MSD2(i+1,j))
            call naive_sampling_acu(Nsamples,vmean(i+1,j),vmean2(i+1,j))
            call naive_sampling_acu(Nsamples,vmeanabs(i+1,j),vmeanabs2(i+1,j))
      end do
      call naive_sampling_acu(Nsamples,phimean(i+1),phimean2(i+1))
      call naive_sampling_acu(Nsamples,thetamean(i+1),thetamean2(i+1))
end do

do i=0,(Nt/nmeas)-1
      call naive_sampling_acu(Nsamples,MASD_lags(i+1),MASD_lags2(i+1))
      do j=1,D+1
            call naive_sampling_acu(Nsamples,MSD_lags(i+1,j),MSD_lags2(i+1,j))
      end do
end do

open(30,file=trim(output_path)//"MSD.dat")
open(32,file=trim(output_path)//"vel.dat")
open(45,file=trim(output_path)//"velabs.dat")
open(33,file=trim(output_path)//"phi.dat")
open(35,file=trim(output_path)//"theta.dat")
open(36,file=trim(output_path)//"MSD_lags.dat")
open(37,file=trim(output_path)//"/MASD_lags.dat")
open(42,file=trim(output_path)//"/phidist.dat")
open(43,file=trim(output_path)//"/thetadist.dat")
open(44,file=trim(output_path)//"/angdispdist.dat")

open(101,file=trim(output_path)//"log.txt")

do i=0,(Nt/nmeas)
      write(30,"(E20.13,2X,8(E20.13,2X))")i*deltat*nmeas,MSD(i+1,:),MSD2(i+1,:)
      write(32,"(E20.13,2X,8(E20.13,2X))")i*deltat*nmeas,vmean(i+1,:),vmean2(i+1,:)
      write(45,"(E20.13,2X,8(E20.13,2X))")i*deltat*nmeas,vmeanabs(i+1,:),vmeanabs2(i+1,:)
      write(33,"(E20.13,2X,2(E20.13,2X))")i*deltat*nmeas,phimean(i+1),phimean2(i+1)
      write(35,"(E20.13,2X,2(E20.13,2X))")i*deltat*nmeas,thetamean(i+1),thetamean2(i+1)
end do
do i=0,(Nt/nmeas)-1
      write(36,"(E20.13,2X,8(E20.13,2X))")(i+1)*deltat*nmeas,MSD_lags(i+1,:),MSD_lags2(i+1,:)
      write(37,"(E20.13,2X,2(E20.13,2X))")(i+1)*deltat*nmeas,MASD_lags(i+1),MASD_lags2(i+1)
end do
do i=1,Nbins
      write(42,"(E20.13,2X,4(F25.15,2X))")i*dphi-PI-dphi/2.d0,phidist(i,1)/sum(phidist(:,1)*dphi),&
      phidist(i,2)/sum(phidist(:,2)*dphi),phidist(i,3)/sum(phidist(:,3)*dphi),phidist(i,4)/sum(phidist(:,4)*dphi)
      write(43,"(E20.13,2X,4(F25.15,2X))")i*dtheta-1.d0-dtheta/2.d0,thetadist(i,1)/sum(thetadist(:,1)*dtheta),&
      thetadist(i,2)/sum(thetadist(:,2)*dtheta),thetadist(i,3)/sum(thetadist(:,3)*dtheta),thetadist(i,4)/sum(thetadist(:,4)*dtheta)
      write(44,"(E20.13,2X,4(F25.15,2X))")i*dangdisp-dangdisp/2.d0,angdispdist(i,1)/sum(angdispdist(:,1)*dangdisp),&
      angdispdist(i,2)/sum(angdispdist(:,2)*dangdisp),&
      angdispdist(i,3)/sum(angdispdist(:,3)*dangdisp),&
      angdispdist(i,4)/sum(angdispdist(:,4)*dangdisp)
end do

call naive_sampling_acu(Nsamples*(Nt/nmeas),globalvmean,globalvmean2)
call naive_sampling_acu(Nsamples*(Nt/nmeas),globalvnoprojmean,globalvnoprojmean2)

call cpu_time(global_stop)
call cpu_time(stop_time)
stat_time = stat_time + stop_time-start_time

print*,"Mean velocity in time: ",sum(vmean(:,D+1))/(Nt/nmeas),sum(vmean2(:,D+1))/(Nt/nmeas)
print*,"Mean velocity in time (direct): ",globalvmean,globalvmean2
print*,"Mean velocity in time (direct not projected): ",globalvnoprojmean,globalvnoprojmean2
print*,"Total wall time (s):          ",global_stop-global_start
print*,"Time spent on integration (s):",integration_time
print*,"Time spent on acumulation (s):",acu_time
print*,"Time spent on statistics  (s):",stat_time
print*,"Time per sample (s):          ",(global_stop-global_start)/Nsamples


write(101,*)"Mean velocity in time: ",sum(vmean(:,D+1))/(Nt/nmeas),sum(vmean2(:,D+1))/(Nt/nmeas)
write(101,*)"Mean velocity in time (direct): ",globalvmean,globalvmean2
write(101,*)"Mean velocity in time (direct not projected): ",globalvnoprojmean,globalvnoprojmean2
write(101,*)"Total wall time (s):          ",global_stop-global_start
write(101,*)"Time spent on integration (s):",integration_time
write(101,*)"Time spent on acumulation (s):",acu_time
write(101,*)"Time spent on statistics  (s):",stat_time
write(101,*)"Time per sample (s):          ",(global_stop-global_start)/Nsamples

write(101,*)"Parameters used:"
write(101,*)"D        = ", D
write(101,*)"dt (ps)  = ", deltat
write(101,*)"Nt       = ", Nt
write(101,*)"nmeas    = ", nmeas
write(101,*)"Nsamples = ", Nsamples
write(101,*)"Mode     = ", v_mode
do i=1,Npar
      write(101,*)"par      = ", i,par(i)
end do
if(v_mode=="EXPPOISSON") write(101,*)"sequence = ", aux_random_sequence
if(v_mode=="EXPPMRND") write(101,*)"sequence = ", aux_random_sequence
write(101,*)"seed      = ", seed
if(D==1) then
      write(101,*)"DT       = ",DT
else
      write(101,*)"DT       = ",DT
      write(101,*)"DR       = ",DR
end if
contains

real*8 function constant_speed(t) result(v)
      implicit none
      real*8 :: t
      v = par(1)
end function constant_speed

real*8 function periodic_gaussian(t) result(v)
      implicit none
      real*8 :: t,teff,td,sigma,vder,vexc,freq
      freq = par(2)
      teff = mod(t,freq)
      
      td = 0.1d0
      sigma = 0.025d0
      vder = 0.d0
      vexc = par(1)
      v = (vexc-vder)*exp(-(teff-td)**2/(2.d0*sigma**2)) + vder
end function periodic_gaussian

real*8 function periodic_linexp(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,timewidth,vder,vexc,freq
      freq = par(4)
      teff = mod(t,freq)

      tdec = par(2)
      timewidth = par(3)
      vder = 0.d0
      vexc = par(1)
      if(teff <= timewidth) then
            v = (vexc/timewidth)*teff + vder
      else
            v = vexc*exp(-(teff-timewidth)/tdec) + vder
      end if
end function periodic_linexp

real*8 function periodic_pw(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq
      freq = par(3)
      teff = mod(t,freq)

      tdec = par(2)
      vexc = par(1)
      if(teff<tdec) then
            v = vexc
      else
            v = 0.d0
      end if
end function periodic_pw

real*8 function periodic_exp(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq
      freq = par(3)
      teff = mod(t,freq)

      tdec = par(2)
      vder = 0.d0
      vexc = par(1)
      v = vexc*exp(-teff/tdec) + vder
end function periodic_exp

real*8 function periodic_exp_pm(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq
      integer :: sign
      freq = par(3)
      teff = mod(t,freq)

      tdec = par(2)
      vder = 0.d0
      vexc = par(1)
      if(mod(int(t/freq),2)==0) then 
            sign = +1
      else
            sign = -1
      end if
      v = sign*vexc*exp(-teff/tdec) + vder
end function periodic_exp_pm

real*8 function periodic_exp_pmrnd(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq
      freq = par(3)
      teff = mod(t,freq)

      tdec = par(2)
      vder = 0.d0
      vexc = par(1)
      v = aux_random_sequence(int(t/freq)+1)*vexc*exp(-teff/tdec) + vder
end function periodic_exp_pmrnd

real*8 function periodic_exp_rnd(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq
      freq = par(3)
      teff = mod(t+freq*aux_random1,freq)

      tdec = par(2)
      vder = 0.d0
      vexc = par(1)
      v = vexc*exp(-teff/tdec) + vder
end function periodic_exp_rnd

real*8 function periodic_exp_poisson(t) result(v)
      implicit none
      real*8 :: t,teff,tdec,vder,vexc,freq

      !If we have passed next actuation move seqindex to the next point
      if(seqindex /= int(par(3))) then
            if(i >= aux_random_sequence(seqindex+1)) seqindex = seqindex + 1
      end if
      if(seqindex>0) then
            !Simulate last actuation
            teff = t - aux_random_sequence(seqindex)*deltat

            tdec = par(2)
            vder = 0.d0
            vexc = par(1)
            v = vexc*exp(-teff/tdec) + vder
      else
            !Return 0 since no actuation has happened yet  
            v = 0.d0 
      end if

end function periodic_exp_poisson

real*8 function sine(t) result(v)
      implicit none
      real*8 :: t,vder,v0,freq

      vder = 0.d0
      v0   = par(1)
      freq = par(2)
      v = v0*cos(2.d0*PI*freq*t) + vder
end function sine

real*8 function abssine(t) result(v)
      implicit none
      real*8 :: t,vder,v0,freq

      vder = 0.d0
      v0   = par(1)
      freq = par(2)
      v = v0*abs(cos(2.d0*PI*freq*t)) + vder
end function abssine

real*8 function sine_rnd(t) result(v)
      implicit none
      real*8 :: t,vder,phase,v0,freq

      vder  = 0.d0
      phase = PI*aux_random1
      v0    = par(1)
      freq  = par(2)
      v = v0*cos(2.d0*PI*freq*t + phase) + vder
end function sine_rnd

real*8 function harm(t) result(v)
      implicit none
      real*8 :: t,vder,v0,freq

      v0   = par(1)
      freq = par(2)
      v = 0.5*v0*(1.d0 + cos(2.d0*PI*freq*t))
end function harm


subroutine integration_step(t,deltat,D,xyz,phi,theta,v,DR,DT)
      implicit none
      real*8,intent(in)    :: t,deltat,DR,DT
      integer,intent(in)   :: D
      real*8,intent(inout) :: xyz(D),phi,theta
      real*8,external      :: v
      real*8               :: nvector(3),Rvector(3)
      real*8               :: g1,g2,g3,g4,g5,g6
      select case(D)
            case(1)
                  xyz(1) = xyz(1) + v(t)*deltat + sqrt(2.d0*DT*deltat)*gaussian_number()
            case(2)
                  call gaussian_number_pair(g1,g2)
                  xyz(1) = xyz(1) + v(t)*cos(phi)*deltat + sqrt(2.d0*DT*deltat)*g1
                  xyz(2) = xyz(2) + v(t)*sin(phi)*deltat + sqrt(2.d0*DT*deltat)*g2
                  phi    = phi    + sqrt(2.d0*DR*deltat)*gaussian_number()
            case(3)
                  call gaussian_number_pair(g1,g2)
                  call gaussian_number_pair(g3,g4)
                  call gaussian_number_pair(g5,g6)
                  nvector = (/cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)/)
                  xyz(1) = xyz(1) + v(t)*nvector(1)*deltat + sqrt(2.d0*DT*deltat)*g1
                  xyz(2) = xyz(2) + v(t)*nvector(2)*deltat + sqrt(2.d0*DT*deltat)*g2
                  xyz(3) = xyz(3) + v(t)*nvector(3)*deltat + sqrt(2.d0*DT*deltat)*g3

                  Rvector = (/g4,g5,g6/)
                  nvector = nvector + sqrt(2.d0*DR*deltat)*cross_product(nvector,Rvector)
                  nvector = nvector/sqrt(sum(nvector**2))
                  call get_phi_theta_from_n(nvector,phi,theta)
      end select
end subroutine integration_step

real*8 function gaussian_number(std_arg) result(x)
      implicit none
      real*8,optional :: std_arg
      real*8          :: std
      if(present(std_arg)) then
            std = std_arg
      else
            std = 1.d0
      end if
      x = 1d20
      do while(isnan(x) .or. abs(x)>1.d10)
            x = std*dsqrt(-2d0*dlog(1.d0-grnd()))*dcos(2d0*PI*grnd())
            ! call random_number(rnd1)
            ! call random_number(rnd2)
            ! x = std*dsqrt(-2d0*dlog(1.d0-rnd1))*dcos(2d0*PI*rnd2)
      end do
end function gaussian_number

subroutine gaussian_number_pair(x,y,std_arg)
      implicit none
      real*8,intent(out) :: x,y
      real*8,optional    :: std_arg
      real*8             :: std,a,b,s
      if(present(std_arg)) then
            std = std_arg
      else
            std = 1.d0
      end if
      x = 1d20
      do while(isnan(x) .or. abs(x)>1.d10)
            a = grnd()
            b = grnd()
            ! call random_number(a)
            ! call random_number(b)
            s = dsqrt(-2d0*dlog(1.d0-a))
            x = std*s*dcos(2d0*PI*b)
      end do
      y = std*s*dsin(2d0*PI*b)
end subroutine gaussian_number_pair

function cross_product(v1,v2) result(v3)
      implicit none
      real*8 :: v1(3),v2(3)
      real*8 :: v3(3)

      v3(1) =   v1(2)*v2(3) - v1(3)*v2(2)
      v3(2) = - v1(1)*v2(3) + v1(3)*v2(1)
      v3(3) =   v1(1)*v2(2) - v1(2)*v2(1)
end function cross_product

subroutine get_phi_theta_from_n(n,phi,theta)
      implicit none
      real*8,intent(in)  :: n(3)
      real*8,intent(out) :: phi,theta
      ! phi = acos(n(1)/sqrt(n(1)**2+n(2)**2))
      phi = atan2(n(2),n(1))
      ! if(isnan(phi)) then
      !       if(n(2)>0.d0) then
      !             phi = PI/2.d0
      !       elseif(n(2)<0.d0) then
      !             phi = PI/2.d0
      !       else
      !             phi = 0.d0
      !       end if
      ! ! else
      ! !       phi = sign(phi,n(2))
      ! end if
      theta = acos(n(3))
end subroutine get_phi_theta_from_n

!----------------Post processing subroutines----------------
subroutine naive_sampling(N,data_array,data_mean,data_sigm,alt_N)
      implicit none
      integer,intent(in) :: N
      real*8,intent(in)  :: data_array(N)
      real*8,intent(out) :: data_mean,data_sigm
      integer,optional   :: alt_N
      integer            :: i

      data_mean = 0.d0
      data_sigm = 0.d0
      do i=1,N
            data_mean = data_mean + data_array(i)
            data_sigm = data_sigm + data_array(i)**2
      end do

      if(present(alt_N)) then
            data_mean = data_mean/alt_N
            data_sigm = data_sigm/alt_N
            if(alt_N>2) then
                  data_sigm = sqrt(data_sigm-data_mean**2)/sqrt(alt_N-1.d0)
            else
                  data_sigm = 0.d0
            end if
      else
            data_mean = data_mean/N
            data_sigm = data_sigm/N
            if(N>2) then
                  data_sigm = sqrt(data_sigm-data_mean**2)/sqrt(N-1.d0)
            else
                  data_sigm = 0.d0
            end if
      end if
end subroutine naive_sampling

subroutine naive_sampling_acu(N,data_mean,data_sigm,alt_N)
      implicit none
      integer,intent(in) :: N
      real*8,intent(inout) :: data_mean,data_sigm
      integer,optional   :: alt_N
      integer            :: i

      if(present(alt_N)) then
            data_mean = data_mean/alt_N
            data_sigm = data_sigm/alt_N
            if(alt_N>2) then
                  data_sigm = sqrt(data_sigm-data_mean**2)/sqrt(alt_N-1.d0)
            else
                  data_sigm = 0.d0
            end if
      else
            data_mean = data_mean/N
            data_sigm = data_sigm/N
            if(N>2) then
                  data_sigm = sqrt(data_sigm-data_mean**2)/sqrt(N-1.d0)
            else
                  data_sigm = 0.d0
            end if
      end if
end subroutine naive_sampling_acu

real*8 function trapezoid_integrate(N,array,dx) result(I)
      implicit none
      integer :: N
      real*8  :: array(N),dx

      I = 0.d0
      if(N>2) then
            I = sum(array(2:N-1))*dx + (array(1)+array(N))*dx/2.d0 !Integral for the rest
      else if(N==1) then 
            I = array(1)*dx/2.d0 !Integral on the first 'half' step
      else if(N==2) then 
            I = (array(1)+array(2))*dx/2.d0 !Integral on the first 'full' step
      end if
end function trapezoid_integrate

subroutine MSD_with_lags(D,N,x,meas,MSD)
      implicit none
      integer,intent(in) :: D,N,meas
      real*8,intent(in)  :: x(N,D)
      real*8,intent(out) :: MSD((N-1)/meas)
      integer :: i,j,k
      MSD = 0.d0
      do i=1,(N-1)/meas
            k = i*meas
            do j=1,N-k
                  MSD(i) = MSD(i) + sum((x(j+k,:)-x(j,:))**2)
            end do
            MSD(i) = MSD(i)/(N-k)
      end do
end subroutine MSD_with_lags

subroutine MASD_with_lags(D,N,x,meas,MASD)
      implicit none
      integer,intent(in) :: N,D,meas
      real*8,intent(in)  :: x(N,D)
      real*8,intent(out) :: MASD((N-1)/meas)
      integer            :: i,j,k
      MASD = 0.d0
      do i=1,(N-1)/meas
            k = i*meas
            do j=1,N-k
                  ! MASD(i) = MASD(i) + acos(sum(x(i+j,:)*x(j,:)))**2
                  MASD(i) = MASD(i) + sum(x(j+k,:)*x(j,:))
            end do
            MASD(i) = MASD(i)/(N-k)
      end do
end subroutine MASD_with_lags

subroutine ACF_with_lags(D,N,x,ACF)
      implicit none
      integer,intent(in) :: N,D
      real*8,intent(in)  :: x(N,D)
      real*8,intent(out) :: ACF(N)
      integer            :: i,j
      ACF = 0.d0
      do i=1,N-1
            do j=1,N-i
                  ACF(i+1) = ACF(i+1) + sum(x(i+j,:)*x(j,:))
            end do
            ACF(i+1) = ACF(i+1)/(N-i)
      end do
      do i=1,N
            ACF(1) = ACF(1) + sum(x(i,:)*x(i,:))
      end do
      ACF(1) = ACF(1)/N
end subroutine ACF_with_lags

end program ABSIM
