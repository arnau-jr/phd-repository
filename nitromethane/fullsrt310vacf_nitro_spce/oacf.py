import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import sys
#sys.path.insert(1, '/home/ajurado/inca/phd/codes/aux_vib')
#import auxvibmod as av

def compute_w_nitro(xnitro,vnitro):
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
    #out[5] -> omega shape 3
    return np.float64(out[5])

def acf_with_lags_improved(v,N=None,Nmaxlag=None):
    if(N is None): N = v.shape[0]
    if(Nmaxlag is None): Nmaxlag=N
    vacf = np.zeros(Nmaxlag)

    for i in tqdm(range(0,Nmaxlag)):
        vacf[i]  = np.sum(v[i:N,:]*np.roll(v,i,axis=0)[i:N,:])
        vacf[i] /= N-i

    return vacf

with open(f'out_nitro.dat','r') as f:
    lines = f.readlines()

Natoms = 7
Nskip = 9
Nframes = int(len(lines)/(Natoms+Nskip))
dt = 0.5 #Output rate

print(f'{Nframes} frames found, {Nframes*dt/1000.0} ps')

Nitrogenlines = lines[10::Natoms+Nskip]

v = np.zeros([Nframes,3])
vp = np.zeros([Nframes])
CNvecs = np.zeros([Nframes,3])
omegas = np.zeros([Nframes,3])
xatom = np.zeros([7,3])
vatom = np.zeros([7,3])
mass = np.array([12.01,14.01,15.99491502,15.99491502,1.00782522,1.00782522,1.00782522])

counter = 0
print('Processing frames')
for i in tqdm(range(Nframes)):
    counter += Nskip #Skip header
    for a in range(Natoms):
        l = lines[counter].split()
        xatom[a,:] = float(l[1]),float(l[2]),float(l[3])
        vatom[a,:] = float(l[4]),float(l[5]),float(l[6])
        counter += 1

    CNvec = xatom[1,:]-xatom[0,:]
    CNvec = CNvec/np.sqrt(np.sum(CNvec**2))
    CNvecs[i,:] = np.copy(CNvec)
#    omegas[i,:] = compute_w_nitro(xatom,vatom) 


maxframes=Nframes
maxlag=20001

oacf = acf_with_lags_improved(CNvecs,maxframes,maxlag)
#wacf = acf_with_lags_improved(omegas,maxframes,maxlag)
times = np.arange(0,maxlag)*dt/1000.0 #in ps

with open('oacf.dat','w') as f:
    f.write(f'Time(ps)                 OACF(norm.)   Averaged over {maxframes*dt/1000.0} ps\n')
    for i in range(times.shape[0]):
        f.write(f"{times[i]:21.14E}    {oacf[i]:12.14E}\n")

#with open('wacf.dat','w') as f:
#    f.write(f'Time(ps)                 WACF(norm.)   Averaged over {maxframes*dt/1000.0} ps\n')
#    for i in range(times.shape[0]):
#        f.write(f"{times[i]:21.14E}    {wacf[i]:12.14E}\n")



plt.figure()
plt.xlabel('t (ps)')
plt.ylabel('OACF')
plt.xlim([0,times[-1]])
plt.plot(times,oacf)
plt.savefig('oacf.pdf')

#plt.figure()
#plt.xlabel('t (ps)')
#plt.ylabel('WACF')
#plt.xlim([0,times[-1]])
#plt.plot(times,wacf)
#plt.savefig('wacf.pdf')
