import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def vacf_with_lags(v,N=None,Nmaxlag=None):
    if(N is None): N = v.shape[0]
    if(Nmaxlag is None): Nmaxlag=N
    vacf = np.zeros(Nmaxlag)

    for i in tqdm(range(1,Nmaxlag)):
        for j in range(1,N-i+1):
            vacf[i] += np.sum(v[i+j-1,:]*v[j-1,:])
        vacf[i] /= N-i
    
    vacf[0] += np.sum(v[:N]*v[:N])
    vacf[0] /= N

    return vacf/vacf[0]

def vacf_with_lags_improved(v,N=None,Nmaxlag=None):
    if(N is None): N = v.shape[0]
    if(Nmaxlag is None): Nmaxlag=N
    vacf = np.zeros(Nmaxlag)

    for i in tqdm(range(0,Nmaxlag)):
        vacf[i]  = np.sum(v[i:N,:]*np.roll(v,i,axis=0)[i:N,:])
        vacf[i] /= N-i

    return vacf/vacf[0]

def vacf_with_lags_improved_1D(v,N=None,Nmaxlag=None):
    if(N is None): N = v.shape[0]
    if(Nmaxlag is None): Nmaxlag=N
    vacf = np.zeros(Nmaxlag)

    for i in tqdm(range(0,Nmaxlag)):
        vacf[i]  = np.sum(v[i:N]*np.roll(v,i,axis=0)[i:N])
        vacf[i] /= N-i

    return vacf/vacf[0]

# with open(f'../short_nitro_spce/0runs_ex2/run{sample:05d}/out_central.dat','r') as f:
#     lines = f.readlines()

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
    v[i,:] = np.sum(mass[:,np.newaxis]*vatom,axis=0)/np.sum(mass)

    CNvec = xatom[1,:]-xatom[0,:]
    CNvec = CNvec/np.sqrt(np.sum(CNvec**2))
    vp[i] = np.sum((np.sum(mass[:,np.newaxis]*vatom,axis=0)/np.sum(mass))*CNvec)


maxframes=Nframes
maxlag=20001

vacf = vacf_with_lags_improved(v,maxframes,maxlag)
vacf_proj = vacf_with_lags_improved_1D(vp,maxframes,maxlag)
# vacfbis = vacf_with_lags_improved(v,maxframes,maxlag)
times = np.arange(0,maxlag)*dt/1000.0 #in ps

with open('vacf.dat','w') as f:
    f.write(f'Time(ps)                 VACF(norm.)   Averaged over {maxframes*dt/1000.0} ps\n')
    for i in range(times.shape[0]):
        f.write(f"{times[i]:21.14E}    {vacf[i]:12.14E}\n")

with open('vacfproj.dat','w') as f:
    f.write(f'Time(ps)                 VACF(norm.)   Averaged over {maxframes*dt/1000.0} ps\n')
    for i in range(times.shape[0]):
        f.write(f"{times[i]:21.14E}    {vacf_proj[i]:12.14E}\n")



plt.figure()
plt.xlabel('t (ps)')
plt.ylabel('VACF')
plt.xlim([0,times[-1]])
plt.plot(times,vacf,label='Non Projected')
plt.plot(times,vacf_proj,label='Projected')
# plt.plot(times,vacfbis)
plt.legend()
plt.savefig('vacf.pdf')