import numpy as np
import matplotlib.pyplot as plt
import sys

import matplotlib as mpl
mpl.rcParams['text.usetex']         = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{upgreek}'
mpl.rcParams['figure.figsize']      = (3.375,2.6)
mpl.rcParams['font.size']           = 10.0
mpl.rcParams['font.family']         = 'serif'
mpl.rcParams['font.serif']          = 'STIX' 
mpl.rcParams["mathtext.fontset"]    = "stix"

from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.insert(1, '/home/ajurado/synched_files/active_brownian/formula_comparison')
from formulas import *

def msd_theory_ct(t, DT = 0.1, DR = 0.1, v = 1.0):
    return 4*(DT+0.5*v**2/DR)*t + (2*v**2/DR**2)*(np.exp(-DR*t)-1)

def msd_theory(t, DT = 0.1, DR = 0.1, v = 1.0, T = 5.0, N=int(1e5)):
    taur = 1/DR
    tauT = tau/T
    w = 2*np.pi/T

    vmean = v*tauT*(1.0-np.exp(-1/tauT))
    
    acu1 = 0.0
    acu2 = 0.0
    for n in range(1,N+1):
        acu1 += (1/(1+(n*w*tau)**2))*(1/(1+(n*w*taur)**2))
        # acu2 += (np.exp(-t/taur)/((1+(n*w*tau)**2)*(1+(n*w*taur)**2)**2))\
        #     *((1-(n*w*taur)**2)*(np.cos(n*w*t)-1)-2*n*w*taur*np.sin(n*w*t))
        acu2 += (np.exp(-t/taur)/((1+(n*w*tau)**2)*(1+(n*w*taur)**2)**2))\
            *((1-(n*w*taur)**2)*(np.cos(n*w*t)-np.exp(t/taur))-2*n*w*taur*np.sin(n*w*t))

    return 4*( DT + 0.5*taur*vmean**2 + taur*vmean**2*acu1 )*t +\
        2*taur**2*vmean**2*(np.exp(-t/taur)-1) +\
        4*taur**2*vmean**2*acu2

DT = 0.22
DR = 0.16

v = 10.0
tau = 0.1
T = 1.0
vmean = v/(T/tau)*(1.0-np.exp(-T/tau))
Pe2 = vmean**2/(2*DT*DR)

Nsamples = 10

Deff = []
Derr = []
data_msd = np.loadtxt('./result2Dexp1_1/MSD_lags.dat')
data_msdavg = np.zeros([data_msd.shape[0],3])
for sample in range(1,Nsamples+1):
    data_msd = np.loadtxt(f'./result2Dexp1_{sample}/MSD_lags.dat')

    data_msdavg[:,0] += data_msd[:,0]
    data_msdavg[:,1] += data_msd[:,3]
    data_msdavg[:,2] += data_msd[:,3]**2

    print(f'{sample}:')
    print(f'Got                 {data_msd.shape[0]} data points')

    t_sample = np.linspace(0,data_msd[-1,0],1000)

    dt = data_msd[1,0]-data_msd[0,0]
    llim = int(20/dt)
    ulim = int(40/dt)
    print(f'Performing fit with {data_msd[llim:ulim,:].shape[0]} data points')
    # Dalt,Salt = fit_D(data_msd[llim:ulim,:])

    p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3],sigma=data_msd[llim:ulim,6])
    m,b = p[0],p[1]

    p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]+data_msd[llim:ulim,6])
    mu,bu = p[0],p[1]
    p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]-data_msd[llim:ulim,6])
    ml,bl = p[0],p[1]
    merr = np.abs(mu-ml)/2.

    Deff.append(m/4)
    Derr.append(merr/4)

    if(sample==0):
        plt.figure()
        plt.xlim([0,data_msd[-1,0]])
        plt.xlabel(r't ($\mathrm{s}$)')
        plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
        freq = 250
        plt.errorbar(data_msd[::freq,0],data_msd[::freq,3],data_msd[::freq,6],fmt='rx',label='Data',capsize=2)
        plt.fill_between(t_sample, t_sample*(m-merr), t_sample*(m+merr),alpha=0.25)
        plt.legend()
        # plt.savefig('plots/MSD2.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
        plt.show()

    print(f'Deff              = {m/4:.6f} +- {merr/4:.6f}')
    print('----------------------------------------------------------------------')

Deff = np.array(Deff)
Derr = np.array(Derr)
print(f'Deff                  = {Deff.mean():.6f} +- {Deff.std(ddof=1):.6f}')
print(f'Deff(mean of errors)  = {Deff.mean():.6f} +- {Derr.mean():.6f}')
print(f'Defftheoryfourier     = {(DT+Dex_exponential_f(v,T,tau,1/DR,N=int(1e6))):.6f}')

data_msdavg /= Nsamples
data_msdavg[:,2] = np.sqrt(data_msdavg[:,2]-data_msdavg[:,1]**2)/np.sqrt(Nsamples-1)

np.save('MSD1_data',data_msdavg)

plt.figure()
plt.xlim([0,data_msdavg[-1,0]])
# plt.ylim([0,250])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
freq = 200
plt.plot(t_sample,4*(DT+DT*Pe2*f_exponential_f(T/tau,T*DR,N=int(1e6)))*t_sample,'g-',label=r'Theo: $D_\mathrm{eff}t$')
plt.plot(t_sample,4*Deff.mean()*t_sample,'b-',label=r'$D_\mathrm{eff}t$')
plt.errorbar(data_msdavg[::freq,0],data_msdavg[::freq,1],data_msdavg[::freq,2],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
plt.close()

plt.figure()
plt.xlim([0,0.2])
plt.ylim([0,0.4])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
# plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,T),'g-',label=r'Theory')
plt.errorbar(data_msdavg[:,0],data_msdavg[:,1],data_msdavg[:,2],fmt='rx',label='Data',capsize=2)
plt.plot(t_sample,4*(DT+DT*Pe2*f_exponential_f(T/tau,T*DR,N=int(1e6)))*t_sample,'g-',label=r'Theo: $D_\mathrm{eff}t$')
plt.plot(t_sample,4*Deff.mean()*t_sample,'b-',label=r'$D_\mathrm{eff}t$')
plt.legend()
plt.savefig('plots/MSD1_short.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)

plt.figure()
plt.xlim([0,data_msdavg[-1,0]])
plt.ylim([10.0,20.0])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'dMSD/dt ($\upmu\mathrm{m}^2/\mathrm{s}$)')
freq = 150
plt.plot(t_sample,4*(DT+DT*Pe2*f_exponential_f(T/tau,T*DR,N=int(1e6)))*np.ones_like(t_sample),'g-',label=r'Theo: $D_\mathrm{eff}t$')
plt.plot(t_sample,4*Deff.mean()*np.ones_like(t_sample),'b-',label=r'$D_\mathrm{eff}t$')
# plt.plot(data_msdavg[::freq,0],np.gradient(data_msdavg[:,1],data_msdavg[:,0])[::freq],'rx',label='Data')
freq = 1
plt.plot(data_msdavg[::freq,0],np.gradient(data_msdavg[:,1],data_msdavg[:,0])[::freq],'r-',label='Data')
plt.vlines(list(range(1,41)),plt.ylim()[0],plt.ylim()[1],linestyles='dashed',colors='black',linewidths=0.75,alpha=0.25)
plt.legend()
plt.savefig('plots/DER1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
plt.close()
