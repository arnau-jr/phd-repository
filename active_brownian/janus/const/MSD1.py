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

sys.path.insert(1, '/home/ajurado/active_brownian/formula_comparison')
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

v = 1.0

Nsamples = 10

Deff = []
Derr = []
data_msd = np.loadtxt('./result2Dct1_1/MSD_lags.dat')
data_msdavg = np.zeros([data_msd.shape[0],3])
for sample in range(1,Nsamples+1):
    data_msd = np.loadtxt(f'./result2Dct1_{sample}/MSD_lags.dat')

    data_msdavg[:,0] += data_msd[:,0]
    data_msdavg[:,1] += data_msd[:,3]
    data_msdavg[:,2] += data_msd[:,3]**2

    print(f'{sample}:')
    print(f'Got                 {data_msd.shape[0]} data points')

    t_sample = np.linspace(0,data_msd[-1,0],1000)

    dt = data_msd[1,0]-data_msd[0,0]
    llim = int(30/dt)
    ulim = int(50/dt)
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
print(f'Defftheory            = {(DT+v**2/2/DR):.6f}')

data_msdavg /= Nsamples
data_msdavg[:,2] = np.sqrt(data_msdavg[:,2]-data_msdavg[:,1]**2)/np.sqrt(Nsamples-1)

np.save('MSD1_data',data_msdavg)

plt.figure()
plt.xlim([0,data_msdavg[-1,0]])
# plt.xlim([0,300])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
freq = 200
plt.plot(t_sample,msd_theory_ct(t_sample,DT,DR,v),'b-',label=r'Constant $v = \langle v\rangle$')
plt.errorbar(data_msdavg[::freq,0],data_msdavg[::freq,1],data_msdavg[::freq,2],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
plt.close()
