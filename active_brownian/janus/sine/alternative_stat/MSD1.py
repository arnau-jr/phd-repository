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

#7-Class printer friendly from color brewer
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#e41a1c", "#377eb8","#4daf4a","#984ea3","#ff7f00","#ffff33","#a65628"]) 

def msd_theory_ct(t, DT = 0.1, DR = 0.1, v = 1.0):
    return 4*(DT+0.5*v**2/DR)*t + (2*v**2/DR**2)*(np.exp(-DR*t)-1)

def msd_theory(t, DT, DR, v, T):
    w = 2*np.pi/T
    taur = 1/DR

    term1 = 4*(DT+(v**2*taur)/(4*(1+(taur*w)**2)))*t
    term2 = (v**2*(w**2-taur**-2))/(w**2+taur**-2)**2
    preterm3 = (v**2*np.exp(-t/taur))/(w**2+taur**-2)**2
    term3 = (w**2-taur**-2)*np.cos(w*t) + 2*(w/taur)*np.sin(w*t)

    return term1 + term2 -preterm3*term3

DT = 0.22
DR = 0.16

v = 10.0
T = 1.0

Pe2 = (v*2/np.pi)**2/2/DT/DR

Nsamples = 10

Deff = []
Derr = []
data_msd = np.loadtxt('./result2Dsine1_1/MSD_lags.dat')
data_msdavg = np.zeros([data_msd.shape[0],3])
for sample in range(1,Nsamples+1):
    data_msd = np.loadtxt(f'./result2Dsine1_{sample}/MSD_lags.dat')

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
print(f'Defftheory            = {DT*(1+Pe2*f_sine_a(T*DR)):.6f}')

data_msdavg /= Nsamples
data_msdavg[:,2] = np.sqrt(data_msdavg[:,2]-data_msdavg[:,1]**2)/np.sqrt(Nsamples-1)

np.save('MSD1_data',data_msdavg)

plt.figure()
plt.xlim([0,data_msdavg[-1,0]])
# plt.xlim([0,300])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
freq = 75
plt.plot(data_msdavg[:,0],data_msdavg[:,1],color="#e41a1c",linestyle='solid',label=r'Simulation')
plt.fill_between(data_msdavg[:,0], data_msdavg[:,1]-data_msdavg[:,2], data_msdavg[:,1]+data_msdavg[:,2],color="#e41a1c",alpha=0.25)
plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,T),color="#377eb8",linestyle='dashed',label=r'Theory')
plt.legend()
plt.savefig('plots/MSD1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
plt.close()

plt.figure()
plt.xlim([0,30])
# plt.ylim([0,10.0])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
ulim = int(30/dt)
freq = 10
t_sample = np.linspace(0,30,1000)
plt.plot(data_msdavg[:ulim,0],data_msdavg[:ulim,1],color="#e41a1c",linestyle='solid',label=r'Simulation')
plt.fill_between(data_msdavg[:ulim,0], data_msdavg[:ulim,1]-data_msdavg[:ulim,2], data_msdavg[:ulim,1]+data_msdavg[:ulim,2],color="#e41a1c",alpha=0.25)
plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,T),color="#377eb8",linestyle='dashed',label=r'Theory')
# plt.errorbar(data_msdavg[:ulim:freq,0],data_msdavg[:ulim:freq,1],data_msdavg[:ulim:freq,2],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1_short.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
