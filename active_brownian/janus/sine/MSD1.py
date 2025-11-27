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

def fit_D(data):
    n = [125,250,500,1000]
    Dn,Sn = [],[]
    for l in n:
        Ds = []
        for i in range(data.shape[0]-l):
            p,cov = curve_fit(lambda x,m,b:m*x+b,data[i:i+l,0],data[i:i+l,3],sigma=data[i:i+l,6])
            Ds.append(p[0]/4)
        # for i in range(int(data.shape[0]/l)):
        #     p,cov = curve_fit(lambda x,m,b:m*x+b,data[i*l:l*(i+1),0],data[i*l:l*(i+1),3],sigma=data[i*l:l*(i+1),6])
        #     Ds.append(p[0]/4)
        Dn.append(np.array(Ds).mean())
        Sn.append(np.array(Ds).std(ddof=1))
    print(Dn)
    print(Sn)
    return Dn[-1],Sn[-1]



data_msd = np.loadtxt('./result2Dsine1/MSD_lags.dat')
data_vel = np.loadtxt('./result2Dsine1/vel.dat')

print(f'Got                 {data_msd.shape[0]} data points')

DT = 0.22
DR = 0.16

v = 10.0
T = 1.0

Pe2 = (v*2/np.pi)**2/2/DT/DR

t_sample = np.linspace(0,data_msd[-1,0],1000)

dt = data_msd[1,0]-data_msd[0,0]
llim = int(20/dt)
ulim = int(40/dt)
print(f'Performing fit with {data_msd[llim:ulim,:].shape[0]} data points')
Dalt,Salt = fit_D(data_msd[llim:ulim,:])

p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3],sigma=data_msd[llim:ulim,6])
# p,cov = curve_fit(lambda x,m:m*x,data_msd[llim:ulim,0],data_msd[llim:ulim,3],sigma=data_msd[llim:ulim,6])
# perr = np.sqrt(np.diag(cov))
m,b = p[0],p[1]
# m = p[0]
# merr,berr = perr[0],perr[1]

p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]+data_msd[llim:ulim,6])
mu,bu = p[0],p[1]
p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]-data_msd[llim:ulim,6])
ml,bl = p[0],p[1]
merr = np.abs(mu-ml)/2.

print(f'Deff              = {m/4:.6f} +- {merr/4:.6f}')
print(f'Deff(alt)         = {Dalt:.6f} +- {Salt:.6f}')
print(f'Defftheory        = {DT*(1+Pe2*f_sine_a(T*DR)):.6f}')


plt.figure()
plt.xlim([0,data_msd[-1,0]])
# plt.xlim([0,300])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
freq = 200
plt.errorbar(data_msd[::freq,0],data_msd[::freq,3],data_msd[::freq,6],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()

plt.figure()
plt.xlim([0,10])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'v ($\upmu\mathrm{m}/\mathrm{s}$)')
freq = 250
plt.plot(data_vel[:,0],data_vel[:,3],'g-')
plt.savefig('plots/vel1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
