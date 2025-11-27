import numpy as np
import matplotlib.pyplot as plt

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

def msd_theory_ct(t, DT = 0.1, DR = 0.1, v = 1.0):
    return 4*(DT+0.5*v**2/DR)*t + (2*v**2/DR**2)*(np.exp(-DR*t)-1)

def msd_theory(t, DT = 0.1, DR = 0.1, v = 1.0, f=0.1):
    w = 2*np.pi*f
    # return 4*(DT + DR*v**2/(4*(DR**2+w**2)))*t + \
    # v**2/(DR**2+w**2)**2*(w**2-DR**2) +\
    # v**2/(DR**2+w**2)**2*np.exp(-DR*t)*(DR**2*np.cos(w*t)-2*w*DR*np.sin(w*t)-w**2*np.cos(w*t))+\
    # v**2/(4*DR)*(1-np.exp(-DR*t))
    return 4*(DT + v**2/(8*DR) + DR*v**2/(16*(DR**2+w**2)))*t + \
    v**2/(DR**2+w**2)**2*(w**2-DR**2)/4 +\
    v**2/(DR**2+w**2)**2*np.exp(-DR*t)*(DR**2*np.cos(w*t)-2*w*DR*np.sin(w*t)-w**2*np.cos(w*t))/4 +\
    v**2/(2*DR**2)*(np.exp(-DR*t)-1)

def fit_D(data):
    n = [50,500,625,1250]
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

data_msd = np.loadtxt('./result2Dharm2/MSD_lags.dat')

print(f'Got                 {data_msd.shape[0]} data points')

DT = 0.22
DR = 0.16

v = 2.0
f = 0.01
w = 2*np.pi*f
vmean = v/2.

t_sample = np.linspace(0,data_msd[-1,0],1000)

dt = data_msd[1,0]-data_msd[0,0]
llim = int(50/dt)
ulim = int(300/dt)

print(f'Performing fit with {data_msd[llim:ulim,:].shape[0]} data points')
Dalt,Salt = fit_D(data_msd[llim:ulim,:])

p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3],sigma=data_msd[llim:ulim,6])
m,b = p[0],p[1]

p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]+data_msd[llim:ulim,6])
mu,bu = p[0],p[1]
p,cov = curve_fit(lambda x,m,b:m*x+b,data_msd[llim:ulim,0],data_msd[llim:ulim,3]-data_msd[llim:ulim,6])
ml,bl = p[0],p[1]
merr = np.abs(mu-ml)/2.

print(f'Deff              = {m/4:.6f} +- {merr/4:.6f}')
print(f'Deff(alt)         = {Dalt:.6f} +- {Salt:.6f}')
print(f'Defftheory        = {(DT +vmean**2/(2*DR)+ DR*vmean**2/(4*(DR**2+w**2))):.6f}')
print(f'Defftheory        = {(DT + v**2/(8*DR) + DR*v**2/(16*(DR**2+w**2))):.6f}')
print(f'Defftheorymean    = {(DT+(0.5*vmean**2/DR)):.6f}')


plt.figure()
plt.xlim([0,data_msd[-1,0]])
plt.xlim([0,300])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu \mathrm{m}^2$)')
freq = 200
plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,f),'g-',label=r'Theory')
plt.plot(t_sample,msd_theory_ct(t_sample,DT,DR,vmean),'b-',label=r'Constant $v = \langle v\rangle$')
plt.errorbar(data_msd[::freq,0],data_msd[::freq,3],data_msd[::freq,6],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD2.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
