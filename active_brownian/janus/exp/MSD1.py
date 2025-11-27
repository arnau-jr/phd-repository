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

def D_ex_fourier(v,tau,taur,T,N=int(1e6)):
    tauT = tau/T
    taurT = taur/T

    vmean = v*tauT*(1.0-np.exp(-1/tauT))
    
    acu = 0.0
    for n in range(1,N+1):
        acu += (1/(1+(2*np.pi*n*tauT)**2))*(1/(1+(2*np.pi*n*taurT)**2))
    return taur*vmean**2*((1.0/2.0)+acu)

def D_ex(v,tau,taur,T):
    Ttau = T/tau
    taurtau = taur/tau
    Ttaur = T/taur
    e = np.exp

    vmean = v/Ttau*(1.0-e(-Ttau))

    preterm = taur*vmean**2*(Ttau**2)/(2.0*(1.0-taurtau)*(1.0-e(-Ttau))**2)
    term1 = ((1.0-e(Ttau))*(e(-2.0*Ttau)-e(Ttaur-Ttau)))/((1.0-e(Ttaur))*(Ttaur+Ttau))
    term2 = (1.0-e(-2.0*Ttau))/(2.0*Ttau)

    return preterm*(term1 + term2)

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



data_msd = np.loadtxt('./result2Dexp1/MSD_lags.dat')

print(f'Got                 {data_msd.shape[0]} data points')

DT = 0.22
DR = 0.16

v = 10.0
tau = 0.1
T = 1.0
vmean = v/(T/tau)*(1.0-np.exp(-T/tau))

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
print(f'Defftheory        = {DT+D_ex(v,tau,1/DR,T):.6f}')
print(f'Defftheoryfourier = {DT+D_ex_fourier(v,tau,1/DR,T):.6f}')
print(f'Defftheorymean    = {(DT+(0.5*vmean**2/DR)):.6f}')
print(f'Ratio             = {(m/4)/(DT+(0.5*vmean**2/DR)):.6f}')
print(f'Ratiotheory       = {(DT+D_ex(v,tau,1/DR,T))/(DT+(0.5*vmean**2/DR)):.6f}')


plt.figure()
plt.xlim([0,data_msd[-1,0]])
# plt.xlim([0,300])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
freq = 200
plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,T),'g-',label=r'Theory')
plt.plot(t_sample,msd_theory_ct(t_sample,DT,DR,vmean),'b-',label=r'Constant $v = \langle v\rangle$')
plt.errorbar(data_msd[::freq,0],data_msd[::freq,3],data_msd[::freq,6],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
# plt.show()
plt.close()

plt.figure()
plt.xlim([0,0.2])
plt.ylim([0,0.4])
plt.xlabel(r't ($\mathrm{s}$)')
plt.ylabel(r'MSD ($\upmu\mathrm{m}^2$)')
plt.plot(t_sample,msd_theory(t_sample,DT,DR,v,T),'g-',label=r'Theory')
plt.errorbar(data_msd[:,0],data_msd[:,3],data_msd[:,6],fmt='rx',label='Data',capsize=2)
plt.legend()
plt.savefig('plots/MSD1_short.pdf',format='pdf',bbox_inches='tight',pad_inches=0.02)
