import numpy as np

Nterms = int(1e5)

# cos
def Dex_cosine_a(v,T,taur):
    omega = 2*np.pi/T
    DR = 1./taur
    return DR*v**2/(4*(DR**2+omega**2))

def f_sine_a(B):
    return (np.pi**2/8)/(1+(2*np.pi/B)**2)

def f_abssine_f(B,N=Nterms):
    acu = np.zeros_like(B)
    for n in range(1,N+1):
        acu += (1/(1-4*n**2)**2)*(1/(1+(4*np.pi*n/B)**2))
    return 1+2*acu

# (1+cos)/2
def Dex_harmonic_a(v,T,taur):
    omega = 2*np.pi/T
    return v**2*taur/8. + v**2/16./(taur*((1/taur)**2 + omega**2))

#Exponentials
def Dex_exponential_f(v,T,tau,taur,N=Nterms):
    Ttau = T/tau
    Ttaur = T/taur
    vmean = v/Ttau*(1-np.exp(-Ttau))

    acu = np.zeros_like(T)
    for n in range(1,N+1):
        acu += (1/(1+(2*np.pi*n/Ttau)**2))*(1/(1+(2*np.pi*n/Ttaur)**2))
    return taur*vmean**2*((1.0/2.0)+acu)

def f_exponential_f(A,B,N=Nterms):
    acu = np.zeros_like(A)
    for n in range(1,N+1):
        acu += (1/(1+(2*np.pi*n/A)**2))*(1/(1+(2*np.pi*n/B)**2))
    return 2*(0.5+acu)

#Exponentials (analytic)
def Dex_exponential_a(v,T,tau,taur):
    Ttau = T/tau
    Ttaur = T/taur
    taurtau = Ttau/Ttaur
    e = np.exp
    vmean = v/Ttau*(1.0-e(-Ttau))

    # preterm = taur*vmean**2*(Ttau**2)/(2.0*(1.0-taurtau)*(1.0-e(-Ttau))**2)
    preterm = taur*vmean**2*(Ttau**2)/(2.0*(1.0-e(-Ttau))**2)
    term1 = ((1.0-e(Ttau))*(e(-2.0*Ttau)-e(Ttaur-Ttau)))/((1.0-e(Ttaur))*(Ttaur+Ttau))
    term2 = (1.0-e(-2.0*Ttau))/(2.0*Ttau)

    preterml = taur*vmean**2*(Ttau**2)/(2.0*(1.0-e(-Ttau))**2)
    numerator = 0.5*e(-2*Ttau)+0.5*e(Ttau)-0.5*e(-Ttau)-0.5+Ttau*(1-e(-Ttau))
    denominator = 2*Ttau*(e(Ttau)-1)

    thres = 1e-10
    mask     = np.where(np.abs(taurtau-1)<thres)
    compmask = np.where(np.abs(taurtau-1)>thres)
    res = np.empty_like(Ttau)
    res[mask]     = preterml[mask]*(numerator[mask]/denominator[mask])
    res[compmask] = preterm[compmask]*(term1[compmask] + term2[compmask])/(1.0-taurtau[compmask])
    # res = np.where(np.abs(taurtau-1)<1e-4,preterml*(numerator/denominator),preterm*(term1 + term2)/(1.0-taurtau))
    return res

def f_exponential_a(A,B):
    e = np.exp
    e1 = np.expm1

    thres = 1e-10
    thresu = 1e4
    limit_1     = np.where(np.abs(A-B)<thres)
    limit_2     = np.where(np.abs(A)>thresu)
    limit_3     = np.where(np.abs(B)>thresu)
    nolimit     = np.where(np.logical_and(np.logical_and(np.abs(A-B)>thres,np.abs(A)<thresu),np.abs(B)<thresu))

    Aback = A.copy()
    Bback = B.copy()

    A = Aback[nolimit]
    B = Bback[nolimit]

    preterm = (A**2*B)/(1.0-e(-A))**2/(B-A)
    term1a  = (e(-2*A)-e(B-A))/(A+B)
    term1b  = (1-e(A))/(1-e(B))
    term2   = (1-e(-2*A))/(2*A)

    A = Aback[limit_1]
    B = Bback[limit_1]

    limit1pre = A**3/(1-e(-A))**2
    limit1num = 1-2*A+e(-A)*(1+2*A)-e(A)-e(-2*A)
    limit1den = 4*A**2*(1-e(A))

    A = Aback[limit_2]
    B = Bback[limit_2]
    limit2 = (B/2)*((e(B)+1)/(e(B)-1))

    A = Aback[limit_3]
    B = Bback[limit_3]
    limit3 = (A/2)*((e(A)+1)/(e(A)-1))

    res = np.empty_like(Aback)
    res[nolimit] = preterm*(term1a*term1b+term2)
    res[limit_1] = limit1pre*limit1num/limit1den
    res[limit_2] = limit2
    res[limit_3] = limit3
    return res

#Exponentials alternating plus and minus
def Dex_plusminus_f(v,T,tau,taur,N=Nterms):
    N *= 2
    Ttau = T/tau
    Ttaur = T/taur
    vmean = v/Ttau*(1-np.exp(-Ttau))

    acu = np.zeros_like(T)
    term = ((1.+np.exp(-Ttau))/(1.-np.exp(-Ttau)))**2
    for n in range(1,N+1,2):
        acu += (1/(1+(np.pi*n/Ttau)**2))*(1/(1+(np.pi*n/Ttaur)**2))
    return (taur*vmean**2)*term*acu

def f_plusminus_f(A,B,N=Nterms):
    N *= 2
    acu = np.zeros_like(A)
    term = ((1.+np.exp(-A))/(1.-np.exp(-A)))**2
    for n in range(1,N+1,2):
        acu += (1/(1+(np.pi*n/A)**2))*(1/(1+(np.pi*n/B)**2))
    return 2*term*acu

def Dex_plusminus_a(v,T,tau,taur,N=Nterms):
    Ttau = T/tau
    Ttaur = T/taur
    e = np.exp
    vmean = v/Ttau*(1-e(-Ttau))

    Delta = T
    beta = 1./tau
    DR = 1./taur

    preterm = 1./(2*Delta*(DR-beta))
    term1 = (e(-2*beta*Delta)-e((DR-beta)*Delta))/(DR+beta)
    term1b = (1.+e(beta*Delta))/(1.+e(DR*Delta))
    term2 = (1.-e(-2*beta*Delta))/(2*beta)

    return v**2*preterm*(term1*term1b + term2)

def f_plusminus_a(A,B):
    e = np.exp

    thres = 1e-10
    thresu = 1e4
    limit_1     = np.where(np.abs(A-B)<thres)
    limit_2     = np.where(np.abs(A)>thresu)
    limit_3     = np.where(np.abs(B)>thresu)
    nolimit     = np.where(np.logical_and(np.logical_and(np.abs(A-B)>thres,np.abs(A)<thresu),np.abs(B)<thresu))

    Aback = A.copy()
    Bback = B.copy()

    A = Aback[nolimit]
    B = Bback[nolimit]

    preterm = (A**2*B)/(1.0-e(-A))**2/(B-A)
    term1a  = (e(-2*A)-e(B-A))/(A+B)
    term1b  = (1+e(A))/(1+e(B))
    term2   = (1-e(-2*A))/(2*A)

    A = Aback[limit_1]
    B = Bback[limit_1]

    limit1pre = A**3/(1-e(-A))**2
    limit1num = 1-2*A-e(-A)*(1+2*A)+e(A)-e(-2*A)
    limit1den = 4*A**2*(1+e(A))

    A = Aback[limit_2]
    B = Bback[limit_2]
    limit2 = (B/2)*((e(B)-1)/(e(B)+1))

    A = Aback[limit_3]
    B = Bback[limit_3]
    limit3 = (A/2)*((e(A)+1)/(e(A)-1))

    res = np.empty_like(Aback)
    res[nolimit] = preterm*(term1a*term1b+term2)
    res[limit_1] = limit1pre*limit1num/limit1den
    res[limit_2] = limit2
    res[limit_3] = limit3
    return res


#Exponentials with randomly alternating plus and minus
def Dex_randomplusminus_a(v,T,tau,taur,N=Nterms):
    Ttau = T/tau
    Ttaur = T/taur
    e = np.exp
    vmean = v/Ttau*(1-e(-Ttau))
    v2mean = v**2/Ttau*(1.0-np.exp(-2*Ttau))/2

    Delta = T
    beta = 1./tau
    DR = 1./taur

    if(type(T)==type(1.0) or type(tau)==type(1.0)):
        preterm = 1./(2*Delta*(DR-beta))
        term1 = (e(-2*beta*Delta)-e((DR-beta)*Delta))/(DR+beta)
        term1b = e((beta-DR)*Delta)
        term2 = (1.-e(-2*beta*Delta))/(2*beta)
        res = v**2*preterm*(term1*term1b+term2)
        return res

    thres = 1e-10
    thres0 = 1e10
    mask     = np.where(np.abs(DR-beta)<thres)
    mask0    = np.where(np.logical_and(np.abs(DR-beta)<thres,np.abs(Delta)>thres0))
    compmask = np.where(np.abs(DR-beta)>thres)

    Deltaback = Delta.copy()
    betaback = beta.copy()

    Delta = Deltaback[compmask]
    beta = betaback[compmask]

    preterm = 1./(2*Delta*(DR-beta))
    term1 = (e(-2*beta*Delta)-e((DR-beta)*Delta))/(DR+beta)
    term1b = e((beta-DR)*Delta)
    term2 = (1.-e(-2*beta*Delta))/(2*beta)

    Delta = Deltaback[mask]
    beta = betaback[mask]

    preterml = 1./(2*Delta)
    numl = 1-e(-2*DR*Delta)*(1+2*DR*Delta)
    denl = 4*DR**2

    Delta = Deltaback[mask0]
    beta = betaback[mask0]

    preterml0 = 1./(2*Delta)
    numl0 = Delta**2*(e(-2*DR*Delta)+2*e(-DR*Delta))
    denl0 = 6.

    res = np.empty_like(T)
    # res[mask]     = 0.0
    # res[mask] = Dex_plusminus_f(v,T[mask],tau[mask],taur)
    res[mask]     = v**2*preterml*(numl/denl)
    res[compmask] = v**2*preterm*(term1*term1b+term2)
    res[mask0]    = v**2*preterml0*(numl0/denl0)

    return res

def f_randomplusminus_a(A,B):
    e = np.exp

    thres= 1e-10
    thresu = 1e4
    limit_1     = np.where(np.abs(A-B)<thres)
    limit_2     = np.where(np.abs(A)>thresu)
    limit_3     = np.where(np.abs(B)>thresu)
    nolimit     = np.where(np.logical_and(np.logical_and(np.abs(A-B)>thres,np.abs(A)<thresu),np.abs(B)<thresu))

    Aback = A.copy()
    Bback = B.copy()

    A = Aback[nolimit]
    B = Bback[nolimit]

    preterm = (A**2*B)/(1.0-e(-A))**2/(B-A)
    term1a  = (e(-2*A)-e(B-A))/(A+B)
    term1b  = e(A-B)
    term2   = (1-e(-2*A))/(2*A)

    A = Aback[limit_1]
    B = Bback[limit_1]

    limit1pre = A**2/(1-e(-A))**2
    limit1num = 1-e(-2*A)*(1+2*A)
    limit1den = 4*A

    A = Aback[limit_2]
    B = Bback[limit_2]
    limit2 = B/2


    res = np.empty_like(Aback)
    res[nolimit] = preterm*(term1a*term1b+term2)
    res[limit_1] = limit1pre*limit1num/limit1den
    res[limit_2] = limit2
    return res

def f_randomplusminus_sa(A,B):
    e = np.exp

    preterm = (A**2*B)/(1.0-e(-A))**2/(B-A)
    term1a  = (e(-2*A)-e(B-A))/(A+B)
    term1b  = e(A-B)
    term2   = (1-e(-2*A))/(2*A)

    res = preterm*(term1a*term1b+term2)
    return res


#Piecewise

def f_pw_f(A,B,N=Nterms):
    acu = np.zeros_like(A)
    for n in range(1,N+1):
        acu += (2*(np.sin(n*np.pi/A))**2)/((n*np.pi/A)**2*(1+(2*n*np.pi/B)**2))
    return 1 + acu

