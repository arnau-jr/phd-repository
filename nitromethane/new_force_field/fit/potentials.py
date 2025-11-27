import numpy as np

def quadratic(x,xe=0.):
    return (x-xe)**2

def cubic(x,xe=0.):
    return (x-xe)**3

def quartic(x,xe=0.):
    return (x-xe)**4

def dcosine(x,xe=0.):
    return np.cos(2.*x) - np.cos(2.*xe)

def dquadratic(x,xe=0.):
    delta = x-xe
    if(delta> np.pi): delta +=-2.*np.pi
    if(delta<-np.pi): delta += 2.*np.pi
    return delta**2

def dcubic(x,xe=0.):
    delta = x-xe
    if(delta> np.pi): delta +=-2.*np.pi
    if(delta<-np.pi): delta += 2.*np.pi
    return delta**3

def dquartic(x,xe=0.):
    delta = x-xe
    if(delta> np.pi): delta +=-2.*np.pi
    if(delta<-np.pi): delta += 2.*np.pi
    return delta**4

def dcosmult1pi(x,xe=0.):
    return (1.+np.cos(1.*x - np.pi))

def dcosmult1zero(x,xe=0.):
    return (1.+np.cos(1.*x))

def dcosmult2(x,xe=0.):
    return (1.+np.cos(2.*x - np.pi))

def dcosmult2neg(x,xe=0.):
    return (1.+np.cos(2.*x + np.pi))

def dcosmult3zero(x,xe=0.):
    return (1.+np.cos(3.*x))

def dcosmult390(x,xe=0.):
    return (1.+np.cos(3.*x - np.pi*90./180.0))

def dcosmult390neg(x,xe=0.):
    return (1.+np.cos(3.*x + np.pi*90./180.0))

def dcosmult3eq(x,xe=0.):
    return (1.+np.cos(3.*x - xe))
