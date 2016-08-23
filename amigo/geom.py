import numpy as np
from scipy.interpolate import interp1d

def length(R,Z,norm=True):
    L = np.append(0,np.cumsum(np.sqrt(np.diff(R)**2+np.diff(Z)**2)))
    if norm: L = L/L[-1]
    return L
    
def vector_length(X,norm=True):
    L = np.append(0,np.cumsum(np.sqrt(np.diff(X[:,0])**2+
                                      np.diff(X[:,1])**2+
                                      np.diff(X[:,2])**2)))
    if norm: L = L/L[-1]
    return L
    
def space(R,Z,npoints):
    L = length(R,Z)
    l = np.linspace(0,1,npoints)
    R,Z = interp1d(L,R)(l),interp1d(L,Z)(l)
    return R,Z
    
def rotate(theta):
    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
                   [np.sin(theta),-np.cos(theta),0],
                   [0,0,1]])
    return Rz
    
def normal(R,Z):
    dR,dZ = np.gradient(R),np.gradient(Z)
    mag = np.sqrt(dR**2+dZ**2)
    index = mag>0
    dR,dZ,mag = dR[index],dZ[index],mag[index]  # clear duplicates
    R,Z = R[index],Z[index]
    t = np.zeros((len(R),3))
    t[:,0],t[:,1] = dR/mag,dZ/mag
    n = np.cross(t, [0,0,1])
    nR,nZ = n[:,0],n[:,1]
    return (nR,nZ)