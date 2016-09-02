import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as spline
import pylab as pl

def theta_sort(R,Z,xo,origin='lfs'):
    if origin == 'lfs':
        theta = np.arctan2(Z-xo[1],R-xo[0])
    elif origin == 'top':
        theta = np.arctan2(xo[0]-R,Z-xo[1])
    index = np.argsort(theta)
    R,Z = R[index],Z[index]
    return R,Z

def rt(R,Z,xo):
    theta = np.unwrap(np.arctan2(Z-xo[1],R-xo[0]))
    radius = np.sqrt((Z-xo[1])**2+(R-xo[0])**2)
    index = np.argsort(theta)
    radius,theta = radius[index],theta[index]
    return radius,theta
    
def rz(radius,theta,xo):
    R = xo[0]+radius*np.cos(theta)
    Z = xo[1]+radius*np.sin(theta)
    return R,Z

def rzSpline(R,Z,xo,npoints=500,w=None,s=0.005):
    radius,theta = rt(R,Z,xo)
    Ts = np.linspace(theta[0],theta[-1],npoints)
    if w is None:
        radius = spline(theta,radius,s=s)(Ts)
    else:
        radius = spline(theta,radius,w=w,s=s)(Ts)
    Rs,Zs = rz(radius,Ts,xo)
    return Rs,Zs,Ts
        
def rzSLine(R,Z,npoints=500,s=0,Hres=False):
    L = length(R,Z)
    if Hres: npoints *= 10
    Linterp = np.linspace(0,1,npoints)
    if s == 0:
        R = interp1d(L,R)(Linterp)
        Z = interp1d(L,Z)(Linterp)
    else:
        R = spline(L,R,s=s)(Linterp)
        Z = spline(L,Z,s=s)(Linterp)
    return R,Z
        
def rzInterp(R,Z,npoints=500,ends=True):
    L = length(R,Z)
    Linterp = np.linspace(0,1,npoints)
    R = interp1d(L,R)(Linterp)
    Z = interp1d(L,Z)(Linterp)
    if not ends:
        R,Z = R[:-1],Z[:-1]
    return R,Z
    
def rzfun(R,Z):  # return interpolation functions
    L = length(R,Z)
    R = interp1d(L,R)
    Z = interp1d(L,Z)
    return R,Z
        
def rzCirc(R,Z):
    radius,theta = rt(R,Z)
    R,Z = rz(radius,theta)
    return R,Z
        
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
    
def max_steps(dR,dr_max):
    dRbar = np.mean(dR)
    nr=int(np.ceil(dRbar/dr_max))
    if nr < 2: nr = 2
    dr=dR/nr
    return dr,nr
    
def offset(R,Z,dR):
    dr_max = 0.02  # maximum step size
    if np.mean(dR) != 0:
        dr,nr = max_steps(dR,dr_max)
        for i in range(nr):
            nR,nZ = normal(R,Z)
            R = R+dr*nR    
            Z = Z+dr*nZ 
    return R,Z 
   
class Loop(object):
    
    def __init__(self,R,Z,xo):
        self.R = R
        self.Z = Z
        self.xo = xo
    
    def rzPut(self):
        self.Rstore,self.Zstore = self.R,self.Z
        
    def rzGet(self):
        self.R,self.Z = self.Rstore,self.Zstore
        
    def fill(self,trim=None,dR=0,dt=0,ref_o=4/8*np.pi,dref=np.pi/4,
             edge=True,ends=True,
             color='k',label=None,alpha=0.8,referance='theta',part_fill=True,
             loop=False,s=0,gap=0,plot=True):
        dt_max = 2.5
        if not part_fill:
            dt_max = dt
        if isinstance(dt,list):
            dt = self.blend(dt,ref_o=ref_o,dref=dref,referance=referance,
                            gap=gap)
        dt,nt = max_steps(dt,dt_max)
        Rin,Zin = offset(self.R,self.Z,dR)  # gap offset
        for i in range(nt):
            self.part_fill(trim=trim,dt=dt,ref_o=ref_o,dref=dref,
             edge=edge,ends=ends,color=color,label=label,alpha=alpha,
             referance=referance,loop=loop,s=s,plot=plot)
             
    def part_fill(self,trim=None,dt=0,ref_o=4/8*np.pi,dref=np.pi/4,
             edge=True,ends=True,
             color='k',label=None,alpha=0.8,referance='theta',loop=False,
             s=0,plot=True):
        Rin,Zin = self.R,self.Z
        if loop:
            Napp = 5  # Nappend
            R = np.append(self.R,self.R[:Napp])
            R = np.append(self.R[-Napp:],R)
            Z = np.append(self.Z,self.Z[:Napp])
            Z = np.append(self.Z[-Napp:],Z)
            R,Z = rzSLine(R,Z,npoints=len(R),s=s)
            if isinstance(dt,(np.ndarray,list)):
                dt = np.append(dt,dt[:Napp])
                dt = np.append(dt[-Napp:],dt)
            Rout,Zout = offset(R,Z,dt)
            Rout,Zout = Rout[Napp:-Napp],Zout[Napp:-Napp]
            Rout[-1],Zout[-1] = Rout[0],Zout[0]
        else:
            R,Z = rzSLine(self.R,self.Z,npoints=len(self.R),s=s)
            Rout,Zout = offset(R,Z,dt)
        self.R,self.Z = Rout,Zout  # update
        if trim is None:
            Lindex = [0,len(Rin)]
        else:
            Lindex = self.trim(trim)
        if plot:
            flag = 0
            for i in np.arange(Lindex[0],Lindex[1]-1):
                Rfill = np.array([Rin[i],Rout[i],Rout[i+1],Rin[i+1]])
                Zfill = np.array([Zin[i],Zout[i],Zout[i+1],Zin[i+1]])
                if flag is 0 and label is not None:
                    flag = 1
                    pl.fill(Rfill,Zfill,facecolor=color,alpha=alpha,
                            edgecolor='none',label=label)
                else:
                    pl.fill(Rfill,Zfill,facecolor=color,alpha=alpha,
                            edgecolor='none')
             
    def blend(self,dt,ref_o=4/8*np.pi,dref=np.pi/4,gap=0,referance='theta'):
        if referance is 'theta':
            theta = np.arctan2(self.Z-self.xo[1],self.R-self.xo[0])-gap
            theta[theta>np.pi] = theta[theta>np.pi]-2*np.pi
            tblend = dt[0]*np.ones(len(theta))  # inner 
            tblend[(theta>-ref_o) & (theta<ref_o)] = dt[1]  # outer 
            if dref > 0:
                for updown in [-1,1]:
                    blend_index = (updown*theta>=ref_o) &\
                                    (updown*theta<ref_o+dref)
                    tblend[blend_index] = dt[1]+(dt[0]-dt[1])/dref*\
                                        (updown*theta[blend_index]-ref_o)
        else:
            L = length(self.R,self.Z)
            tblend = dt[0]*np.ones(len(L))  # start
            tblend[L>ref_o] = dt[1]  # end
            if dref > 0:
                blend_index = (L>=ref_o) & (L<ref_o+dref)
                tblend[blend_index] = dt[0]+(dt[1]-dt[0])/dref*(L[blend_index]-
                                                                ref_o)
        return tblend
        
 
        
