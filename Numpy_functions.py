from math import sin, cos, tan
from cmath import exp
import numpy as np

def numpy_N_for_all_angles(n_structure,wl=780, angle_start=20, angle_stop=30, points=20):
    angles=np.linspace(angle_start,angle_stop,points)
    N=np.zeros((len(angles),len(n_structure)),dtype=np.complex)
    for ind,angle in enumerate(angles):
        N[ind]=(n_structure ** 2 - n_structure[0] ** 2 * (sin(angle/180*np.pi)) ** 2 +0.j) ** 0.5
    return N

def numpy_N_fixed_angle(n_structure,wl_start=400, wl_stop=800, angle=0, points=200):
    return (n_structure ** 2 - n_structure[0] ** 2 * (sin(angle/180*np.pi)) ** 2 +0.j) ** 0.5
    #wavelengths=np.linspace(wl_start,wl_stop,points)
    #N=np.zeros((len(wavelengths),len(n_structure)),dtype=np.complex)
    #for ind,wl in enumerate(wavelengths):

def numpy_create_tm_all_angles(N):
    a=(N[:,1:]+N[:,:-1])/(2.*N[:,1:])
    b=(N[:,1:]-N[:,:-1])/(2.*N[:,1:])
    t=np.zeros((a.shape[0],a.shape[1],2,2),dtype=np.complex)
    t[:,:,0,0]=t[:,:,1,1]=a
    t[:,:,0,1]=t[:,:,1,0]=b
    return t
 
def numpy_create_tm_fixed_angle(N):
    a=(N[1:]+N[:-1])/(2.*N[1:])
    b=(N[1:]-N[:-1])/(2.*N[1:])
    t=np.zeros((a.shape[0],2,2),dtype=np.complex)
    t[:,0,0]=t[:,1,1]=a
    t[:,0,1]=t[:,1,0]=b
    return t

def numpy_phase_all_angles(d,N,k0):
    d=d[:,None,:]
    phase=np.zeros((d.shape[0],N.shape[0],d.shape[2],2,2),dtype=np.complex)
    phase[:,:,:,0,0]=np.exp(1.j * N[:,1:-1]*d * k0)
    phase[:,:,:,1,1]=np.exp(-1.j * N[:,1:-1]*d * k0)  
    return phase

def numpy_phase_all_wavelengths(d,N,k0):
    d=d[:,None,:]
    phase=np.zeros((d.shape[0],k0.shape[0],d.shape[2],2,2),dtype=np.complex)
    phase[:,:,:,0,0]=np.exp(1.j * N[None, None, 1:-1]*d * k0[None,:,None])
    phase[:,:,:,1,1]=np.exp(-1.j * N[None, None, 1:-1]*d * k0[None,:,None])  
    return phase

def numpy_R_for_all_angles(phase,transfer): 
    
    Total_transfer_matrix = np.zeros((phase.shape[0],phase.shape[1],2,2),dtype=np.complex)
    Total_transfer_matrix[:,:,0,0]=Total_transfer_matrix[:,:,1,1]=1    
    for i in range(phase.shape[2]):        
        Total_transfer_matrix = np.matmul(transfer[None,:,i], Total_transfer_matrix)
        Total_transfer_matrix = np.matmul(phase[:,:,i], Total_transfer_matrix)        
    Total_transfer_matrix = np.matmul(transfer[None,:,-1], Total_transfer_matrix)
    return -(Total_transfer_matrix[:,:,1, 0] / Total_transfer_matrix[:,:,1, 1])

def numpy_R_for_all_wavelengths(phase,transfer): 
    
    Total_transfer_matrix = np.zeros((phase.shape[0],phase.shape[1],2,2),dtype=np.complex)
    Total_transfer_matrix[:,:,0,0]=Total_transfer_matrix[:,:,1,1]=1    
    for i in range(phase.shape[2]):        
        Total_transfer_matrix = np.matmul(transfer[None,None,i], Total_transfer_matrix)
        Total_transfer_matrix = np.matmul(phase[:,:,i], Total_transfer_matrix)        
    Total_transfer_matrix = np.matmul(transfer[None,None,-1], Total_transfer_matrix)
    return -(Total_transfer_matrix[:,:,1, 0] / Total_transfer_matrix[:,:,1, 1])