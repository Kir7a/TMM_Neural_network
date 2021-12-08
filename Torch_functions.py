import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from math import sin
import numpy as np

def create_N_for_all_angles(n_structure,wl=780, angle_start=20, angle_stop=30, points=20):
    angles=np.linspace(angle_start,angle_stop,points)
    N=np.zeros((len(angles),len(n_structure)),dtype=np.complex)
    for ind,angle in enumerate(angles):
        N[ind]=(n_structure ** 2 - n_structure[0] ** 2 * (sin(angle/180*np.pi)) ** 2+0.j) ** 0.5
    return N

def create_N_for_fixed_angle(n_structure, angle=0):
    return (n_structure ** 2 - n_structure[0] ** 2 * (sin(angle/180*np.pi)) ** 2+0.j) ** 0.5
    

def create_transfer_matrix_all_angles(N,device='cpu'):
    a=(N[:,1:]+N[:,:-1])/(2.*N[:,1:])
    real_a=torch.tensor(np.real(a))
    imag_a=torch.tensor(np.imag(a))
    b=(N[:,1:]-N[:,:-1])/(2.*N[:,1:])
    real_b=torch.tensor(np.real(b))
    imag_b=torch.tensor(np.imag(b))
    t=torch.empty(a.shape[0],a.shape[1],2,2,2,device=device)
    t[:,:,0,0,0]=t[:,:,0,1,1]=real_a
    t[:,:,0,0,1]=t[:,:,0,1,0]=real_b
    t[:,:,1,0,0]=t[:,:,1,1,1]=imag_a
    t[:,:,1,0,1]=t[:,:,1,1,0]=imag_b
    return t

def create_transfer_matrix_fixed_angle(N,device='cpu'):
    a=(N[1:]+N[:-1])/(2.*N[1:])
    real_a=torch.tensor(np.real(a))
    imag_a=torch.tensor(np.imag(a))
    b=(N[1:]-N[:-1])/(2.*N[1:])
    real_b=torch.tensor(np.real(b))
    imag_b=torch.tensor(np.imag(b))
    t=torch.empty(a.shape[0],2,2,2,device=device)
    t[:,0,0,0]=t[:,0,1,1]=real_a
    t[:,0,0,1]=t[:,0,1,0]=real_b
    t[:,1,0,0]=t[:,1,1,1]=imag_a
    t[:,1,0,1]=t[:,1,1,0]=imag_b
    return t

def tensor_phase_all_angles(d,N,k0,device='cpu'):
    d=d[:,None,:]
    phase=torch.empty(d.size(0),N.size(0),d.size(2),2,2,2,device=device)
    phase[:,:,:,0,0,0]=phase[:,:,:,0,1,1]=torch.cos(N[:,1:-1]*d*k0)
    phase[:,:,:,1,0,0]=torch.sin(N[:,1:-1]*d*k0)
    phase[:,:,:,1,1,1]=torch.sin(-N[:,1:-1]*d*k0)
    phase[:,:,:,0,0,1]=phase[:,:,:,0,1,0]=phase[:,:,:,1,0,1]=phase[:,:,:,1,1,0]=0  
    return phase

def tensor_phase_all_wavelengths(d,N,k0,device='cpu'):
    d=d[:,None,:]
    phase=torch.empty(d.size(0),k0.size(0),d.size(2),2,2,2,device=device)
    phase[:,:,:,0,0,0]=phase[:,:,:,0,1,1]=torch.cos(N[None,None,1:-1]*d*k0[None,:,None])
    phase[:,:,:,1,0,0]=torch.sin(N[None,None,1:-1]*d*k0[None,:,None])
    phase[:,:,:,1,1,1]=torch.sin(-N[None,None,1:-1]*d*k0[None,:,None])
    phase[:,:,:,0,0,1]=phase[:,:,:,0,1,0]=phase[:,:,:,1,0,1]=phase[:,:,:,1,1,0]=0  
    return phase

def torch_R_for_all_angles(phase,transfer,device='cpu'): 
    
    Total_transfer_matrix = torch.zeros((phase.size(0),phase.size(1),2,2,2),device=device)
    Total_transfer_matrix[:,:,0,0,0]=Total_transfer_matrix[:,:,0,1,1]=1    
    for i in range(phase.size(2)):        
        Total_transfer_matrix = complex_matrix_mul_batch(transfer[None,:,i], Total_transfer_matrix)
        Total_transfer_matrix = complex_matrix_mul_batch(phase[:,:,i], Total_transfer_matrix)        
    Total_transfer_matrix = complex_matrix_mul_batch(transfer[None,:,-1], Total_transfer_matrix)
    return Total_transfer_matrix

def torch_R_for_all_wavelengths(phase,transfer,device='cpu'): 
    
    Total_transfer_matrix = torch.zeros((phase.size(0),phase.size(1),2,2,2),device=device)
    Total_transfer_matrix[:,:,0,0,0]=Total_transfer_matrix[:,:,0,1,1]=1    
    for i in range(phase.size(2)):        
        Total_transfer_matrix = complex_matrix_mul_batch(transfer[None,None,i], Total_transfer_matrix)
        Total_transfer_matrix = complex_matrix_mul_batch(phase[:,:,i], Total_transfer_matrix)        
    Total_transfer_matrix = complex_matrix_mul_batch(transfer[None,None,-1], Total_transfer_matrix)
    return Total_transfer_matrix

def complex_matrix_mul_batch(A,B):
    ac=torch.matmul(A[:,:,0],B[:,:,0])
    bd=torch.matmul(A[:,:,1],B[:,:,1])
    ad=torch.matmul(A[:,:,0],B[:,:,1])
    bc=torch.matmul(A[:,:,1],B[:,:,0])
    return torch.stack((ac - bd, ad + bc),dim=2)

def Real_R_b(R):
    return -(R[:,:,0,1,0]*R[:,:,0,1,1]+R[:,:,1,1,0]*R[:,:,1,1,1])/(R[:,:,0,1,1]**2+R[:,:,1,1,1]**2)

def Imag_R_b(R):
    return -(R[:,:,1,1,0]*R[:,:,0,1,1]-R[:,:,0,1,0]*R[:,:,1,1,1])/(R[:,:,0,1,1]**2+R[:,:,1,1,1]**2)

def Abs_R_b(R):
    return torch.sqrt(Real_R_b(R)**2+Imag_R_b(R)**2)

def R_for_batch(prediction,N,transfer, wl=780, points=20,device='cpu'):
    phase=tensor_phase_all_angles(prediction,N,2*np.pi/wl,device)
    total_matrix=torch_R_for_all_angles(phase,transfer,device)
    Spectrum_pred=torch.empty((prediction.size(0),2*points),device=device)
    Spectrum_pred[:,:points]=Real_R_b(total_matrix)
    Spectrum_pred[:,points:]=Imag_R_b(total_matrix)
    return Spectrum_pred

def abs_R_for_batch(prediction,N,transfer, wl=780, points=20):
    phase=tensor_phase_all_angles(prediction,N,2*np.pi/wl)
    total_matrix=torch_R_for_all_angles(phase,transfer)
    return Abs_R_b(total_matrix)

def R_Spectrum_for_batch(prediction,N,transfer, k0, points=20,device='cpu'):
    phase=tensor_phase_all_wavelengths(prediction,N,k0,device)
    total_matrix=torch_R_for_all_wavelengths(phase,transfer,device)
    Spectrum_pred=torch.empty((prediction.size(0),2*points),device=device)
    Spectrum_pred[:,:points]=Real_R_b(total_matrix)
    Spectrum_pred[:,points:]=Imag_R_b(total_matrix)
    #return Spectrum_pred
    return (Spectrum_pred)

def abs_R_Spectrum_for_batch(prediction,N,transfer, k0, points=20):
    phase=tensor_phase_all_wavelengths(prediction,N,k0)
    total_matrix=torch_R_for_all_wavelengths(phase,transfer)
    return Abs_R_b(total_matrix)