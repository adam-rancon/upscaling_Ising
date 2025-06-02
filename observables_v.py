import torch
import numpy as np

# energy and magnetization calculation work with 0s and 1s because for the 
# energy it is easier like this to make a XNOR gate which is not implemented in pytorch
# and which is very useful for determining energy.


def e(x):
    """
    return the energy per spin of a spin configuration x.
    x must be a tensor of size (B,C,Lx,Ly).
    """
    expected_ndim = 4
    if x.ndim != expected_ndim:
        raise ValueError(f'Expected tensor to have {expected_ndim} dimensions, but found {x.ndim}')
    # convert to true spins
    spins = x.int()
    # compute the sum of the spins of the neighbors (i.e. neighbors_spin[...,X,Y] = sum of neighbors of spin at X,Y)
    neighbors_1 = spins.roll(shifts=1, dims=2)
    neighbors_2 = spins.roll(shifts=1, dims=3)
    
    e1=spins+neighbors_1
    e1[e1>1]=0
    e1=(-e1+1).float()
    
    e2=spins+neighbors_2
    e2[e2>1]=0
    e2=(-e2+1).float()

    return torch.mean(e1+e2, dim=(1, 2, 3))


def m(x):
    """
    return the magnetization of a spin configuration x.
    x must be a tensor of size (B,C,Lx,Ly).
    returns the mean over the last three dimensions
    """
    expected_ndim = 4
    if x.ndim != expected_ndim:
        raise ValueError(f'Expected tensor to have {expected_ndim} dimensions, but found {x.ndim}')
    spins = 2. * x - 1
    return torch.mean(spins, dim=(1, 2, 3))


def couplings(x):

    """
    return the 36 mcrg couplings averaged over lattice
    """
   
    ccc=torch.zeros((x.size(0),36),dtype=x.dtype)
   
    expected_ndim = 4
    if x.ndim != expected_ndim:
        raise ValueError(f'Expected tensor to have {expected_ndim} dimensions, but found {x.ndim}')
        
    spins = 2*x-1
   
    spins_i1j0 = spins.roll(shifts=1, dims=2)
    spins_i0j1 = spins.roll(shifts=1, dims=3)
    spins_i2j0 = spins.roll(shifts=2, dims=2)
    spins_i0j2 = spins.roll(shifts=2, dims=3)
    spins_i1j1 = spins.roll(shifts=1, dims=2).roll(shifts=1, dims=3)
    spins_i1j2 = spins.roll(shifts=1, dims=2).roll(shifts=2, dims=3)
    spins_i2j2 = spins.roll(shifts=2, dims=2).roll(shifts=2, dims=3)
    spins_i2j1 = spins.roll(shifts=2, dims=2).roll(shifts=1, dims=3)
    
    # I must not work with torch.sum to get rid of batch index because I have odd and even couplings and when I batch the odd ones 
    # they tend to average to 0!
    # I need to return the torch tensor on the batch index for every ccc and then make matrix multiplication when I do T
    
    ccc[:,0]=(torch.mean(spins, dim=(1, 2, 3)))
    
    ccc[:,1]=(torch.mean(torch.mul(spins,spins_i1j0), dim=(1, 2, 3)))
    ccc[:,2]=(torch.mean(torch.mul(spins,spins_i1j1), dim=(1, 2, 3)))
    ccc[:,3]=(torch.mean(torch.mul(spins,spins_i2j0), dim=(1, 2, 3)))
    ccc[:,4]=(torch.mean(torch.mul(spins,spins_i1j2), dim=(1, 2, 3)))
    ccc[:,5]=(torch.mean(torch.mul(spins,spins_i2j2), dim=(1, 2, 3)))
    
    ccc[:,6]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j1),spins_i0j1), dim=(1, 2, 3)))
    ccc[:,7]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),spins_i0j2), dim=(1, 2, 3)))
    ccc[:,8]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),spins_i1j2), dim=(1, 2, 3)))
    ccc[:,9]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),spins_i2j1), dim=(1, 2, 3)))
    ccc[:,10]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),spins_i2j2), dim=(1, 2, 3)))
    ccc[:,11]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j1),spins_i0j2), dim=(1, 2, 3)))
    ccc[:,12]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j1),spins_i2j2), dim=(1, 2, 3)))
    ccc[:,13]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j2),spins_i2j1), dim=(1, 2, 3)))
    ccc[:,14]=(torch.mean(torch.mul(torch.mul(spins,spins_i2j0),spins_i1j2), dim=(1, 2, 3)))
    ccc[:,15]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j2),spins_i2j2), dim=(1, 2, 3)))
    
    ccc[:,16]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j1,spins_i1j1)), dim=(1, 2, 3)))
    ccc[:,17]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i1j1,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,18]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i0j2,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,19]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j2,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,20]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i1j2,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,21]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i1j2,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,22]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i0j2,spins_i1j1)), dim=(1, 2, 3)))
    ccc[:,23]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j1,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,24]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i1j1,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,25]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i2j1,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,26]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j1),torch.mul(spins_i1j2,spins_i2j1)), dim=(1, 2, 3)))
    ccc[:,27]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j1,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,28]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i0j2,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,29]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j1),torch.mul(spins_i0j2,spins_i2j1)), dim=(1, 2, 3)))
    ccc[:,30]=(torch.mean(torch.mul(torch.mul(spins_i1j0,spins_i0j1),torch.mul(spins_i2j1,spins_i1j2)), dim=(1, 2, 3)))
    ccc[:,31]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j2),torch.mul(spins_i1j1,spins_i2j1)), dim=(1, 2, 3)))
    ccc[:,32]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j1,spins_i1j1)), dim=(1, 2, 3)))
    ccc[:,33]=(torch.mean(torch.mul(torch.mul(spins,spins_i1j0),torch.mul(spins_i0j2,spins_i2j1)), dim=(1, 2, 3)))
    ccc[:,34]=(torch.mean(torch.mul(torch.mul(spins,spins_i0j2),torch.mul(spins_i1j1,spins_i2j2)), dim=(1, 2, 3)))
    ccc[:,35]=(torch.mean(torch.mul(torch.mul(spins,spins_i2j0),torch.mul(spins_i0j2,spins_i2j2)), dim=(1, 2, 3)))
    
    return ccc
