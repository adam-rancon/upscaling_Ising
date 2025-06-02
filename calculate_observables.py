import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import re
import fnmatch
from tqdm import tqdm

import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader

import observables_v as obs

#import mcrgc


from models import MajorityPooling2d, DecoderSymmetrizedConv,Decoder1layer, DegWeights, LossMagnetizationW


plt.rc('font', size=22)
plt.rc('axes', titlesize=22)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rcParams["figure.figsize"] = [9,6]
plt.rcParams["figure.autolayout"] = True


#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device=torch.device('cpu')
print(f"\nselected device: {device}\n")

# This is advanced version of loading models, it is not bound by input lists of parameters but 
# looks for models that are there and reads them

upsampling_method='nearest'
upsampler={}

############################################################################################


current_path = os.getcwd()
path0=current_path+'/models/'
patt='conv_32'

files_in_models = os.listdir(path0)

# Here we read the trained models from the subfolder "models/". Depending on whether the name has a pattern 
#"conv_32" (which designates the 2 layer convolutional network see details in the article), the naming is slightly different and the 
# upsampler is initialized differently

with os.scandir(path0) as entries:
    for entry in entries:
        if entry.is_file():
            file = entry.name
            print(file)
            if patt in file:
                templist0 = re.findall(r'[+-]?\d+(?:\.\d+)?', file)
                #templist1=[int(templist0[-3]),float(templist0[-1])]
                #print(di)
                #print(os.path.join(root,file))
                #print(templist0)
                key2=templist0[1]
                key1=templist0[3]
                key3=templist0[5]
                key0=1
                print(key0,key1,key2,key3)
                
                if int(key2) == 3:
                    mod=3
                    kernel_size=int(key2)
                    ups = Decoder1layer(upsampling_method=upsampling_method, kernel_size=kernel_size).to(device)
                if int(key2) == 7:
                    mod=5
                    kernel_size=int(key2)
                    ups = Decoder1layer(upsampling_method=upsampling_method, kernel_size=kernel_size).to(device)
                    
            else:
                templist0 = re.findall(r'[+-]?\d+(?:\.\d+)?', file)
                #templist1=[int(templist0[-3]),float(templist0[-1])]
                #print(di)
                #print(os.path.join(root,file))
                #print(templist0)
                #print(file)
                key0=0
                key2=templist0[1]
                key1=templist0[4]
                key3=templist0[6]
                print(key0,key1,key2,key3)
                
                if int(key2) == 3:
                    mod=0
                    kernel_size=int(key2)
                    ups = DecoderSymmetrizedConv(upsampling_method=upsampling_method, kernel_size=kernel_size).to(device)
                if int(key2) == 5:
                    mod=1
                    kernel_size=int(key2)
                    ups = DecoderSymmetrizedConv(upsampling_method=upsampling_method, kernel_size=kernel_size).to(device)
                if int(key2) == 7:
                    mod=2
                    kernel_size=int(key2)
                    ups = DecoderSymmetrizedConv(upsampling_method=upsampling_method, kernel_size=kernel_size).to(device)
            
            model_parameter_file = "models/"+file
            #print(model_parameter_file)
            #print(key2==7)
            ups.load_state_dict(torch.load(model_parameter_file,map_location=torch.device(device)))
            upsampler.update({(mod,int(key1),int(key3)):ups})
            
########################################################################

# In the downsampler "FullRandom" pooling means that each block that has 2 spins in one and 2 in other 
# direction is represented by a block spin of random direction.

pooling_method = 'FullRandom'

# define the downscaling model
downsampler = MajorityPooling2d(kernel_size=(2, 2), stride=2, pooling_half=pooling_method).to(device)


########################################################################

#Here we upsample using the trained models and calculate all the physical observables from the 
# upscaled configurations. We explain the various quantities in the definitions of the arrays related 
# to them.

# this is the dimensionality of the system
dim = 2


nb_samples = 10000
batchsize=100
rb=float(batchsize)

# Number of upscale steps from L=1 
nb_upscale=7
#hi=np.zeros(nb_upscale)

for ukeys in upsampler:

    print(f"\n calculating for model: {ukeys}\n", flush=True)

    # mean order parameter (magnetization) density
    ama=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # mean absolute order parameter density
    aabsma=torch.zeros(nb_upscale+1,dtype=torch.float64)
    #mean energy density
    aea=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # mean order parameter density squared
    amma=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # mean energy density squared
    aeea=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # mean energy magentization cumulant (disconnected)
    amea=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # mean order parameter density to the power 4
    am4a=torch.zeros(nb_upscale+1,dtype=torch.float64)

    # the 36 couplings that fit on a 3x3 plaquette (see the article for details)  
    # calculated on a lattice of size L and followed by that of size L/2
    aca=torch.zeros(36,nb_upscale+1,dtype=torch.float64)
    acAa=torch.zeros(36,nb_upscale+1,dtype=torch.float64)

    # the "me" cumulant
    me=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # the magnetic susceptibility
    ch=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # the heat capacity
    cp=torch.zeros(nb_upscale+1,dtype=torch.float64)
    # the Binder ratio
    V4=torch.zeros(nb_upscale+1,dtype=torch.float64)

    # the matrices needed to build the RSRG matrix T. See details in the 
    # article. We have 2 of each sets because we calculate the RSRG matrix in 
    # 2 different ways, a priori different, but which ultimately give the same result
    # up to statistical error. 
    # One way is at given L ask for an upscaled lattice of size 2L and then use 
    # these 2 lattices to calculate the matrix. Other way is at given L downscale the 
    # configuration and calculate the matrix with these two L. 
    aD0a=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    aU0a=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    D1=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    U1=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)

    DDm1=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    UU=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    TT=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)

    aD0Aa=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    aU0Aa=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    D1A=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    U1A=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)

    DDAm1=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    UUA=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)
    TTA=torch.zeros(36,36,nb_upscale+1,dtype=torch.float64)


    Ls=torch.zeros(nb_upscale)

    lattice={}
    
    # the statistics of magnetizations for every of N configurations of every lattice 
    mags_of_L={}

    t1=time.time()

    # This is the upscale step for a batch of lattices
    for i in range(nb_samples // batchsize):
    
        #lattice0 = torch.cat((torch.zeros((batchsize//2, 1, 1, 1)), torch.ones((batchsize//2, 1, 1, 1))), dim=0).to(device=device, dtype=torch.float32)
        #
        if batchsize == 1: 
            lattice0 = torch.tensor(np.array([[np.mod(i,2)]])).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        else :
            #lattice0 = torch.cat((torch.zeros((batchsize//2, 1, 1, 1)), torch.ones((batchsize//2, 1, 1, 1))), dim=0).to(device=device, dtype=torch.float32)
            lattice0 = torch.tensor(np.array([([0],[1])*(batchsize // 2)])).squeeze(0).unsqueeze(1).unsqueeze(1).to(device=device, dtype=torch.float32)
            
        lattice.update({0:lattice0})
        
        for ii in range(nb_upscale):
            LL = 2**(ii+1)
        
#        # if I want k=7 (or 5), I need to upscale for ii<2 (or1) with k=3 because of padding problems
#        DANGER : In order for below to work for k=7 and 5 I need to have a model with k=3 and other keys the same! 
            mod=ukeys[0]
    
            if int(ukeys[0])==1:
                if ii<1:
                    mod=0
                else:
                    mod=int(ukeys[0])
                    
            if int(ukeys[0])==2:
                if ii<2:
                    mod=0
                else:
                    mod=int(ukeys[0])
            
            if int(ukeys[0])==5:
                if ii<2:
                    mod=3
                else:
                    mod=int(ukeys[0])
        
        
            #lattice.update({ii+1:upsampler[mod,ukeys[1],ukeys[2]](lattice[ii])})
            #lattice[ii+1] = (torch.rand_like(lattice[ii+1]) < torch.sigmoid(lattice[ii+1])).float()
            lattice[ii+1] = upsampler[mod,ukeys[1],ukeys[2]].upsample(lattice[ii])
        
            lattice_down=downsampler(lattice[ii+1])
    
            if ii>1:
                # here I want to do averages of all the quantities that I am interested in.
                # I shal calculate temprorary tensors with m, abs(m) and e and then using vector multiplication 
                # calculate contributions of everything below
                tm=obs.m(lattice[ii+1])
                tam=abs(tm)
                te=obs.e(lattice[ii+1])
                tae=abs(te)
                tm2=torch.mul(tm,tm)
            
                ama[ii]+=torch.sum(tm).view(ama[ii].shape)   
                aabsma[ii]+=torch.sum(tam).view(ama[ii].shape)   
                aea[ii]+=torch.sum(te).view(aea[ii].shape)        
                amea[ii]+=torch.matmul(tae,tam).view(amea[ii].shape)        
                amma[ii]+=torch.matmul(tm,tm).view(amma[ii].shape) 
                aeea[ii]+=torch.matmul(te,te).view(aeea[ii].shape)
                am4a[ii]+=torch.matmul(tm2,tm2).view(am4a[ii].shape)
            
                coip1=obs.couplings(lattice[ii+1])
                cod=obs.couplings(lattice_down)
                coi=obs.couplings(lattice[ii])
            
                aca[:,ii]+=torch.sum(coip1,0).view(aca[:,ii].shape)  
                acAa[:,ii-1]+=torch.sum(cod,0).view(acAa[:,ii-1].shape)  
                aD0a[:,:,ii]+=torch.mm(coi.t(),coi).view(aD0a[:,:,ii].shape)
                aU0a[:,:,ii]+=torch.mm(coi.t(),4*coip1).view(aU0a[:,:,ii].shape)
                aD0Aa[:,:,ii]+=torch.mm(cod.t(),cod).view(aD0Aa[:,:,ii].shape)
                aU0Aa[:,:,ii]+=torch.mm(cod.t(),4*coip1).view(aU0Aa[:,:,ii].shape)
            
                if LL not in mags_of_L.keys():
                    mags_of_L.update({(LL):tm.tolist()})
                else:
                    mags_of_L[LL]+=tm.tolist()
                      
    for ii in range(2,nb_upscale):
        LL = 2**(ii+1)
    
        Ls[ii]=LL
    
        ama[ii]=ama[ii]/nb_samples
        aabsma[ii]=aabsma[ii]/nb_samples
        aea[ii]=aea[ii]/nb_samples
        amea[ii]=amea[ii]/nb_samples
        amma[ii]=amma[ii]/nb_samples
        aeea[ii]=aeea[ii]/nb_samples
        am4a[ii]=am4a[ii]/nb_samples
        aca[:,ii]=aca[:,ii]/nb_samples
        acAa[:,ii]=acAa[:,ii]/nb_samples
    
        aD0a[:,:,ii]=aD0a[:,:,ii]/nb_samples
        aU0a[:,:,ii]=aU0a[:,:,ii]/nb_samples
        aD0Aa[:,:,ii]=aD0Aa[:,:,ii]/nb_samples
        aU0Aa[:,:,ii]=aU0Aa[:,:,ii]/nb_samples
    
        D1[:,:,ii]=torch.outer(aca[:,ii-1],aca[:,ii-1])
        U1[:,:,ii]=torch.outer(aca[:,ii-1],4*aca[:,ii])
    
        D1A[:,:,ii]=torch.outer(acAa[:,ii-1],acAa[:,ii-1])
        U1A[:,:,ii]=torch.outer(acAa[:,ii-1],4*aca[:,ii])
    
        me[ii]=(LL**dim)*(amea[ii]-aabsma[ii]*aea[ii])/aabsma[ii]
        ch[ii]=(LL**dim)*(amma[ii]-ama[ii]*ama[ii])
        cp[ii]=(LL**dim)*(aeea[ii]-aea[ii]*aea[ii])
        V4[ii]=am4a[ii]/(amma[ii]**2)
    
        DDm1[:,:,ii]=torch.pinverse(aD0a[:,:,ii]-D1[:,:,ii])
    
        DDAm1[:,:,ii]=torch.pinverse(aD0Aa[:,:,ii]-D1A[:,:,ii])
    
        TT[:,:,ii]=torch.mm(DDm1[:,:,ii],(aU0a[:,:,ii]-U1[:,:,ii]))
        TTA[:,:,ii]=torch.mm(DDAm1[:,:,ii],(aU0Aa[:,:,ii]-U1A[:,:,ii]))

        
    #here we print all the observables in the files.
    file0='quantities_from_models/av_m_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file1='quantities_from_models/av_e_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file2='quantities_from_models/me_cumulant_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file3='quantities_from_models/chi_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file3a='quantities_from_models/cp_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file4='quantities_from_models/av_couplings_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file5='quantities_from_models/av_T_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file6='quantities_from_models/av_couplingsA_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file7='quantities_from_models/av_TA_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file8='quantities_from_models/Vfour_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'
    file9='quantities_from_models/all_mags_L'+str(ukeys[1])+'_m'+str(ukeys[0])+'_t'+str(ukeys[2])+'_s'+str(nb_samples)+'.npy'


    np.save(file0,ama)
    np.save(file1,aea)
    np.save(file2,me)
    np.save(file3,ch)
    np.save(file3a,cp)
    np.save(file4,aca)
    np.save(file5,TT)
    np.save(file6,acAa)
    np.save(file7,TTA)
    np.save(file8,V4)
    np.save(file9,mags_of_L)
            
    t2=time.time()

    print('Execution time:', t2-t1, 'seconds')
