# Imports basics
import os
import numpy as np
import h5py
import json
import setGPU
import sklearn
import corner
import scipy
import time
from tqdm import tqdm 
import utils #import *
# Imports neural net tools
import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc

import argparse
parser = argparse.ArgumentParser(description='Test.')                                 
parser.add_argument('--opath', action='store', type=str, help='Path to output files.') 
#parser.add_argument('--mpath', action='store', type=str, default="", help='Path to input files.') 

args = parser.parse_args()

os.system("mkdir "+args.opath)

# Defines the interaction matrices
class GraphNetnoSV(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De=5, Do=6, softmax=False):
        super(GraphNetnoSV, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.Nv = 0 
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.S = 0
        self.n_targets = n_targets
        self.assign_matrices()
        self.softmax = softmax
           
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        self.fo1 = nn.Linear(self.P + self.Dx + (self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    def forward(self, x):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        

        ####Final output matrix for particles###
        

        C = torch.cat([x, Ebar_pp], 1)
        del Ebar_pp
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        
        ### Classification MLP ###

        N = self.fc_fixed(N)
        
        if self.softmax:
            N = nn.Softmax(dim=1)(N)
        
        return N
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


def reshape_inputs(array, n_features):
    array = np.split(array, n_features, axis=-1)
    array = np.concatenate([np.expand_dims(array[i],axis=-1) for i in range(n_features)],axis=-1)
    return array

def train_val_test_split(array,train=0.8,val=0.1,test=0.1):
    n_events = array.shape[0]
    return array[:int(n_events*train)], array[int(n_events*train):int(n_events*(train+val))], array[int(n_events*(train+val)):]

def axis_settings(ax):
    import matplotlib.ticker as plticker
    #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    ax.xaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
    ax.tick_params(direction='in', axis='both', which='minor' , length=6)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')    
    #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
    ax.grid(which='major', alpha=0.9, linestyle='dotted')
    return ax

def plot_features(array, labels, feature_labels):
    array = np.nan_to_num(array,nan=0.0, posinf=0., neginf=0.) 
    if labels.shape[1]==2: 
        processes = ["Z'","QCD"]
    else: 
        processes = ["Z'(bb)","Z'(cc)","Z'(qq)","QCD"]

    if len(array.shape) == 2:
        for ifeat in range(array.shape[-1]):
            plt.clf()
            fig,ax = plt.subplots() 
            ax = axis_settings(ax)
            for ilabel in range(labels.shape[1]):

               tmp = array[labels[:,ilabel].astype(bool),ifeat]
               ax.hist(tmp,  
                       label=processes[ilabel], bins=20, 
                       histtype='step',alpha=0.7, 
                       density=True,
               )
               ax.set_xlabel(feature_labels[ifeat])
               ax.legend(loc="upper right")
               plt.savefig(args.opath+'/'+feature_labels[ifeat]+'.png')

    elif len(array.shape) == 3:
            plt.clf()
            fig,ax = plt.subplots() 
            ax = axis_settings(ax)
            for ilabel in range(labels.shape[1]):
               for ipart in range(array.shape[1]):
                   tmp = array[labels[:,ilabel].astype(bool),ipart,ifeat]
                   ax.hist(tmp,  
                           label=processes[ilabel], bins=20, 
                           histtype='step',alpha=0.7, 
                           density=True,
                   )
                   ax.set_xlabel(feature_labels[ifeat])
                   ax.legend(loc="upper right")
                   plt.savefig(args.opath+'/'+'ipart_'+str(ipart)+'_'+feature_labels[ifeat]+'.png')
                   if ipart > 40: break
    else:
        raise ValueError("I don't understand this array shape",array.shape)
 
srcDir = '/n/holyscratch01/iaifi_lab/jkrupa/10Mar22-MiniAODv2/18May22-morevars-v3_test/zpr_fj_msd/2017/'
n_particle_features = 6
n_particles = 150
n_vertex_features = 13
n_vertex = 5
is_binary = False
# In[ ]:


# conver training file
data = h5py.File(os.path.join(srcDir, 'total_df.h5'),'r')
particleData  = reshape_inputs(data['p_features'], n_particle_features)
vertexData    = reshape_inputs(data['SV_features'], n_vertex_features)
singletonData = np.array(data['singletons'])
labels        = singletonData[:,-3:]
singletonFeatureData = np.array(data['singleton_features'])

if is_binary:
   labels = np.expand_dims(np.sum(labels, axis=1),axis=-1)
labels[labels>0] = 1

qcd_label = np.zeros((len(labels),1))
is_qcd    = np.where(np.sum(labels,axis=1)==0)
qcd_label[is_qcd] = 1.
labels    = np.concatenate((labels,qcd_label),axis=1)
plot_features(singletonData,labels,utils._singleton_labels)
plot_features(vertexData,labels,utils._SV_features_labels)
plot_features(particleData,labels,utils._p_features_labels)
plot_features(singletonFeatureData,labels,utils._singleton_features_labels)
np.random.seed(42)

p = np.random.permutation(particleData.shape[0])
print(p,particleData.shape, vertexData.shape, singletonData.shape, singletonFeatureData.shape, labels.shape)

particleData, vertexData, singletonData, singletonFeatureData, labels = particleData[p,:,:], vertexData[p,:,:], singletonData[p,:], singletonFeatureData[p,:], labels[p,:] #np.random.shuffle(particleData, vertexData, singletonData, singletonFeatureData, labels)

particleDataTrain,  particleDataVal,  particleDataTest  = train_val_test_split(particleData)
vertexDataTrain,    vertexDataVal,    vertexDataTest    = train_val_test_split(vertexData)
singletonDataTrain, singletonDataVal, singletonDataTest = train_val_test_split(singletonData)

