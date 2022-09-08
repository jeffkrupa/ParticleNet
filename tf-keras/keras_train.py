#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import awkward0 as awkward
import sys, os
import argparse
parser = argparse.ArgumentParser(description='Test.')                                 
parser.add_argument('--opath', action='store', type=str, help='Path to output files.') 
parser.add_argument('--mpath', action='store', type=str, default="", help='Path to input files.') 

args = parser.parse_args()

os.system("mkdir "+args.opath)
# In[2]:


import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


# In[3]:


def stack_arrays(a, keys, axis=-1):
    flat_arr = np.stack([a[k].flatten() for k in keys], axis=axis)
    return awkward.JaggedArray.fromcounts(a[keys[0]].counts, flat_arr)


# In[4]:


def pad_array(a, maxlen, value=0., dtype='float32'):
    #print("before padding",a,a.shape)
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, :len(trunc)] = trunc
    #print("after padding",x,x.shape)
    return x


# In[5]:


class Dataset(object):

    def __init__(self, filepath, singleton_list = {}, feature_dict = {}, label='label', pad_len=50, data_format='channel_first'):
        self.filepath = filepath
        self.singleton_list = singleton_list
        self.feature_dict = feature_dict
        if len(self.singleton_list) == 0:
            self.singleton_list = ["jet_msd","jet_pt","jet_eta","jet_phi","jet_n2b1","n_parts"]
        if len(feature_dict)==0:
            feature_dict['points'] = ['part_etarel', 'part_phirel']
            feature_dict['features'] = ['part_pt', 'part_dz','part_d0','part_etarel', 'part_phirel','part_pdgId']
            feature_dict['mask'] = ['part_pt']
        self.label = label
        self.pad_len = pad_len
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        self._values = {}
        self._singletons = {}
        self._label = None
        self._load()

    def _load(self):
        logging.info('Start loading file %s' % self.filepath)
        counts = None
        with awkward.load(self.filepath) as a:
            self._label = a[self.label]
            #arrs = []
            for s in self.singleton_list:
                print(s)# np.array(a[s][:10000]))
                #arrs.append([s][:10000])
                self._singletons[s] = np.array(a[s][:])
                #print(s, self._singletons[s])
            for k in self.feature_dict:
                cols = self.feature_dict[k]
                if not isinstance(cols, (list, tuple)):
                    cols = [cols]
                arrs = []
                for col in cols:
                    if counts is None:
                        counts = a[col].counts
                    else:
                        assert np.array_equal(counts, a[col].counts)
                    arrs.append(pad_array(a[col][:], self.pad_len))
                    #arrs.append(a[col])
                self._values[k] = np.stack(arrs, axis=self.stack_axis)
        logging.info('Finished loading file %s' % self.filepath)


    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key==self.label:
            return self._label
        else:
            return self._values[key]

    @property
    def singletons(self):
        return self._singletons
    
    @property
    def X(self):
        return self._values
    
    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]
        for s in self._singletons:
            self._singletons[s] = self._singletons[s][shuffle_indices]

# In[6]:


train_dataset = Dataset('/n/holyscratch01/iaifi_lab/jkrupa/10Mar22-MiniAODv2/18May22-morevars-v3_test/zpr_fj_msd/2017/converted/train_file_0.awkd', data_format='channel_last')
#val_dataset = Dataset('converted/val_file_0.awkd', data_format='channel_last')


# In[ ]:




# In[7]:


import tensorflow as tf
from tensorflow import keras
from tf_keras_model import get_particle_net, get_particle_net_lite


# In[8]:


model_type = 'particle_net_lite' # choose between 'particle_net' and 'particle_net_lite'
num_classes = train_dataset.y.shape[1]
input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
if 'lite' in model_type:
    model = get_particle_net_lite(num_classes, input_shapes)
else:
    model = get_particle_net(num_classes, input_shapes)

    
train_dataset.shuffle(1)
#print("part_pt",train_dataset.X['features'],train_dataset.X['features'].shape)#["part_pt"])
#print("part_pt",train_dataset.X['mask'],train_dataset.X['features'].shape)#["part_pt"])
#sys.exit(1)

nevents= train_dataset.X['features'].shape[0]
train_frac=0.8
val_frac=0.1

train_X = train_dataset.X.copy()
#print("train_dataset",train_dataset["features"].shape)

train_idxs = range(int(train_frac*nevents))
val_idxs   = range(int(train_frac*nevents),int((train_frac+val_frac)*nevents))
test_idxs  = range(int((train_frac+val_frac)*nevents),nevents)


if len(args.mpath) > 0:
    model= keras.models.load_model(args.mpath)
    test_X,  test_y  = {x : y[:,:,:]  for x,y in train_dataset.X.items()}, train_dataset.y[:]
    test_y_hat = model.predict([test_X["points"],test_X["features"],test_X["mask"]])
    test_singletons  = {x : y[:]  for x,y in train_dataset.singletons.items()}
    path=args.mpath.split("/")[:-1]
    path="/".join(path)
    print("inferring path ",path)
    np.savez(path+"/results.npz",
         y_hat=test_y_hat,
         y_true=test_y,
         jet_msd=test_singletons["jet_msd"],
         jet_pt=test_singletons["jet_pt"],
         jet_eta=test_singletons["jet_eta"],
         jet_phi=test_singletons["jet_phi"],
         jet_n2b1=test_singletons["jet_n2b1"],
         jet_nparts=test_singletons["n_parts"],
    )
else:
    # In[9]:
    
    
    # Training parameters
    batch_size = 1024 if 'lite' in model_type else 384
    epochs = 100
    
    
    # In[10]:
    
    
    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 10:
            lr *= 0.1
        elif epoch > 20:
            lr *= 0.01
        logging.info('Learning rate: %f'%lr)
        return lr
    
    
    # In[11]:
    
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    model.summary()
    
    
    # In[12]:
    
    
    # Prepare model model saving directory.
    import os
    save_dir = 'model_checkpoints'
    model_name = '%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, args.opath+"/"+model_name)
    
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=False)
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    #progress_bar = keras.callbacks.ProgbarLogger()
    callbacks = [checkpoint, lr_scheduler]#, progress_bar]
    
    
    # In[13]:
    
    
    #print(train_dataset.singletons)
    #print(train_dataset.singletons.keys())
    
    train_X, train_y = {x : y[train_idxs,:,:] for x,y in train_dataset.X.items()}, train_dataset.y[train_idxs]
    val_X,   val_y   = {x : y[val_idxs,:,:]   for x,y in train_dataset.X.items()}, train_dataset.y[val_idxs]
    test_X,  test_y  = {x : y[test_idxs,:,:]  for x,y in train_dataset.X.items()}, train_dataset.y[test_idxs]
    
    test_singletons  = {x : y[test_idxs]  for x,y in train_dataset.singletons.items()}
    
    #print(train_X["features"].shape)
    
    #print(train_X["features"].shape,train_y.shape,val_X["features"].shape,val_y.shape)
    #print("train_X",train_X["features"].shape)
    
    
    model.fit(train_X,train_y,
              #train_dataset.X[:int(train_frac*nevents),:,:], train_dataset.y,
              batch_size=batch_size,
              # epochs=epochs,
              epochs=epochs, # --- train only for 1 epoch here for demonstration ---
              #validation_data=(val_dataset.X, val_dataset.y),
              validation_data=(val_X,val_y),
              shuffle=True,
              callbacks=callbacks)
    

    # In[ ]:
    test_X,  test_y  = {x : y[test_idxs,:,:]  for x,y in train_dataset.X.items()}, train_dataset.y[test_idxs]
    test_y_hat = model.predict([test_X["points"],test_X["features"],test_X["mask"]])
    np.savez(save_dir+"/"+args.opath+"/results.npz",
             y_hat=test_y_hat,
             y_true=test_y,
             jet_msd=test_singletons["jet_msd"],
             jet_pt=test_singletons["jet_pt"],
             jet_eta=test_singletons["jet_eta"],
             jet_phi=test_singletons["jet_phi"],
             jet_n2b1=test_singletons["jet_n2b1"],
             jet_nparts=test_singletons["n_parts"],
    )
    
    #print(test_y[:100])#,test_y_hat[:100])
    
