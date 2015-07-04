""" testing fuel dataset """

import glob;

import h5py;
import numpy as np;

from fuel.datasets.hdf5 import H5PYDataset; 

datafiles_train=glob.glob("../data/sfew_features/Train_SFEW_Features/*.h5")
datafiles_valid=glob.glob("../data/sfew_features/Valid_SFEW_Features/*.h5")

def load_dataset(df):
    
    data=np.array([]);
    label=np.array([]);
    idx=0;
    for adr in df:
        f=h5py.File(adr, "r");
        d=f[adr[42:-3]][...];
    
        if not data.size:
            data=d;
        else:
            data=np.vstack((data, d));
        
        ind=np.ones((d.shape[0]))*idx;
    
        if not label.size:
            label=ind;
        else:
            label=np.hstack((label, ind));
    
        idx+=1;
        
    return data, label;

data_train, label_train=load_dataset(datafiles_train);
data_val, label_val=load_dataset(datafiles_valid);

data=np.vstack((data_train, data_val));
labels=np.hstack((label_train, label_val));

print "data loaded"

f=h5py.File("dataset.hdf5", "w");

features=f.create_dataset("features", data.shape, dtype="float32", data=data);
targets=f.create_dataset("targets", labels.shape, dtype="uint8", data=labels);

features.dims[0].label="batch";
features.dims[1].label="feature";
targets.dims[0].label="index";

split_dict = {
     'train': {'features': (0, data_train.shape[0]),
               'targets': (0, label_train.shape[0])},
     'test': {'features': (data_train.shape[0], data.shape[0]),
              'targets': (label_train.shape[0], labels.shape[0])}}

f.attrs['split']=H5PYDataset.create_split_array(split_dict);

f.flush();
f.close();

print "dataset created"


""" testing created dataset """

train_set=H5PYDataset("dataset.hdf5", which_sets=('train', ))
test_set=H5PYDataset("dataset.hdf5", which_sets=('test', ));
