#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
# This code block enables this notebook to run on google colab.
try:
    from google.colab import drive
    print('Running in colab...\n===================')
    COLAB = True
    get_ipython().system('pip install madmom torch==1.4.0 torchvision==0.5.0 librosa --upgrade')
    print('Installed dependencies!\n=======================')

    if not os.path.exists('data'):
        print('Downloading data...\n===================')
        get_ipython().system('mkdir data')
        get_ipython().system('cd data')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.1.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.2.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.3.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.4.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.5.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.6.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.7.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.audio.8.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.doc.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.error.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/45739/files/TUT-acoustic-scenes-2016-development.meta.zip?download=1')
            
        get_ipython().system('wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.1.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.2.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.audio.3.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.doc.zip?download=1')
        get_ipython().system('wget https://zenodo.org/record/165995/files/TUT-acoustic-scenes-2016-evaluation.meta.zip?download=1')
            
        get_ipython().system('for file in *.*; do mv $file ${file%?download=1}; done')
        
        get_ipython().system('unzip "*.zip"')
        get_ipython().system('rm *.zip')
        get_ipython().system('cd ..')

    print('===================\nMake sure you activated GPU support: Edit->Notebook settings->Hardware acceleration->GPU\n==================')
except:
    print('=======================\nNOT running in colab...\n=======================')
    COLAB = False


# # Data Processing (10 Points)

# In[ ]:


import os
import numpy as np

# get dataset path
dataset_path = os.path.join(os.environ['HOME'], 'shared', 'data', 'assignment_7')
if os.path.exists('data'):
    dataset_path = 'data'

development_path = os.path.join(dataset_path, 'TUT-acoustic-scenes-2016-development')
evaluation_path = os.path.join(dataset_path, 'TUT-acoustic-scenes-2016-evaluation')

development_audio_path = os.path.join(development_path, 'audio')
development_annotation_file = os.path.join(development_path, 'meta.txt')
development_error_file = os.path.join(development_path, 'error.txt')
split_definition_path = os.path.join(development_path, 'evaluation_setup')

evaluation_annotation_file = os.path.join(evaluation_path, 'meta.txt')
evaluation_audio_path = os.path.join(evaluation_path, 'audio')

data_file_clip_info = os.path.join(dataset_path, 'clip_info_final.csv')
data_file_annotations = os.path.join(dataset_path, 'annotations_final.csv')

# collect list of audio files:
development_audio_files = [af for af in os.listdir(development_audio_path) if af.endswith('.wav')]
evaluation_audio_files = [af for af in os.listdir(evaluation_audio_path) if af.endswith('.wav')]

dev_audio_total_count = len(development_audio_files)
eval_audio_total_count = len(evaluation_audio_files)

print(f'Total number of development audio files: {dev_audio_total_count}')
print(f'Total number of evaluation audio files: {eval_audio_total_count}')


# In[ ]:


# import packages
import torch
import warnings
warnings.filterwarnings("ignore")
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as torch_func
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import Dataset
#from tqdm import tqdm  # For nice progress bar!

import numpy as np
from time import time as get_time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import sklearn
from sklearn.preprocessing import LabelEncoder

import librosa


# In[ ]:


class_names = [
    "beach",
    "bus",
    "cafe/restaurant",
    "car",
    "city_center",
    "forest_path",
    "grocery_store",
    "home",
    "library",
    "metro_station",
    "office",
    "park",
    "residential_area",
    "train",
    "tram"
]

le = LabelEncoder().fit(class_names)


# # Implementation

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


# Getting a list of train and evaluation audio_file_path for each fold for TRIANING
annot_dev_train_paths = []
annot_dev_eval_paths = []
for txt_file in os.listdir(split_definition_path):
    if 'evaluate' in txt_file:
        annot_dev_eval_paths.append(os.path.join(split_definition_path, txt_file))
    if 'train' in txt_file:
        annot_dev_train_paths.append(os.path.join(split_definition_path, txt_file))

print(annot_dev_train_paths[0])
print(annot_dev_eval_paths[0])


# In[ ]:


n_mels = 32
n_mfcc = 32
sample_rate = 22050
number_of_examples_to_plot = 2
n_fft = 254
t = np.e**-10
t = 0

def MFCC(numpyArray):
    s = librosa.feature.mfcc(numpyArray, sr = sample_rate, n_mfcc = n_mfcc)
    return np.abs(s)+t

def logMel(numpyArray):
    S = librosa.feature.melspectrogram(numpyArray, sr=sample_rate, n_mels = n_mels)
    return np.log(np.abs(S)+t)

def rootmeansqrt(numpyArray):
    y = librosa.feature.rms(numpyArray)
    return y+t

def centroid(numpyArray):
    y = librosa.feature.spectral_centroid(numpyArray, sr = sample_rate, n_fft = n_fft)
    return y+t

def band(numpyArray1, numpyArray2):
    y = librosa.feature.spectral_bandwidth(numpyArray1, sr = sample_rate, n_fft = n_fft, centroid = numpyArray2)
    return y+t


# In[ ]:


# Put your data handling code here. 
# You can add additional cells below this one for structuring the notebook.
# Feel free to add markdown cells / plots / tests / etc. if it helps your presentation.

# YOUR CODE HERE
#raise NotImplementedError()

s = 200
audio_file_name = pd.read_csv(annot_dev_train_paths[0], header = None)[0][s].split('\t')[0].split('/')[1]
audio_path = os.path.join(development_audio_path,
             pd.read_csv(annot_dev_train_paths[0], header = None)[0][s].split('\t')[0].split('/')[1])
audio_label = pd.read_csv(annot_dev_train_paths[0], header = None)[0][s].split('\t')[1]
audio, sr = librosa.load(audio_path)

# we can also play the audio in this jupyter notebook
import IPython.display as ipd
print("audio for file '{}':".format(audio_file_name))
print("audio shape: {}".format(str(audio.shape)))
print("audio label '{}':".format(audio_label))
print("length of audio: '{} seconds'".format(len(audio)/sr))


# Extract features from the signal
feat = MFCC(audio) #(MFCC)
plt.imshow(feat, aspect='auto', origin='lower')
print('fmfcc',feat.shape)
plt.title('MFCC')
plt.colorbar()
plt.show()

# Extract features from the signal
feat = logMel(audio) #(mel)
plt.imshow(feat, aspect='auto', origin='lower')
print('fmel',feat.shape)
plt.title('mel')
plt.colorbar()
plt.show()

#ipd.Audio(audio, rate=sr)


# In[ ]:


audio2, sr2 = librosa.load(audio_path, sr/3)
print(sr2, len(audio2))

# Extract features from the signal
feat = librosa.feature.mfcc(audio2, sr = sr2, n_mfcc = n_mfcc)
print('fmfcc',feat.shape)
feat = np.abs(feat)+t
plt.imshow(feat, aspect='auto', origin='lower')
plt.title('mfcc')
plt.colorbar()
plt.show()

feat = librosa.feature.melspectrogram(audio2, sr = sr2, n_mels = n_mels)
print('fmel',feat.shape)
feat = np.log(np.abs(feat)+t)
plt.imshow(feat, aspect='auto', origin='lower')
plt.title('mel')
plt.colorbar()
plt.show()

ipd.Audio(audio2, rate=sr2)


# # Machine Learning Approach

# In[1]:


## Implementation of Custom Data Loader


# In[ ]:


#Dataset Loader


class DDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir, labelEncoder=le):
        super(DDataset, self).__init__()

        # self.annotations = pd.read_csv(annotations_file, header=None)
        self.annotations = pd.DataFrame(annotations_file)
        self.audio_dir = audio_dir

        self.target = self._get_all_target(self.annotations)

        self.labelEncoder = labelEncoder

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        cat_label = self._get_audio_sample_label(index)
        num_label = self.labelEncoder.transform([cat_label])
        n_mels = 32
        n_mfcc = 32
        audio, sr = librosa.load(audio_sample_path)

        #feat = librosa.feature.mfcc(audio, sr)
        #feat = np.abs(feat)+np.e**-10
        
        feat = librosa.feature.melspectrogram(audio, sr=sr, n_mels = n_mels)
        feat = np.log(np.abs(feat)+np.e**-10)
        
        signal = feat
        # convert to PyTorch tensor and return
        return  [torch.from_numpy(signal), torch.from_numpy(num_label)]


    def _get_audio_sample_path(self, index):
        sample = self.annotations.iloc[index].str.split('\t')[0][0].split('/')[1]  
        audio_path = os.path.join(self.audio_dir, sample)
        return audio_path

    def _get_audio_sample_label(self, index):
        sample_label = self.annotations.iloc[index].str.split('\t')[0][1]
        return sample_label
    
    def _get_all_target(self, annotations):
        targ = []
        for i in range(len(self.annotations)):
            targ.append(self.annotations.iloc[i].str.split('\t')[0][1])
        return targ
    
    def transformAcousticScenesToString(self, targets=[]):
        return self.labelEncoder.inverse_transform(targets) # Decode values into labels
    


# In[ ]:


# NN architecture
class Net(nn.Module):
    def __init__(self, debug=False):
        super(Net, self).__init__()
        self.debug = debug # we use this switch to enable some debug outputs...
        self.first_conv = nn.Conv2d(1, 16, 2, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.second_conv = nn.Conv2d(16, 32, 2, 1)a
        self.bn2 = nn.BatchNorm2d(32)
        self.third_conv = nn.Conv2d(32, 32, 2, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*160*3, 128)
        self.fc2 = nn.Linear(128, 15)
 

        #self.bn5 = nn.BatchNorm1d()

    def forward(self, x):

        x.unsqueeze_(1)  
        if self.debug:
            print('shapes for x:')
            print('(batch, channel, frames, features)')
            print(tuple(x.shape))
        #
        #print(x.shape)
        x = F.relu(self.first_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn1(x)
        if self.debug: # debug output
            print(tuple(x.shape))
        
        
        x = F.relu(self.second_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn2(x)
        if self.debug: # debug output
            print(tuple(x.shape))
        
        x = F.relu(self.third_conv(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn3(x)
        if self.debug: # debug output
            print(tuple(x.shape))
        
        x = x.view(-1, 32*160*3)
        if self.debug: # debug output
            print(tuple(x.shape))        
        
        x = F.relu(self.fc1(x))
        if self.debug: # debug output
            print(tuple(x.shape))        
        
        x = self.fc2(x)
        if self.debug: # debug output
            print(tuple(x.shape))
            
        return F.log_softmax(x, dim=1)
    
# tests

net = Net()
module_list = net.__dict__['_modules'].values()

num_modules = len(module_list)
#assert num_modules == 14, "expected 14 modules!"

num_conv2d = 0
num_bn2d = 0
num_do2d = 0
num_lin = 0
num_do1d = 0
# count modules...
print("Found modules:")
for idx, module in enumerate(module_list):
    print("{}:\t{}".format(idx+1, module))
    if isinstance(module, nn.Conv2d):
        num_conv2d += 1
    if isinstance(module, nn.BatchNorm2d):
        num_bn2d += 1
    if isinstance(module, nn.Dropout2d):
        num_do2d += 1
    if isinstance(module, nn.Linear):
        num_lin += 1
    if isinstance(module, nn.Dropout):
        num_do1d += 1
print('-----------')
print('All tests successful!')


# tests

net = Net(debug=True)
net.eval()
out = net.forward(torch.rand(16, 1292, 32))
assert out.shape == (16, 15), 'output shape did not match! check layer output shapes...'


print('All tests successful!')


# # Training, Inference, and Evaluation

# In[2]:


CNN_MODEL_NAME = 'cnn_adt_'

# Helper class for neural network hyper-parameters
# Do not change them until you have a working model 
class Args:
    pass

DEFAULT_ARGS = Args()
# general params
DEFAULT_ARGS.use_cuda = True
DEFAULT_ARGS.seed = 1

# architecture setup
DEFAULT_ARGS.batch_size = 32

# optimizer parameters
DEFAULT_ARGS.lr = 0.01
DEFAULT_ARGS.momentum = 0.5

# training protocoll
DEFAULT_ARGS.max_epochs = 10
DEFAULT_ARGS.patience = 5
DEFAULT_ARGS.log_interval = 20 # seconds


# In[3]:


def train_epoch_cnn(model, train_loader, optimizer, args):
    """
    Training loop for one epoch of NN training.
    Within one epoch, all the data is used once, we use mini-batch gradient descent.
    :param model: The model to be trained
    :param train_loader: Data provider
    :param optimizer: Optimizer (Gradient descent update algorithm)
    :param args: NN parameters for training and inference
    :return:
    """
    model.train()  # set model to training mode (activate dropout layers for example)
    train_loss = 0
    train_correct = 0
    acc_scores = []
    t = get_time() # we measure the needed time
    for batch_idx, (data, target) in enumerate(train_loader):  # iterate over training data
        target = target.squeeze_()
        data, target = data.to(args.device), target.to(args.device)  # move data to device (GPU) if necessary
        optimizer.zero_grad()  # reset optimizer
        output = model(data)   # forward pass: calculate output of network for input
        pred = output.max(1)[1].cpu()
        #loss = torch_func.binary_cross_entropy(output, target)  # calculate loss
        loss = torch_func.cross_entropy(output, target)
        loss.backward()  # backward pass: calculate gradients using automatic diff. and backprop.
        optimizer.step()  # udpate parameters of network using our optimizer
        train_loss += torch_func.cross_entropy(output, target, reduction='sum').item()
        acc_scores.append(sklearn.metrics.accuracy_score(target.to('cpu').numpy(), pred))
        #train_correct += (target.to('cpu').numpy() == output.max(1)[1].cpu()).sum().item()
        cur_time = get_time()
        # print some outputs if we reached our logging intervall
        if cur_time - t > args.log_interval or batch_idx == len(train_loader)-1:  
            print('[{}/{} ({:.0f}%)]\tloss: {:.6f}, took {:.2f}s'.format(
                       batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),cur_time - t))
            t = cur_time
    train_loss /= len(train_loader.dataset)  # calc mean loss
    acc_score = sum(acc_scores) / len(acc_scores)
    #train_correct /= len(train_loader.dataset)  # calc mean accuracy
    return train_loss, acc_score*100


# In[4]:


def test_cnn(model, test_loader, args):
    """
    Function wich iterates over test data (eval or test set) without performing updates and calculates loss.
    :param model: The model to be tested
    :param test_loader: Data provider
    :param args: NN parameters for training and inference
    :return: cumulative test loss
    """
    model.eval()  # set model to inference mode (deactivate dropout layers for example)
    test_loss = 0  # init overall loss
    test_correct = 0
    acc_scores = []
    with torch.no_grad():  # do not calculate gradients since we do not want to do updates
        for data, target in test_loader:  # iterate over test data
            target = target.squeeze_()
            data, target = data.to(args.device), target.to(args.device)  # move data to device 
            output = model(data) # forward pass
            pred = output.max(1)[1].cpu()
            # claculate loss and add it to our cumulative loss
            #test_loss += torch_func.binary_cross_entropy(output, target, reduction='sum').item()
            test_loss += torch_func.cross_entropy(output, target, reduction='sum').item()
            acc_scores.append(sklearn.metrics.accuracy_score(target.to('cpu').numpy(), pred))
    test_loss /= len(test_loader.dataset)  # calc mean loss
    #test_correct /= len(test_loader.dataset)  # calc mean accuracy
    acc_score = sum(acc_scores) / len(acc_scores)
    #print('Average eval loss: {:.4f}\n'.format(test_loss, len(test_loader.dataset)))
    #print("Average loss: {:.4f}, Accuracy:{:.4f}".format(test_loss,acc_score))
    return test_loss, acc_score*100


# In[5]:


# YOUR CODE HERE
#raise NotImplementedError()
def train_network(smoke_test=False, load_model=False, args=DEFAULT_ARGS):
    """
    Run CNN training using the datasets.
    :param smoke_test: bool, run a quick pseudo training to check if everything works
    :param load_model: bool or string, load the last trained model (if set to True), or the model in file (string)
    :param args: hyperparameters for training
    :return: trained model
    """
    if smoke_test:  # set hyperparameters to run a quick pseudo training...
        max_epochs = 1
    else:
        max_epochs = args.max_epochs
    
    #print(len(train_data), len(test_data))
    # setup pytorch
    use_cuda = args.use_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if use_cuda else "cpu")

    # create model and optimizer, we use plain SGD with momentum
    model = Net().to(args.device)
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    # find model filename for last run, and next free filename for model
    model_cnt = 0
    new_model_file = os.path.join(CNN_MODEL_NAME + str(model_cnt) + '.model')
    last_model_file = None
    while os.path.exists(new_model_file):
        model_cnt += 1
        last_model_file = new_model_file
        new_model_file = os.path.join(CNN_MODEL_NAME + str(model_cnt) + '.model')
    if load_model is None or not load_model:  # let's train
                # setup our datasets for training, evaluation and testing
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {'num_workers': 0}
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=args.batch_size, shuffle=False, **kwargs)
        
        
        best_valid_loss = 9999.  # let's init it with something really large - loss will be << 1 usually
        cur_patience = args.patience  # we keep track of our patience here.
        print('Training CNN...')
        start_t = get_time()
        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
        
        for epoch in range(1,max_epochs+1):
            train_loss, train_acc = train_epoch_cnn(model, train_loader, optimizer, args)
            print('Trainig took: {:.2f}s for {} epochs'.format(get_time()-start_t, epoch))

            print('Testing...')
            # Call test_cnn using the test_loader to calculate the loss on the test dataset
            # YOUR CODE HERE
            #2
            valid_loss, valid_acc = test_cnn(model, valid_loader, args)
            
            
            #raise NotImplementedError()
            if valid_loss < best_valid_loss:
                if smoke_test == False:
                    torch.save(model.state_dict(), new_model_file)
                    best_valid_loss = valid_loss
                    cur_patience = cur_patience
            else:
                if cur_patience <= 0:
                    print("training stopped")
                    break
                else:
                    print("patience left")
                    cur_patience = cur_patience-1
            
            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch,
                                                                                                             max_epochs,
                                                                                                             train_loss,
                                                                                                             valid_loss,
                                                                                                             train_acc,
                                                                                                             valid_acc))
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(valid_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(valid_acc)
            
            
            

    else:  # we should just load a model...
        if not isinstance(load_model, str) or not os.path.exists(load_model):
            load_model = last_model_file
        if load_model is None or not os.path.exists(load_model):
            print('Model file not found, unable to load...')
        else:
            model.load_state_dict(torch.load(load_model, map_location=args.device))
            print("Model file loaded: {}".format(load_model))
            return model
        
    return model, history


# In[6]:


# This function reduces the dataset size (ie., keeps only 30 % of the original data) based on the original target distribution
def reduce_dataset_size(annotation_folder, dataset_size = 0.3):
    annotations = pd.read_csv(annotation_folder, header = None)
    label = []
    aud_path = []
    for ann in annotations[0]:
        aud_path.append(ann.split()[0])
        label.append(ann.split()[-1])

    df = pd.DataFrame(list(zip(aud_path, label)),columns =['audio_path', 'label'])
    x1, x2, y1, y2 = sklearn.model_selection.train_test_split(df.audio_path, df.label,
                                                    test_size=dataset_size,
                                                    random_state=0,shuffle = True,
                                                    stratify=df.label)
    dfshort = pd.DataFrame(list(zip(x2, y2)),columns =['audio_path', 'label'])
    d = dfshort.audio_path+'\t'+ dfshort.label
    return d


# In[ ]:


foldperf={}
fold = 1
for annot_train, annot_eval in zip(sorted(annot_dev_train_paths), sorted(annot_dev_eval_paths)):
    print('--------'+str(fold)+'_out of 4-fold-----------------')
    annot_train = reduce_dataset_size(annot_train)
    annot_eval = reduce_dataset_size(annot_eval)
    # setup our datasets for training, evaluation and testing
    train_data = DDataset(annot_train, development_audio_path)
    test_data = DDataset(annot_eval, development_audio_path)
    # tests
    #model, train_loss, test_loss, train_acc, test_acc = train_network(smoke_test=True)  # just a quick run to check everything is working
    model, history = train_network() 

    foldperf['fold{}'.format(fold)] = history  
    fold = fold+1

    a_file = open("foldperf.pkl", "wb")
    pickle.dump(foldperf, a_file)
    a_file.close()


# In[ ]:


a_file = open("foldperf.pkl", "rb")
output = pickle.load(a_file)
foldperf = output


# In[ ]:


testl_f,tl_f,testa_f,ta_f=[],[],[],[]
k=1
for f in range(1,k+1):

     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))


# In[ ]:


diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}

for i in range(9):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))
      diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
      diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(k)]))


# In[ ]:


# Plot losses
plt.figure(figsize=(10,8))
plt.plot(diz_ep['train_loss_ep'], label='Train')
plt.plot(diz_ep['test_loss_ep'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.grid()
plt.legend()
plt.title('Average model loss across 4 folds')
plt.show()


# In[ ]:


# Plot accuracies
plt.figure(figsize=(10,8))
plt.plot(diz_ep['train_acc_ep'], label='Train')
plt.plot(diz_ep['test_acc_ep'], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
#plt.grid()
plt.legend()
plt.title('Average model accuracy across 4 folds')
plt.show()


# # Results and Conclusion

# In[ ]:


# Put the evaluation code on the evaulation dataset in these code cells.
# You can add additional cells below this one for structuring the notebook.
# Feel free to add markdown cells / plots / tests / etc. if it helps your presentation.

# YOUR CODE HERE
#raise NotImplementedError()
annot_final_eval = reduce_dataset_size(evaluation_annotation_file, dataset_size = 0.3)
final_eval_data = DDataset(annot_final_eval, evaluation_audio_path)
print(len(final_eval_data))
# setup our datasets for training, evaluation and testing
use_cuda = DEFAULT_ARGS.use_cuda and torch.cuda.is_available()
torch.manual_seed(DEFAULT_ARGS.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {'num_workers': 0}
eval_loader = torch.utils.data.DataLoader(final_eval_data,
                                           batch_size=32, shuffle=False, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # set model to inference mode (deactivate dropout layers for example)
test_loss = 0  # init overall loss
test_correct = 0
acc_scores = []

from sklearn.metrics import confusion_matrix
# iterate over test data
y_pred = []
y_true = []
with torch.no_grad():  # do not calculate gradients since we do not want to do updates
    for data, target in eval_loader:  # iterate over test data
        target = target.squeeze_()
        data, target = data.to(device), target.to(device)  # move data to device 
        output = model(data) # forward pass
        test_loss += torch_func.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1)[1].cpu()
        labels = target.to('cpu').numpy()
        y_true.extend(labels) # Save Truth
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        acc_scores.append(sklearn.metrics.accuracy_score(target.to('cpu').numpy(), pred))
test_loss /= len(eval_loader.dataset)  # calc mean loss
acc_score = sum(acc_scores) / len(acc_scores) 
print("On evaluation set:\n Loss: {:.2f}, \n Accuracy:{:.2f}".format(test_loss,acc_score))


# In[ ]:


y_true_labels = le.inverse_transform(y_true)
y_pred_labels = le.inverse_transform(y_pred)
# constant for classes
classes = class_names

# Build confusion matrix
cf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
plt.savefig('output.png')


# In[ ]:


# Example of distribution of labels before and after reducing the dataset size

annotations = pd.read_csv(annot_dev_train_paths[0], header = None)
label = []
aud_path = []
for ann in annotations[0]:
    aud_path.append(ann.split()[0])
    label.append(ann.split()[-1])

df = pd.DataFrame(list(zip(aud_path, label)),columns =['audio_path', 'label'])
df = df.sort_values(by = ['label'])

ax = sns.countplot(x="label", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title('original dataset distribution')
plt.show()

x1, x2, y1, y2 = sklearn.model_selection.train_test_split(df.audio_path, df.label,
                                                    test_size=0.3,
                                                    random_state=0, shuffle = True,
                                                    stratify=df.label)
dfshort = pd.DataFrame(list(zip(x2, y2)),columns =['audio_path', 'label'])
dfshort = dfshort.sort_values(by = ['label'])

ax = sns.countplot(x="label", data=dfshort)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.title('shortened dataset distribution')
plt.tight_layout()
plt.show()

