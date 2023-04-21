#import libraries

import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import Linear, ReLU, BCEWithLogitsLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout
from torch.optim import Adam,SGD,Adagrad
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import Counter
from torch.utils.data import DataLoader, SubsetRandomSampler,Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown
import zipfile

path = './/data//images//'  # give the path of the directory in which we would like to extract the data

isExist = os.path.exists(path)
if isExist:
    print("data already present")
    
else:
    print("creating data folder")
    # Create a new directory because it does not exist
    os.makedirs(path)
    
data_path = './data/images/'

isExist = os.path.exists(data_path)

gdd.download_file_from_google_drive(file_id='14PNJnjSUJ0FxbqJG8xvSoZSHTc-eHIPT',  #follow step 2 to find the file_id and dest_path 
                                    dest_path='./data/image_list.zip',
                                    unzip=True)                                   #unzip=true means we can see the progress of the data downloaded from the google drive in the code editor

## Fetch data from Google Drive 
# Root directory for the dataset
data_root = './data/images/' 
# Path to pkl with the images
dataset_folder = f'{data_root}images/' #this will create a new folder name "download" and the dataset of the folder will be extracted in the download folder
# URL for the pkl 
url = 'https://drive.google.com/uc?id=14PNJnjSUJ0FxbqJG8xvSoZSHTc-eHIPT' #copy this URL and replace the file ID with the other file ID that would be needed to extract the data

# Path to download the pkl
download_path = f'{data_root}/image_list.zip'

# Create required directories 
if not os.path.exists(data_root):
    os.makedirs(data_root)
    os.makedirs(dataset_folder)

# Download the pkl from google drive
gdown.download(url, download_path, quiet=False)

# Unzip the downloaded file 
with zipfile.ZipFile(download_path, 'r') as ziphandler:  #this unzip the folder once downlaoded 
    ziphandler.extractall(dataset_folder)
    
# Load the images/superclass labels/ subclass labels back into memory
file_pathimages = './data/images/images/image_list.pkl'
with open(file_pathimages, 'rb') as f:
    images = pickle.load(f)

with open('superclass_labels.pkl', 'rb') as f:
    superclass_labels = pickle.load(f)

with open('subclass_labels.pkl', 'rb') as f:
    subclass_labels = pickle.load(f)
    
#check if cuda is supported to the system
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#check the version of cuda
print("Pytorch CUDA Version is", torch.version.cuda)
print("Whether CUDA is supported by our system:", torch.cuda.is_available())
    
#split data into train/val/test set
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(images, superclass_labels, subclass_labels, test_size=0.3, stratify=subclass_labels, shuffle=True)
x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train, y_train, z_train, test_size=0.3, stratify=z_train, shuffle=True)
## images count for train/val/test 
# get the unique labels in y
unique_labels = np.unique(superclass_labels)

# count the number of samples in each set for each class
train_counts = [np.sum(y_train== label) for label in unique_labels]
val_counts = [np.sum(y_val == label) for label in unique_labels]
test_counts = [np.sum(y_test == label) for label in unique_labels]

# print the counts for each class in each set
for i, label in enumerate(unique_labels):
    print(f"Class {label}: Train={train_counts[i]}, Val={val_counts[i]}, Test={test_counts[i]}")
#create dataloader for train/val/test dataset
class YourDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

train_dataset = YourDataset(x_train, y_train, z_train)
val_dataset = YourDataset(x_val, y_val, z_val)
test_dataset = YourDataset(x_test, y_test, z_test)

# #use of gdro loss
# subclass_labels = torch.tensor(z_train)
# subclasses = torch.unique(subclass_labels)
# subclass_freqs = []

# for subclass in subclasses:
#     subclass_counts = sum(subclass_labels == subclass)
#     subclass_freqs.append(1/subclass_counts)

# subclass_weights = torch.zeros_like(subclass_labels).float()

# for idx, label in enumerate(subclass_labels):
#     subclass_weights[idx] = subclass_freqs[int(label)]



# # Create a sampler to handle imbalanced data for multiclass
# class_counts = np.bincount(z_train)
# weights = 1 / class_counts[z_train]
# weights = torch.FloatTensor(weights)
# sampler = WeightedRandomSampler(weights, len(weights))
#sampler = WeightedRandomSampler(subclass_weights, len(subclass_weights))
# # Create a sampler for validation data
#val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
import torch
import keras
import tensorflow as tf
from keras.models import load_model

model = load_model('RadImageNet-ResNet50_notop.h5')
keras_weights = model.get_weights()
# print(keras_weights)

layer_names = []

for i, layer in enumerate(model.layers):
    if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization)):
          layer_names.append(layer.name)

        #   print(layer.name)
# print(layer_names)

# Loop through the layers and create a dictionary mapping layer names to their associated weights
weights_dict = {layer.name: layer.get_weights() for layer in model.layers if isinstance(layer, tf.keras.layers.Layer) and layer.weights}

# Print the weights_dict to check its contents
# print(weights_dict)


new_weights = []
new_layer_order = ['conv1_conv', 'conv1_bn', 'conv2_block1_1_conv', 'conv2_block1_1_bn', 'conv2_block1_2_conv',
'conv2_block1_2_bn','conv2_block1_3_conv','conv2_block1_3_bn','conv2_block1_0_conv','conv2_block1_0_bn',
'conv2_block2_1_conv',
'conv2_block2_1_bn',
'conv2_block2_2_conv',
'conv2_block2_2_bn',
'conv2_block2_3_conv',
'conv2_block2_3_bn',
'conv2_block3_1_conv',
'conv2_block3_1_bn',
'conv2_block3_2_conv',
'conv2_block3_2_bn',
'conv2_block3_3_conv',
'conv2_block3_3_bn',

'conv3_block1_1_conv',
'conv3_block1_1_bn',
'conv3_block1_2_conv',
'conv3_block1_2_bn',
'conv3_block1_3_conv',
'conv3_block1_3_bn',
'conv3_block1_0_conv',
'conv3_block1_0_bn',
'conv3_block2_1_conv',
'conv3_block2_1_bn',
'conv3_block2_2_conv',
'conv3_block2_2_bn',
'conv3_block2_3_conv',
'conv3_block2_3_bn',
'conv3_block3_1_conv',
'conv3_block3_1_bn',
'conv3_block3_2_conv',
'conv3_block3_2_bn',
'conv3_block3_3_conv',
'conv3_block3_3_bn',
'conv3_block4_1_conv',
'conv3_block4_1_bn',
'conv3_block4_2_conv',
'conv3_block4_2_bn',
'conv3_block4_3_conv',
'conv3_block4_3_bn',

'conv4_block1_1_conv',
'conv4_block1_1_bn',
'conv4_block1_2_conv',
'conv4_block1_2_bn',
'conv4_block1_3_conv',
'conv4_block1_3_bn',
'conv4_block1_0_conv',
'conv4_block1_0_bn',
'conv4_block2_1_conv',
'conv4_block2_1_bn',
'conv4_block2_2_conv',
'conv4_block2_2_bn',
'conv4_block2_3_conv',
'conv4_block2_3_bn',
'conv4_block3_1_conv',
'conv4_block3_1_bn',
'conv4_block3_2_conv',
'conv4_block3_2_bn',
'conv4_block3_3_conv',
'conv4_block3_3_bn',
'conv4_block4_1_conv',
'conv4_block4_1_bn',
'conv4_block4_2_conv',
'conv4_block4_2_bn',
'conv4_block4_3_conv',
'conv4_block4_3_bn',
'conv4_block5_1_conv',
'conv4_block5_1_bn',
'conv4_block5_2_conv',
'conv4_block5_2_bn',
'conv4_block5_3_conv',
'conv4_block5_3_bn',
'conv4_block6_1_conv',
'conv4_block6_1_bn',
'conv4_block6_2_conv',
'conv4_block6_2_bn',
'conv4_block6_3_conv',
'conv4_block6_3_bn',

'conv5_block1_1_conv',
'conv5_block1_1_bn',
'conv5_block1_2_conv',
'conv5_block1_2_bn',
'conv5_block1_3_conv',
'conv5_block1_3_bn',
'conv5_block1_0_conv',
'conv5_block1_0_bn',
'conv5_block2_1_conv',
'conv5_block2_1_bn',
'conv5_block2_2_conv',
'conv5_block2_2_bn',
'conv5_block2_3_conv',
'conv5_block2_3_bn',
'conv5_block3_1_conv',
'conv5_block3_1_bn',
'conv5_block3_2_conv',
'conv5_block3_2_bn',
'conv5_block3_3_conv', 'conv5_block3_3_bn']

# Rearrange the weights according to the new layer order
for layer_name in new_layer_order:
    weight = weights_dict[layer_name]
    # print(weight)
    new_weights.append(weight)

# print(new_weights[1][0])
#remove bias of convolutional layer
new_weights_wo_convbias = []

for i, weight in enumerate(new_weights):
    if i % 6 == 1:
        continue  # Skip the indices to neglect
    new_weights_wo_convbias.append(weight)

# print(new_weights_wo_convbias)

import torchvision
import tensorflow as tf
import numpy as np


# Load the weights
keras_weights = new_weights
# print(keras_weights)
# Load the PyTorch model
resnet_model = torchvision.models.resnet50(weights=False) #implemented based on the previous model (by myself)
new_state_dict = {}
with torch.no_grad():
    y = 1
    for i, layer in enumerate(resnet_model.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            # extract the weights and biases from the TensorFlow weights
            weight_tf_conv = keras_weights[(2*y)-2][0]
            # print(weight_tf_conv)
            
            layer.weight.data = torch.tensor(weight_tf_conv.transpose(), dtype=torch.float)
            # print(layer.weight.data)
            # print(layer)

        if isinstance(layer, torch.nn.BatchNorm2d):
            weights_tf_batch = keras_weights[(2*y)-1][0]
            bias_tf_batch = keras_weights[(2*y)-1][1]
            layer.weight.data= torch.tensor(weights_tf_batch, dtype=torch.float)
            layer.bias.data = torch.tensor(bias_tf_batch, dtype=torch.float)
            y = y+1


resnet_model.load_state_dict(new_state_dict, strict=False)
# Print the PyTorch model layer name and its pre-trained weights
for name, param in resnet_model.named_parameters():
    print(name)
    print(param)

# Freeze the layers
for param in resnet_model.parameters():
    param.requires_grad = False
# Modify the first convolutional layer to accept images of size 256x256
resnet_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_ftrs, 2)
# define resnet50 cnn model
# model =  resnet_model.to(device)
model =  resnet_model

# define optimizer
optimizer = Adam(model.parameters(), lr = 0.0001)

# define loss function
criterion = BCEWithLogitsLoss()
print(model)
#train the model
def train(epoch):
    total = 0
    correct = 0
    tr_loss=0
    model.train()
    
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    num_subclasses = 8
    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    subgroup_correct_total = 0

    for i, data in enumerate(train_loader):
        inputs, labels, subclass = data
        inputs = inputs.float()
        labels = labels.float()

        # print(labels)
        # print(subclass)
        # inputs = inputs.to(device)
        # labels = labels.to(device)
        # subclass = subclass.to(device)
        # predcition for training set
        outputs = model(inputs)
        # print(outputs)
        
        outputs = outputs.flatten()
        total += labels.size(0)
        # print(total)
        
        y_2 = torch.zeros(len(outputs))
        y_2[outputs>=0.0] = 1
        y_2= y_2.int()
        # y_2 = y_2.to(device)
        # print(y_2)
        
        for subclasses in range(num_subclasses):
            subclass_idx = subclass == subclasses
            num_samples[subclasses] += torch.sum(subclass_idx)
            subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
                torch.float).sum().item()

        subgroup_accuracy = subgroup_correct / num_samples
        # print(y_2)
        correct += (y_2 == labels).sum().item()
        train_accuracy = correct / total 
        # for t, p in zip(labels.view(-1), y_2.view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1
        
        # clearing the gradients of the model parameters
        optimizer.zero_grad()
        
        # compute loss
        loss_train = criterion(outputs, labels)
        # print(loss_train)
        
        # compute updates weights of all the parameters
        loss_train.backward()
        optimizer.step()
        tr_loss += loss_train.item()

    # Calculate training loss value
    train_loss_value = tr_loss/len(train_loader) 
    print("Epoch: {:.3f}, Loss: {:.3f}, Train_Accuracy: {:.3f}".format(epoch+1, train_loss_value, train_accuracy)) 
    # print('confusion matrix of training images: {}'.format(confusion_matrix))

    # print("Train Accuracy:", accuracy, "\nTrain Accuracy over subgroups:", subgroup_accuracy, "\nTrain Worst Group Accuracy:",
    #           min(subgroup_accuracy))
    
    return train_accuracy, subgroup_accuracy 
        
def val(epoch):
    
    model.eval()
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    num_subclasses = 8
    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    subgroup_correct_total = 0

    with torch.no_grad():
        total = 0
        correct = 0
        tr_loss = 0
        for i, data in enumerate(val_loader):
            inputs, labels,subclass = data
            inputs = inputs.float()
            labels = labels.float()

            # print(labels)
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            
            # predcition for training set
            outputs = model(inputs)
            # print(outputs)
            
            outputs = outputs.flatten()
            total += labels.size(0)
            # print(total)
            
            loss_train = criterion(outputs, labels)
            tr_loss += loss_train.item()
            val_loss_value = tr_loss/len(test_loader)
            
            y_2 = torch.zeros(len(outputs))
            y_2[outputs>=0.0] = 1
            y_2 = y_2.int()
            # y_2 = y_2.to(device)
            
            for subclasses in range(num_subclasses):
                subclass_idx = subclass == subclasses
                num_samples[subclasses] += torch.sum(subclass_idx)
                subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
                    torch.float).sum().item()

            subgroup_accuracy = subgroup_correct / num_samples

            for t, p in zip(labels.view(-1), y_2.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            
            correct += (y_2 == labels).sum().item()
            val_accuracy = correct / total
            
        print("Epoch: {:.3f}, Loss: {:.3f}, Val Accuracy: {:.3f}".format(epoch+1, val_loss_value, val_accuracy)) 
        # print('confusion matrix of validation images: {}'.format(confusion_matrix)) 
        # print("Val Accuracy:", accuracy, "\nVal Accuracy over subgroups:", subgroup_accuracy, "\nVal Worst Group Accuracy:",
        #       min(subgroup_accuracy))       
    
        return val_accuracy, subgroup_accuracy
        
        
def test(epoch):
    
    model.eval()
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    subgroup_correct_total = 0

    with torch.no_grad():
        total = 0
        correct = 0
        tr_loss = 0
        num_subclasses = 8
        num_samples = np.zeros(num_subclasses)
        subgroup_correct = np.zeros(num_subclasses)
        for i, data in enumerate(test_loader):
            inputs, labels,subclass = data
            inputs = inputs.float()
            labels = labels.float()
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # print(labels)
            
            # predcition for training set
            outputs = model(inputs)
            # print(outputs)
            
            outputs = outputs.flatten()
            total += labels.size(0)
            # print(total)
            
            loss_train = criterion(outputs, labels)
            tr_loss += loss_train.item()
            test_loss_value = tr_loss/len(test_loader)
            
            y_2 = torch.zeros(len(outputs))
            y_2[outputs>=0.0] = 1
            y_2 = y_2.int()
            # y_2 = y_2.to(device)
            
            for subclasses in range(num_subclasses):
                subclass_idx = subclass == subclasses
                num_samples[subclasses] += torch.sum(subclass_idx)
                subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
                    torch.float).sum().item()

            subgroup_accuracy = subgroup_correct / num_samples

            
            for t, p in zip(labels.view(-1), y_2.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            
            correct += (y_2 == labels).sum().item()
            test_accuracy = correct / total
            
        print("Epoch: {:.3f}, Loss: {:.3f}, Test Accuracy: {:.3f}".format(epoch+1, test_loss_value, test_accuracy)) 
        # print('confusion matrix of testing images: {}'.format(confusion_matrix))
        # print("Test Accuracy:", accuracy, "\nTest Accuracy over subgroups:", subgroup_accuracy, "\nTest Worst Group Accuracy:",
        #       min(subgroup_accuracy)) 
   
        return test_accuracy, subgroup_accuracy
        
# number of epochs
n_epochs = 58
tr_loss = []
tr_accuracies = []
val_loss = []
val_accuracies = []
test_loss = []
test_accuracies = []
max_worst_accuracy = 0

patience = 10
best_val_loss = float('inf')
num_trials = 1
df1 = pd.DataFrame(columns=['trial', 'subtype', 'Train ERM accuracy'])
df2 = pd.DataFrame(columns=['trial', 'subtype', 'Val ERM accuracy'])
df3 = pd.DataFrame(columns=['trial', 'subtype', 'Test ERM accuracy'])
subgroups = {'adenosis',
            'fibroadenoma',
            'tubular_adenoma',
            'phyllodes_tumor',
            'ductal_carcinoma',
            'lobular_carcinoma',
            'mucinous_carcinoma',
            'papillary_carcinoma'}

# training the model
for i in range(num_trials):
    for epoch in range(n_epochs):
        temptrain_acc, trainsubgroup_acc = train(epoch)
        tempval_acc, valsubgroup_acc = val(epoch)
        temptest_acc, testsubgroup_acc = test(epoch)
        
    # append accuracy values for each subgroup to the dataframes
    for j, acc in zip(subgroups, trainsubgroup_acc):
        df1 = df1.append({'trial': i, 'subtype': j, 'Train ERM accuracy': acc}, ignore_index=True)
    
    # add overall accuracy to Train gDRO accuracy for each trial
    df1 = df1.append({'trial': i, 'subtype': 'overall', 'Train ERM accuracy': temptrain_acc}, ignore_index=True)
    
    for j, acc in zip(subgroups, valsubgroup_acc):
        df2 = df2.append({'trial': i, 'subtype': j, 'Val ERM accuracy': acc}, ignore_index=True)
    
    # add overall accuracy to Val gDRO accuracy for each trial
    df2 = df2.append({'trial': i, 'subtype': 'overall', 'Val ERM accuracy': tempval_acc}, ignore_index=True)
    
    for j, acc in zip(subgroups, testsubgroup_acc):
        df3 = df3.append({'trial': i, 'subtype': j, 'Test ERM accuracy': acc}, ignore_index=True)
        
    # add overall accuracy to Test gDRO accuracy for each trial
    df3 = df3.append({'trial': i, 'subtype': 'overall', 'Test ERM accuracy': temptest_acc}, ignore_index=True)

        # Check if the validation loss has increased and update the best model if it hasn't
        # a =  min(valsubgroup_acc)
        # print(a)
        # if a >= max_worst_accuracy:
        #     max_worst_accuracy = a
        #     best_epoch = epoch
        #     print('I am saving the best dro model at Epoch: ', best_epoch)
        #     torch.save(model.state_dict(), r'Best_gdromodel.pth')
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience: # Stop early if the validation loss has increased for `patience` epochs
        #         break
     
    # print("For Trial:",i,"Train Accuracy:", temptrain_acc, "\nTrain Accuracy over each subgroups:", trainsubgroup_acc, "\nTrain Worst Group Accuracy:",min(trainsubgroup_acc))
    # print("For Trial:",i,"Val Accuracy:", tempval_acc, "\nVal Accuracy over each subgroups:", valsubgroup_acc, "\nVal Worst Group Accuracy:",min(valsubgroup_acc))
    print("For Trial:",i,"Test Accuracy:", temptest_acc,"\nTest Accuracy over each subgroups:", testsubgroup_acc, "\ntest Worst Group Accuracy:",min(testsubgroup_acc))