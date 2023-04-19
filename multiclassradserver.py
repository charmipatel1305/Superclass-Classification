import torch
import keras
from keras.models import load_model

model = load_model('RadImageNet-ResNet50_notop.h5')


model.summary()
# # Print the weights of the model
# for layer in model.layers:
#     print(layer.get_weights())
keras_weights = model.get_weights()
print(keras_weights)
import tensorflow as tf
# Loop through the layers and print the layer name and weights
# for i, layer in enumerate(model.layers):
#      print("Layer %d:" % (i+1), layer.name, "Weights:", layer.get_weights())

# # Loop through the layers and print them with numbers and weight shapes
# for i, layer in enumerate(model.layers):
#     for weight in layer.get_weights():
#         print(f'Layer Name: {layer.name}, Weight size: {weight.shape}')
    
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



keras_weights = model.get_weights()
# print(keras_weights)
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

# Remove square brackets from list
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
resnet_model1 = torchvision.models.resnet50(pretrained=False) #implemented based on the previous model (by myself)
new_state_dict = {}
with torch.no_grad():
    y = 1
    for i, layer in enumerate(resnet_model1.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            # extract the weights and biases from the TensorFlow weights
            weight_tf_conv = keras_weights[(2*y)-2][0]
            # print(weight_tf_conv)
            
            layer.weight.data = torch.tensor(weight_tf_conv.transpose(), dtype=torch.float)
            print(layer.weight.data)
            # print(layer)

        if isinstance(layer, torch.nn.BatchNorm2d):
            weights_tf_batch = keras_weights[(2*y)-1][0]
            bias_tf_batch = keras_weights[(2*y)-1][1]
            layer.weight.data= torch.tensor(weights_tf_batch, dtype=torch.float)
            layer.bias.data = torch.tensor(bias_tf_batch, dtype=torch.float)
            y = y+1


resnet_model1.load_state_dict(new_state_dict, strict=False)

# Save the updated state dict to a file
torch.save(resnet_model1.state_dict(), 'RadImageNet.pt')

for name, param in resnet_model1.named_parameters():
    if 'weight' or  'bias' in name:
        print(f'Layer Name: {name}, Weight size: {param.size()}')

#freeze all layers except fc
for name, param in resnet_model1.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# Modify the first convolutional layer to accept images of size 256x256
resnet_model1.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

num_ftrs = resnet_model1.fc.in_features
resnet_model1.fc = torch.nn.Linear(num_ftrs, 8)
# Print the PyTorch model weights
for name, param in resnet_model1.named_parameters():
    print(name)
    print(param)
print("Model's state_dict:")
for param_tensor in resnet_model1.state_dict():
    print(param_tensor, "\t", resnet_model1.state_dict()[param_tensor].size())
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
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout
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

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch CUDA Version is", torch.version.cuda)
print("Whether CUDA is supported by our system:", torch.cuda.is_available())

file_pathimages = './data/images/images/image_list.pkl'
with open(file_pathimages, 'rb') as f:
    images = pickle.load(f)

with open('superclass_labels.pkl', 'rb') as f:
    superclass_labels = pickle.load(f)

with open('subclass_labels.pkl', 'rb') as f:
    subclass_labels = pickle.load(f)
    

#split data into train/val/test set
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(images, superclass_labels, subclass_labels, test_size=0.3, stratify=subclass_labels, shuffle=True)
x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train, y_train, z_train, test_size=0.3, stratify=z_train, shuffle=True)

unique_labels = np.unique(subclass_labels)
# count the number of samples in each set for each class
train_counts = [np.sum(z_train== label) for label in unique_labels]
val_counts = [np.sum(z_val == label) for label in unique_labels]
test_counts = [np.sum(z_test == label) for label in unique_labels]

# print the counts for each class in each set
for i, label in enumerate(unique_labels):
    print(f"Class {label}: Train={train_counts[i]}, Val={val_counts[i]}, Test={test_counts[i]}")
import torch
from torch.utils.data import Dataset, DataLoader

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

subclass_labels = torch.tensor(z_train)
subclasses = torch.unique(subclass_labels)
subclass_freqs = []

for subclass in subclasses:
    subclass_counts = sum(subclass_labels == subclass)
    subclass_freqs.append(1/subclass_counts)

subclass_weights = torch.zeros_like(subclass_labels).float()

for idx, label in enumerate(subclass_labels):
    subclass_weights[idx] = subclass_freqs[int(label)]

# # Create a sampler to handle imbalanced data
# class_counts = np.bincount(z_train)
# weights = 1 / class_counts[z_train]
# weights = torch.FloatTensor(weights)
# sampler = WeightedRandomSampler(weights, len(weights))
sampler = WeightedRandomSampler(subclass_weights, len(subclass_weights))
# # Create a sampler for validation data
val_sampler = torch.utils.data.sampler.SequentialSampler(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
# optimizer and loss function

# define cnn model
model = resnet_model1

# define optimizer
optimizer = Adam(model.fc.parameters(), lr = 0.0001)

# define loss function
criterion = CrossEntropyLoss()
# print(model)
#train the model
def train(epoch):
    total = 0
    correct = 0
    tr_loss=0
    model.train()
    
    nb_classes = 2
    num_subclasses = 8
    correct1 = 0
    correct2 = 0
    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    
    #create confusion matrix
    true_labels = []
    predicted_labels = []

    for i, data in enumerate(train_loader):
        inputs, labels, subclass = data
        inputs = inputs
        subclass = subclass
        labels = labels
        # print(inputs)
        # print(subclass)
        
        # predcition for training set
        outputs = model(inputs)
        # print(outputs)
        
        y_pred = outputs.argmax(axis=1)
        # print(y_pred)
        y_2 = torch.zeros(len(y_pred))
        mask = y_pred <=3
        y_2[mask] = 0
        y_2[~mask] = 1
        y_2 = y_2.int()
        y_2 = y_2
        # print(y_2)
        total += subclass.size(0)
        # print(total)
        # for t, p in zip(labels.view(-1), y_2.view(-1)):
        #         confusion_matrix[t.long(), p.long()] += 1
        
        # clearing the gradients of the model parameters
        optimizer.zero_grad()
        
        # compute loss
        q = torch.tensor([])
        eta = 0.01
        normalize_loss=False
        batch_size = inputs.shape[0]
        
        if len(q) == 0:
            q = torch.ones(num_subclasses)
            q /= q.sum()

        losses = torch.zeros(num_subclasses)

        subclass_counts = torch.zeros(num_subclasses)
        
        # computes gDRO loss
        # get relative frequency of samples in each subclass
        for subclasses in range(num_subclasses):
            subclass_idx = subclass == subclasses
            subclass_counts[subclasses] = torch.sum(subclass_idx)

            # only compute loss if there are actually samples in that class
            if torch.sum(subclass_idx) > 0:
                losses[subclasses] = criterion(outputs[subclass_idx], labels[subclass_idx])

        # update q
        if model.training:
            q *= torch.exp(eta * losses.data)
            q /= q.sum()

        if normalize_loss:
            losses *= subclass_counts
            loss = torch.dot(losses, q)
            loss /= batch_size
            loss *= num_subclasses
        else:
            loss = torch.dot(losses, q)
        
        # # compute loss
        # loss_train = criterion(outputs, subclass)
        # # print(loss_train)
        
        # compute updates weights of all the parameters
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        # Calculate training loss value
        train_loss_value = tr_loss/len(train_loader)
        
        for subclasses in range(num_subclasses):
            subclass_idx = subclass == subclasses
            num_samples[subclasses] += torch.sum(subclass_idx)
            subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
            torch.float).sum().item()

        superclass_accuracy = subgroup_correct / num_samples
        
        correct1 += (y_pred == subclass).sum().item()
        correct2 += (y_2 == labels).sum().item()
        overall_subgroup = correct1 / total
        overall_superclass = correct2/total
        
        # Append true and predicted labels to lists
        true_labels.append(subclass.numpy())
        predicted_labels.append(y_pred.numpy())

    # Concatenate the true and predicted labels to get the confusion matrix
    true_labels = np.concatenate(true_labels)
    predicted_labels = np.concatenate(predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # calculate the accuracy of each class
    subgroup_accuracy = np.diag(cm) / np.sum(cm, axis=1)


    # print(subgroup_accuracy) 
    print("Epoch: {:.3f}, Subclass training loss: {:.3f}, Train accuracy: {:.3f}".format(epoch+1, train_loss_value, overall_subgroup)) 
    # print('confusion matrix of training images: {}'.format(confusion_matrix))
    # print("Train Accuracy over subgroups:", subgroup_accuracy, "\nTrain Accuracy over superclass:", superclass_accuracy, "\nWorst Group Accuracy:",min(subgroup_accuracy))
    
    return train_loss_value, overall_subgroup, overall_superclass, subgroup_accuracy, superclass_accuracy, cm
def val(epoch):
    
    model.eval()
    nb_classes = 2

    num_subclasses = 8
    correct1 = 0
    correct2 = 0
    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    
    #create confusion matrix
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        total = 0
        correct = 0
        tr_loss = 0
        for i, data in enumerate(val_loader):
            inputs, labels,subclass = data
            inputs = inputs
            subclass = subclass
            # print(labels)
            # print(subclass)
            
            outputs = model(inputs)
            # print(outputs)
            y_pred = outputs.argmax(axis=1)
            # print(y_pred)
            y_2 = torch.zeros(len(y_pred))
            mask = y_pred <=3
            y_2[mask] = 0
            y_2[~mask] = 1
            y_2 = y_2.int()
            # print(y_2)

            total += subclass.size(0)
            # print(total)
            
            loss_train = criterion(outputs, subclass)
            tr_loss += loss_train.item()
            val_loss_value = tr_loss/len(test_loader)
            
            for subclasses in range(num_subclasses):
                subclass_idx = subclass == subclasses
                num_samples[subclasses] += torch.sum(subclass_idx)
                subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
                    torch.float).sum().item()

            superclass_accuracy = subgroup_correct / num_samples
            
            # for t, p in zip(subclass.view(-1), y_2.view(-1)):
            #         confusion_matrix[t.long(), p.long()] += 1
            
            correct1 += (y_pred == subclass).sum().item()
            correct2 += (y_2 == labels).sum().item()
            overall_subgroup = correct1 / total
            overall_superclass = correct2/total
            
            # Append true and predicted labels to lists
            true_labels.append(subclass.numpy())
            predicted_labels.append(y_pred.numpy())

        # Concatenate the true and predicted labels to get the confusion matrix
        true_labels = np.concatenate(true_labels)
        predicted_labels = np.concatenate(predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # calculate the accuracy of each class
        subgroup_accuracy = np.diag(cm) / np.sum(cm, axis=1)
            
        print("Epoch: {:.3f}, Subclass val loss: {:.3f}, Val accuracy: {:.3f}".format(epoch+1, val_loss_value, overall_subgroup)) 
        # print('confusion matrix of validation images: {}'.format(confusion_matrix)) 
        # print("Val Accuracy over subgroups:", subgroup_accuracy, "\nVal Accuracy over super class:", superclass_accuracy, "\nWorst Group Accuracy:",
        #    min(subgroup_accuracy))       
    
        return val_loss_value, overall_subgroup, overall_superclass, subgroup_accuracy, superclass_accuracy, cm
def test(epoch):
    
    model.eval()
    nb_classes = 2
    #create confusion matrix
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        total = 0
        correct1 = 0
        correct2 = 0
        tr_loss = 0
        num_subclasses = 8
        num_samples = np.zeros(num_subclasses)
        subgroup_correct = np.zeros(num_subclasses)
        for i, data in enumerate(test_loader):
            inputs, labels,subclass = data
            inputs = inputs
            subclass = subclass
            # print(labels)
            # print(subclass)
            
            # predcition for training set
            outputs = model(inputs)
            # print(outputs)
            y_pred = outputs.argmax(axis=1)
            # print(y_pred)
            y_2 = torch.zeros(len(y_pred))
            mask = y_pred <=3
            y_2[mask] = 0
            y_2[~mask] = 1
            y_2 = y_2.int()
            # print(y_2)
            total += subclass.size(0)
            # print(total)
            
            loss_train = criterion(outputs, subclass)
            tr_loss += loss_train.item()
            test_loss_value = tr_loss/len(test_loader)
            
            for subclasses in range(num_subclasses):
                subclass_idx = subclass == subclasses
                num_samples[subclasses] += torch.sum(subclass_idx)
                subgroup_correct[subclasses] += (y_2[subclass_idx] == labels[subclass_idx]).type(
                    torch.float).sum().item()

            superclass_accuracy = subgroup_correct / num_samples            
            # for t, p in zip(labels.view(-1), y_2.view(-1)):
            #         confusion_matrix[t.long(), p.long()] += 1

            correct1 += (y_pred == subclass).sum().item()
            correct2 += (y_2 == labels).sum().item()
            overall_subgroup = correct1 / total
            overall_superclass = correct2/total
            
            # Append true and predicted labels to lists
            true_labels.append(subclass.numpy())
            predicted_labels.append(y_pred.numpy())

        # Concatenate the true and predicted labels to get the confusion matrix
        true_labels = np.concatenate(true_labels)
        predicted_labels = np.concatenate(predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # calculate the accuracy of each class
        subgroup_accuracy = np.diag(cm) / np.sum(cm, axis=1)
            
        print("Epoch: {:.3f}, Subclass test Loss: {:.3f}, Test accuracy: {:.3f},".format(epoch+1, test_loss_value, overall_subgroup)) 
        # print('confusion matrix of testing images: {}'.format(confusion_matrix)) 
        # print("Test Accuracy over subgroup:", subgroup_accuracy, "\nTest Accuracy over superclass: ", superclass_accuracy, "\nWorst Group Accuracy:",
            #   min(subgroup_accuracy)) 
        return test_loss_value, overall_subgroup, overall_superclass, subgroup_accuracy, superclass_accuracy, cm
# number of epochs
n_epochs = 10
tr_loss = []
tr_accuracies = []
val_loss = []
val_accuracies = []
test_loss = []
test_accuracies = []
max_worst_accuracy = 0

counter=0
patience = 10
best_val_loss = float('inf')
num_trials = 1
df1 = pd.DataFrame(columns=['trial', 'subtype', 'Train ERM subgroup accuracy', 'Train ERM superclass accuracy'])
df2 = pd.DataFrame(columns=['trial', 'subtype', 'Val ERM subgroup accuracy', 'Val ERM superclass accuracy'])
df3 = pd.DataFrame(columns=['trial', 'subtype', 'Test ERM subgroup accuracy', 'Test ERM superclass accuracy'])
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
        
        train_loss, trainoverall_subgroup, trainoverall_superclass, trainsubgroup_accuracy, trainsuperclass_accuracy, trainconfusion_matrix = train(epoch)
        val_loss, valoverall_subgroup, valoverall_superclass, valsubgroup_accuracy, valsuperclass_accuracy, valconfusion_matrix = val(epoch)
        test_loss, testoverall_subgroup, testoverall_superclass, testsubgroup_accuracy, testsuperclass_accuracy, testconfusion_matrix = test(epoch)
        tr_accuracies.append(trainoverall_subgroup)
        val_accuracies.append(valoverall_subgroup)
        test_accuracies.append(testoverall_subgroup)
        
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping after {epoch} epochs.")
        #     break
        print("For Trial:",i, "Train subgroup accuracy:", trainoverall_subgroup, "Val subgroup accuracy:", valoverall_subgroup, "Test subgroup accuracy:", testoverall_subgroup,"\nTest Accuracy over each subgroups:", testsubgroup_accuracy, "\ntest Worst Group Accuracy:",min(testsubgroup_accuracy))
    # append accuracy values for each subgroup to the dataframes
    for j, trainacc_sub, trainacc_super in zip(subgroups, trainsubgroup_accuracy, trainsuperclass_accuracy):
        df1 = df1.append({'trial': i, 'subtype': j, 'Train ERM subgroup accuracy': trainacc_sub, 'Train ERM superclass accuracy': trainacc_super}, ignore_index=True)
    
    # add overall accuracy to Train  accuracy for each trial
    df1 = df1.append({'trial': i, 'subtype': 'overall', 'Train ERM subgroup accuracy': trainoverall_subgroup, 'Train ERM superclass accuracy': trainoverall_superclass}, ignore_index=True)
    
    for j, valacc_sub, valacc_super in zip(subgroups, valsubgroup_accuracy, valsuperclass_accuracy):
        df2 = df2.append({'trial': i, 'subtype': j, 'Val ERM subgroup accuracy': valacc_sub, 'Val ERM superclass accuracy': valacc_super}, ignore_index=True)
    
    # add overall accuracy to Val accuracy for each trial
    df2 = df2.append({'trial': i, 'subtype': 'overall', 'Val ERM subgroup accuracy': valoverall_subgroup, 'Val ERM superclass accuracy': valoverall_superclass}, ignore_index=True)
    
    for j, testacc_sub, testacc_super in zip(subgroups, testsubgroup_accuracy, testsuperclass_accuracy):
        df3 = df3.append({'trial': i, 'subtype': j, 'Test ERM subgroup accuracy': testacc_sub, 'Test ERM superclass accuracy': testacc_super}, ignore_index=True)
    
    # add overall accuracy to Test accuracy for each trial
    df3 = df3.append({'trial': i, 'subtype': 'overall', 'Test ERM subgroup accuracy': testoverall_subgroup, 'Test ERM superclass accuracy': testoverall_superclass}, ignore_index=True)



     
    # print("For Trial:",i,"Train Accuracy:", temptrain_acc, "\nTrain Accuracy over each subgroups:", trainsubgroup_acc, "\nTrain Worst Group Accuracy:",min(trainsubgroup_acc))
    # print("For Trial:",i,"Val Accuracy:", tempval_acc, "\nVal Accuracy over each subgroups:", valsubgroup_acc, "\nVal Worst Group Accuracy:",min(valsubgroup_acc))
    print("For Trial:",i, "Train subgroup accuracy:", trainoverall_subgroup, "Val subgroup accuracy:", valoverall_subgroup, "Test subgroup accuracy:", testoverall_subgroup,"\nTest Accuracy over each subgroups:", testsubgroup_accuracy, "\ntest Worst Group Accuracy:",min(testsubgroup_accuracy))