import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import asarray
import pickle
import torchvision
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import Linear, ReLU, BCEWithLogitsLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, Dropout
from torch.optim import Adam,SGD
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import Counter
from torch.utils.data import DataLoader, SubsetRandomSampler,Subset, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Pytorch CUDA Version is", torch.version.cuda)
print("Whether CUDA is supported by our system:", torch.cuda.is_available())
# class ImageDataset(Dataset):
#     def __init__(self, csv_file, transform=None):
#         self.data = pd.read_csv(csv_file)
#         self.data.to_string(index=False)
#         self.transform = transform
#         self.data['Super Class'] = self.data['Super Class'].apply(lambda x: 0 if x == 'benign' else 1)
#         label_map = {
#             'adenosis': 0,
#             'fibroadenoma': 1,
#             'tubular_adenoma': 2,
#             'phyllodes_tumor': 3,
#             'ductal_carcinoma': 4,
#             'lobular_carcinoma': 5,
#             'mucinous_carcinoma': 6,
#             'papillary_carcinoma': 7
#         }
#         self.data['Sub Class'] = self.data['Sub Class'].map(label_map)

        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         image_path = self.data.iloc[idx, 0]
#         image = Image.open(image_path).convert("RGB")
#         SuperClass = self.data.iloc[idx, 1]
#         # SuperClass = torch.tensor(SuperClass, dtype=torch.long)
#         SubClass = self.data.iloc[idx, 3]
#         if self.transform:
#             image = self.transform(image)
#         image = asarray(image)
#         return (image, SuperClass, SubClass)

# transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor()
# ])

# dataset = ImageDataset(csv_file='csv windows.csv', transform=transform)
# images = [x[0] for x in dataset]
# superclass_labels = [x[1] for x in dataset]
# subclass_labels = [x[2] for x in dataset]
# import pickle

# # Serialize and save the list to a file
# with open('image_list.pkl', 'wb') as f:
#     pickle.dump(images, f)

# with open('superclass_labels.pkl', 'wb') as f:
#     pickle.dump(superclass_labels, f)
    

# with open('subclass_labels.pkl', 'wb') as f:
#     pickle.dump(subclass_labels, f)

# Load the list back into memory
file_pathimages = '/home/cpatel/superclass/data/images/images/image_list.pkl'

with open(file_pathimages, 'rb') as f:
    images = pickle.load(f)
    
with open('superclass_labels.pkl', 'rb') as f:
    superclass_labels = pickle.load(f)

with open('subclass_labels.pkl', 'rb') as f:
    subclass_labels = pickle.load(f)
    
x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(images, superclass_labels, subclass_labels, test_size=0.3, stratify=superclass_labels, random_state=42)
x_train, x_val, y_train, y_val, z_train, z_val = train_test_split(x_train, y_train, z_train, test_size=0.3, stratify=y_train, random_state=42)
# get the unique labels in y
unique_labels = np.unique(subclass_labels)

# count the number of samples in each set for each class
train_counts = [np.sum(z_train == label) for label in unique_labels]
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

train_loader = DataLoader(train_dataset, batch_size=16)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torchvision
# model = torchvision.models.resnet50(weights=True)

# lt=10
# cntr = 0
# for child in model.children():
#     cntr+=1

#     if cntr < lt:
#         for param in child.parameters():
#             param.requires_grad = False

# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(in_features = num_ftrs, out_features = 1, bias=True)
# # optimizer and loss function

# # define cnn model
# model = model.to(device)

# # print(model)

# # define optimizer
# optimizer = Adam(model.parameters(), lr = 0.0001)

# # define loss function
# criterion = BCEWithLogitsLoss()
# # print(model)
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
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # predcition for training set
        outputs = model(inputs)
        # print(outputs)
        
        outputs = outputs.flatten()
        total += labels.size(0)
        # print(total)
        
        y_2 = torch.zeros(len(outputs))
        y_2[outputs>=0.0] = 1
        y_2= y_2.int()
        y_2 = y_2.to(device)
        
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
    
    return train_accuracy, subgroup_accuracy, train_loss_value
        
def val(epoch):
    
    model.eval()
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    num_subclasses = 8
    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    subgroup_correct_total = 0
    val_loss = 0

    with torch.no_grad():
        total = 0
        correct = 0
        tr_loss = 0
        for i, data in enumerate(val_loader):
            inputs, labels,subclass = data
            inputs = inputs.float()
            labels = labels.float()
            # print(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # predcition for training set
            outputs = model(inputs)
            # print(outputs)
            
            outputs = outputs.flatten()
            total += labels.size(0)
            # print(total)
            
            loss_val = criterion(outputs, labels)
            val_loss += loss_val.item()
            val_loss_value = val_loss/len(val_loader)
            
            y_2 = torch.zeros(len(outputs))
            y_2[outputs>=0.0] = 1
            y_2 = y_2.int()
            y_2 = y_2.to(device)
            
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
    
        return val_accuracy, subgroup_accuracy, val_loss_value
        
        
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
            inputs = inputs.to(device)
            labels = labels.to(device)

            # print(labels)
            
            # predcition for training set
            outputs = model(inputs)
            # print(outputs)
            
            outputs = outputs.flatten()
            total += labels.size(0)
            # print(total)
            
            loss_test = criterion(outputs, labels)
            test_loss += loss_test.item()
            test_loss_value = test_loss/len(test_loader)
            
            y_2 = torch.zeros(len(outputs))
            y_2[outputs>=0.0] = 1
            y_2 = y_2.int()
            y_2 = y_2.to(device)
            
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
   
        return test_accuracy, subgroup_accuracy, test_loss_value
        
        
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
num_trials = 30
df1 = pd.DataFrame(columns=['trial', 'epochs', 'subtype', 'Train ERM accuracy', 'Train ERM loss'])
df2 = pd.DataFrame(columns=['trial', 'epochs', 'subtype', 'Val ERM accuracy', 'Val ERM loss'])
df3 = pd.DataFrame(columns=['trial', 'epochs', 'subtype', 'Test ERM accuracy', 'Test ERM loss'])
minmax_acc_test = pd.DataFrame(columns=['trial', 'epochs', 'Min_subtype', 'Min subclass accuracy', 'Max_subtype', 'Max subclass accuracy'])

subgroups = ['adenosis',
            'fibroadenoma',
            'tubular_adenoma',
            'phyllodes_tumor',
            'ductal_carcinoma',
            'lobular_carcinoma',
            'mucinous_carcinoma',
            'papillary_carcinoma']

# training the model
for i in range(num_trials):
    model = torchvision.models.resnet50(weights=True)

    lt=10
    cntr = 0
    for child in model.children():
        cntr+=1

        if cntr < lt:
            for param in child.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(in_features = num_ftrs, out_features = 1, bias=True)
        
    # optimizer and loss function

    # define cnn model
    model = model.to(device)

    # print(model)

    # define optimizer
    optimizer = Adam(model.parameters(), lr = 0.0001)

    # define loss function
    criterion = BCEWithLogitsLoss()
    # print(model)
    
    for epoch in range(n_epochs):
        temptrain_acc, trainsubgroup_acc, train_loss = train(epoch)
        tempval_acc, valsubgroup_acc, val_loss = val(epoch)
        temptest_acc, testsubgroup_acc, test_loss = test(epoch)
        
        test_accuracies.append(temptest_acc)

        
        min_val = min(testsubgroup_acc)
        min_index = testsubgroup_acc.argmin()
        max_val = max(testsubgroup_acc)
        max_index = testsubgroup_acc.argmax()
        minmax_acc_test = minmax_acc_test.append({'trial': i, 'epochs': epoch, 'Min_subtype': min_index, 'Min subclass accuracy': min_val, 'Max_subtype': max_index, 'Max subclass accuracy': max_val}, ignore_index=True)
        
        # append accuracy values for each subgroup to the dataframes
        for j, acc in zip(subgroups, trainsubgroup_acc):
            df1 = df1.append({'trial': i, 'epochs': epoch, 'subtype': j, 'Train ERM accuracy': acc, 'Train ERM loss': train_loss}, ignore_index=True)
        
        # add overall accuracy to Train gDRO accuracy for each trial
        df1 = df1.append({'trial': i, 'epochs': epoch, 'subtype': 'overall', 'Train ERM accuracy': temptrain_acc, 'Train ERM loss': train_loss}, ignore_index=True)
        
        for j, acc in zip(subgroups, valsubgroup_acc):
            df2 = df2.append({'trial': i , 'epochs': epoch, 'subtype': j, 'Val ERM accuracy': acc, 'Val ERM loss': val_loss}, ignore_index=True)
        
        # add overall accuracy to Val gDRO accuracy for each trial
        df2 = df2.append({'trial': i, 'epochs': epoch, 'subtype': 'overall', 'Val ERM accuracy': tempval_acc, 'Val ERM loss': val_loss}, ignore_index=True)
        
        for j, acc in zip(subgroups, testsubgroup_acc):
            df3 = df3.append({'trial': i , 'epochs': epoch, 'subtype': j, 'Test ERM accuracy': acc, 'Test ERM loss': test_loss,},  ignore_index=True)
            
        # add overall accuracy to Test gDRO accuracy for each trial
        df3 = df3.append({'trial': i,'epochs': epoch, 'subtype': 'overall', 'Test ERM accuracy': temptest_acc, 'Test ERM loss': test_loss}, ignore_index=True)

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
# assuming that df1 is your data frame
df1.to_csv('Train_bc-erm_imagenet.csv', index=False)
df2.to_csv('Val_bc-erm_imagenet.csv', index=False)
df3.to_csv('Test_bc-erm_imagenet.csv', index=False)
minmax_acc_test.to_csv('MinMax_acc.csv', index=False)