
import torch
import torchvision
import numpy as np
import time
#import cv2
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn 
import matplotlib.pyplot as plt
from collections import OrderedDict
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler
from torchvision.models import resnet50,vgg19
from PIL import Image
from typing import List
import os
n_epochs=15
class Model():
    def __init__(self,
                 fc_layer:List[int],
                 pretrained: bool, lr: int,
                 train_data_loader,
                 valid_data_loader,
                 test_data_loader,
                 device):
        self.model = vgg19(pretrained)
        
        self.classifier = nn.Sequential(OrderedDict([
                            ('cnn_to_fc', nn.Linear(2048, fc_layer[0])),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout()),
                            ('fc1', nn.Linear(fc_layer[0], fc_layer[1])),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout()),
                            ('fc2', nn.Linear(fc_layer[1], 4))
                        ]))
        
        self.patient = 5
        
        self.device = device
        self.lr = lr
        self.train_data_loader =train_data_loader
        self.valid_data_loader = valid_data_loader      
        self.test_data_loader = test_data_loader
        
        self.train_loss_list = []
        self.valid_loss_list = []
        self.train_acc_list = []
        self.valid_acc_list = []
        self.cm = np.array([[0, 0],
                            [0, 0]])
        self.test_acc = 0
        self.test_time = 0
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda epoch:self.scheduler(epoch))
        self.lr_list = []
        
        self.loss_list = {'train': [], 'valid': []}
        self.acc_list = {'train':[], 'valid': []}
                
#         for param in self.model.parameters():
#             param.requires_grad = False 
        self.model.fc = self.classifier
        
        self.model.to(self.device)
        
    @staticmethod
    def scheduler(epoch):
#         if epoch < n_epochs/5:
#             epoch_delta =  n_epochs/5 
#             m = (1e-2 - 1e-4)/epoch_delta
#             return 0.01 - m * (epoch-1)
#         else:
#             new_epoch = epoch - n_epochs/5
#             epoch_delta =  n_epochs/5 * 4
#             m = (1e-2 - 1e-6)/epoch_delta
#             return 0.01 - m * new_epoch
        return 0.01 - (1e-2-1e-6)/n_epochs * (epoch-1)

    @staticmethod
    def ROC_curve(output, labels):
        output = torch.from_numpy(output)
        o = torch.nn.Softmax(dim=1)
        p = o(output).data.cpu().numpy()
        l = np.copy(labels)

        for i in p:
            i[0] += i[2]

        l[l == 2] = 0

        P = l[l == 0].shape[0]
        L = l[l == 1].shape[0]
        print(P, L)
        y_unit=1/P
        X_unit=1/L

        data = []
        for i in range(p.shape[0]):
            data.append([p[i][0], l[i]])
        data.sort(reverse=True)
        X=[]
        y=[]
        current_X=0
        current_y=0
        for row in data:
            if row[1] == 0:
                current_y+=y_unit
            else:
                current_X+=X_unit
            X.append(current_X)
            y.append(current_y)

        X=np.array(X)        
        y=np.array(y)    

        plt.figure(figsize=[12,6])
        plt.title('Receiver Operating Characteristic')
        plt.plot(X, y, color = 'orange')
        plt.legend()
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
   
    @staticmethod
    def show_grid(loss, acc): 
    
        plt.figure(figsize=[12,6])
        plt.subplot(1,2,1)

        plt.plot(loss['train'], label='train_loss')
        plt.plot(loss['valid'], label='val_loss')   
        plt.xlabel('epochs')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(acc['train'], label='train_acc')
        plt.plot(acc['valid'], label='val_acc')   
        plt.xlabel('epochs')
        plt.legend()
        plt.show()
    
    def early_stopping(self, current_loss_delta, val_acc):
        last_loss_delta = 0
        if current_loss_delta > last_loss_delta:
            self.patient -= 1
        if self.patient <= 0 and val_acc > 0.97:
            return True
        return False
    
    def train(self, epoch):
        if epoch == n_epochs/5 + 1:
            for param in self.model.parameters():
                param.requires_grad = True
    
        self.model.train()
        
        epoch_loss=0 
        correct=0 
        total=0
        
        for inputs, labels in self.train_data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad() # zeroed grads
            outputs = self.model(inputs) # forward pass
            loss = self.criterion(outputs, labels) # softmax + cross entropy
            loss.backward() # back pass
            self.optimizer.step() # updated params
            epoch_loss += loss.item() # train loss
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
        self.lr_scheduler.step()
        print(epoch, self.lr_scheduler.get_last_lr()[0])
        self.lr_list.append(self.lr_scheduler.get_last_lr()[0])
        train_acc = correct / total
        train_loss = epoch_loss/len(labels)
        
        self.loss_list['train'].append(round(train_loss, 4)) 
        self.acc_list['train'].append(round(train_acc, 4))
        return train_loss, train_acc
        
    def validation(self):
        self.model.eval()
        
        a=0
        pred_val=0
        correct_val=0
        total_val=0
        
        with torch.no_grad():
            for inp_val, lab_val in self.valid_data_loader:
                inp_val = inp_val.to(self.device)
                lab_val = lab_val.to(self.device)
                out_val = self.model(inp_val)
                loss_val = self.criterion(out_val, lab_val)
                a += loss_val.item()
                _, pred_val = torch.max(out_val, dim=1)
                correct_val += (pred_val.cpu()==lab_val.cpu()).sum().item()    
                total_val += lab_val.shape[0]
            acc_val = correct_val / total_val
            loss_val = a/len(lab_val)
            
            self.loss_list['valid'].append(round(loss_val, 4))
            self.acc_list['valid'].append(round(acc_val, 4))
        return loss_val, acc_val
    
    def test(self):
        re_output = np.array([[]])
        re_labels = np.array([])
        correct = 0
        total = 0
        start_time = time.time()
        with torch.no_grad():
            for images, labels in self.test_data_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                p = np.copy(predicted.cpu())
                l = np.copy(labels.cpu())
                p[p == 2] = 0
                l[l == 2] = 0
                print(p)
                print(l)
                self.cm[0][0] += (l+p == 2).sum().item()
                self.cm[1][0] += (p-l == 1).sum().item() 
                self.cm[0][1] += (l-p == 1).sum().item()
                self.cm[1][1] += (l+p == 0).sum().item()
                try:
                    re_output = np.concatenate((re_output, outputs.data.cpu())) 
                    re_labels = np.concatenate((re_labels, labels.data.cpu())) 
                except:
                    re_output, re_labels = outputs.data.cpu(), labels.data.cpu()
        self.test_acc = round(float(correct)/float(total)*100, 4)
        end_time = time.time()
        self.test_time = end_time - start_time
        print(f"Accuracy of the network on the test images: {self.test_acc:.2f}")
        return re_output, re_labels
    
    def save(self, file_path):
        torch.save(self.model, file_path)