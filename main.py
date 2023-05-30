
import gdown
import torch
import torchvision
import numpy as np
import gradio as gr

#import cv2
import os

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn 
from collections import OrderedDict
import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler
from torchvision.models import resnet50,vgg19
import tempfile
tmpdir = tempfile.gettempdir()

from pretrained import Model
url = 'https://drive.google.com/uc?id=1-0ExaDwcpNslOXMGL7K3wdu2kmyj-Z41'
output = tmpdir+'/skin_model_merged_data.pickle'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1G-iyTADMKBtHPZFPlQNc1yviVzVNzPdH'
output = tmpdir+'/brain_model_merged_data.pickle'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=1BJvd2Dn9fui_lSgxawMwf6epCUUGONtG'
output = tmpdir+'/kidney_model_merged_data.pickle'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1-4WDVwsqrrwabOUpo7aHQ-gt-blHnXv9'
output = tmpdir+'/breast_model_merged_data.pickle'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=16OhmHaoKQ7vAdzprp9XZqPBqdPsa1MHY'
output =tmpdir+ '/lung_model_merged_data.pickle'
gdown.download(url, output, quiet=False)
import os
os.chdir(tmpdir)


import pickle
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
def load_model(name):
    pickle_in =  open("./"+str(name)+"_model_merged_data.pickle","rb")
    fc_layer = [1024, 512]
    lr = 0.01
    contents = CPU_Unpickler(pickle_in).load()
    model2 = Model(fc_layer, True, lr, None, None, None, 'cpu')
    model2.model=contents
    return model2
test_transforms2 = transforms.Compose([ 
                                       
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]),
])

transforms.ToPILImage(),
cuda_kwargs_te2 = {
    'num_workers' : 2,
    'pin_memory' : True,
    'batch_size' : 64,
    'shuffle' : False
}

import matplotlib.image as mpimg
from torchvision.io import read_image




# Define the function to make predictions on an image
def predict(img,txt):
    try:
        #print(img.shape)
        img = test_transforms2(img)
        model2=load_model(txt)
        print(txt)
        
        test_data_loader2  = torch.utils.data.DataLoader(img, **cuda_kwargs_te2) #,**cuda_kwargs_te2
        imgs = next(iter(test_data_loader2))
        for inputs in test_data_loader2:
            inputs = inputs #.to(device)
            inputs = torch.unsqueeze(inputs, 0)
            #labels = labels.to(device)
            #model.optimizer.zero_grad() # zeroed grads
            outputs = model2.model(inputs) # forward pass
            _, predicted = torch.max(outputs.data, 1)
        if(txt=='skin'):
          lab={0:'Benign',1:'Malignant'}
          return lab[int(predicted[0])]
        if(txt=='brain'):
          lab={0:'No Brain Tumor',1:'Brain Tumor Detected'}
  
          return lab[int(predicted[0])]
        if(txt=='kidney'):
          lab={0:'No Kidney Tumor',1:'Kidney Tumor Detected'}
  
          return lab[int(predicted[0])]     

        if(txt=='breast'):
          lab={0:'Breast Cancer Detected',1:'Normal Breast'}
  
          return lab[int(predicted[0])]
            
        if(txt=='lung'):
          lab={0:'Benign Lung Tumor Detected',1:'Malignant Lung Tumor Detected',2:'Normal Lung'}
  
          return lab[int(predicted[0])]  
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []

# Define the interface
def app(request):
    title = "Cancer Identification"

    gr.Interface(
        title=title,
        fn=predict,
        inputs=[gr.Image(type="pil"),gr.Textbox(placeholder="Organ?")],
        outputs=gr.Label(
            num_top_classes=1,
        ),
        examples=[

        ],
    ).launch()




# # Run the app
# if __name__ == "__main__":
