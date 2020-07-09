# import here
import argparse
import torch
from torchvision import models
from torch import nn
from torch import optim
import numpy as np
from functions import ProjectClassifier,train_model,test_model,save_model
from data_processing import load_data
# Set up arguments to be used in the training
parser= argparse.ArgumentParser(description='Train Nerual Network')
parser.add_argument('--data_dir',dest ='data_directory',type=str,action='store',help='Provide Dataset directory')
parser.add_argument('--save_dir',type=str,dest = 'save_directory',action='store',help='Provide checkpoint directory',default = 'checkpoint.pth')
parser.add_argument('--arch',type=str,action='store',
                    dest = 'model_arch',help='Choose the network architecture. The default is VGG-11',default='vgg11',choices=['vgg11', 'vgg13', 'vgg16'])
parser.add_argument('--hidden_units',dest='units',type=int,action='store',help='Hidden units in a classifier. The default is 1000 ', default=1000)
parser.add_argument('--learning_rate',dest='lr',type=float,action='store',help='Provide Learning rate. The default is 0.0005',default=0.0005)
parser.add_argument('--epochs',dest='num_epochs',type=int,action='store',help='Number of epochs. The default is 3',default=3)
parser.add_argument('--gpu',dest='gpu',help='Use GPU or not for training. The default is True',type=bool,default=True)

args=parser.parse_args()
device= torch.device('cuda:0'if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
data_dir = args.data_directory
save_dir = args.save_directory
learning_rate = args.lr
hidden_units = args.units
epochs = args.num_epochs 
arch=args.model_arch
# Process and load the data 
train_data,vaild_data,test_data,trainloader,validloader,testloader = load_data(data_dir) 
# Loading the pre-trained network
model = getattr(models,arch)(pretrained=True)
input_units = model.classifier[0].in_features
# Creating the model
model=ProjectClassifier(model,input_units,hidden_units)
criterion=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)
model.to(device)
#Training the model
model, optimizer=train_model(model,trainloader, validloader,criterion,optimizer,epochs,device)
# Testing the model
test_model(model,testloader,criterion,device)    
# Saving the model
save_model(model,arch,epochs,optimizer,train_data, save_dir)
       