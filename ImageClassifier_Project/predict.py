import argparse
import json
import torch
from torchvision import models
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from functions import ProjectClassifier,train_model,test_model,save_model,load_checkpoint,predict
from data_processing import process_image


parser= argparse.ArgumentParser(description='Predict flower name')
parser.add_argument('--top_k',type=int,dest='k_value',action='store',help='Enter L to show top K most likely classes. The default is 5',default=5)
parser.add_argument('--category_names',dest='cat_names',action='store',help='Mapping of categories to real names. JSON file name',default='cat_to_name.json')
parser.add_argument('--image_path',type=str,dest='image_dir',required=True,action='store',help='Enter image path')
parser.add_argument('--save_dir',type=str,dest = 'save_directory',action='store',help='Provide checkpoint directory',default = 'checkpoint.pth')
parser.add_argument('--arch',type=str,action='store',
                    dest = 'model_arch',help='Choose the network architecture. The default is VGG-11',default='vgg11',choices=['vgg11', 'vgg13', 'vgg16'])
parser.add_argument('--gpu',dest='gpu',help='Use GPU or not for training. The default is True',type=bool,default=True)

args=parser.parse_args()
device= torch.device('cuda:0'if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
top_k=args.k_value
category_names=args.cat_names
image_path=args.image_dir
save_dir = args.save_directory
arch=args.model_arch

# Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the model 
model = getattr(models,arch)(pretrained=True)
model=load_checkpoint(model,save_dir)

# Make predictions
probs,classes=predict(process_image(image_path), model, top_k,device)
flower_names = [cat_to_name[item] for item in classes]

# Print predicted flower names with top K classes
print("The flower is most likely to be a:")
for k in range(top_k):
     print("Number: {}/{}.. ".format(k+1, top_k),
            "Class name: {}.. ".format(flower_names [k]),
            "Probability: {:.3f}..% ".format(probs [k]*100),
            )

