import torch
import numpy as np
from torch import nn
from PIL import Image
from torchvision import datasets, transforms


def load_data(data_dir):
    
    # Defining transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]) 
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]) 
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir+'/train',transform=train_transforms)
    vaild_data = datasets.ImageFolder(data_dir+'/valid',transform=valid_transforms)
    test_data = datasets.ImageFolder(data_dir+'/test',transform=test_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(vaild_data,batch_size=32,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=True)

    return train_data,vaild_data,test_data,trainloader,validloader,testloader


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    # Converting image to PIL image using image file path
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = im.transpose((2,0,1))
    
    return torch.from_numpy(im)

