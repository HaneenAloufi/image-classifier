import torch
import time
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from PIL import Image

# Function for Creating the Network
def ProjectClassifier(model,input_units,hidden_units):

    # Turning off gradients for the model
    for p in model.parameters():
        p.requires_grad=False

    # Define a new claaifier
    model.classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(input_units,hidden_units)),
                                            ('relu',nn.ReLU()),
                                            ('dropout',nn.Dropout(p=0.2)),
                                            ('fc2',nn.Linear(hidden_units,102)),
                                            ('outputs',nn.LogSoftmax(dim=1))
                                            ]))
    return model
# Function for Training the model on train and valid data
def train_model(model,trainloader, validloader,criterion,optimizer,epochs,device):
    # Training the model
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0
    steps = 0
    print_every = 40
    start = time.time()
    for e in range(epochs):
        for images,labels in trainloader:
            steps+=1
            images, labels= images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs=model.forward(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss+= loss.item()

            # Vaildation and calculating accracy
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images,labels= images.to(device),labels.to(device)
                        outputs=model.forward(images)
                        v_loss= criterion(outputs,labels)
                        valid_loss+=v_loss.item()
                        # Accuracy
                        ps=torch.exp(outputs)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


                print(f"Epoch {e+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Vaild loss: {valid_loss/len(validloader):.3f}.. "
                  f"Vaild accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    time_taken = time.time() - start
    print("\nTotal Time: {:.0f}m {:.0f}s".format(time_taken//60, time_taken % 60))
    return model,optimizer

# Function to do validation on the test set
def test_model(model,testloader,criterion,device):
    test_loss=0
    test_accuracy=0
    with torch.no_grad():
        for images, labels in testloader:
            images,labels= images.to(device),labels.to(device)
            outputs=model.forward(images)
            t_loss= criterion(outputs,labels)
            test_loss+=t_loss.item()
            # Accuracy
            ps=torch.exp(outputs)
            top_p,top_class=ps.topk(1,dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Test Accuracy: {:.3f}%".format(test_accuracy/len(testloader)*100))

# Function to save the model
def save_model(model,arch,epochs,optimizer,train_data, save_dir):
    model.class_to_idx = train_data.class_to_idx
    checkpoint={'classifier': model.classifier,
            'arch':arch,
            'optim_state': optimizer.state_dict,
            'state_dict': model.state_dict(),
            'num_epochs': epochs,
            'mapping': model.class_to_idx}

    return torch.save(checkpoint,save_dir)

# Function to load a checkpoint and rebuilds the model
def load_checkpoint(model,save_dir):
    checkpoint= torch.load(save_dir)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']
    for param in model.parameters():
        param.requires_grad = False
    return model

# Function to predict flower name
def predict(process_image, model, top_k,device):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # processing the image and converting numpy to tensor

    image=process_image
    image=image.to(device).float()
    image=image.unsqueeze_(0)
    # findding the top 5 probability
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps=torch.exp(output)
    top_p, top_class = ps.topk(top_k)
    # coverting tensor to numpy then to a list
    top_p=top_p.cpu().numpy().tolist()[0]
    probs=top_p
    top_class=top_class.cpu().numpy().tolist()[0]
    # 
    idx_to_class={val:key for key,val in model.class_to_idx.items()}
    classes=[idx_to_class[i]for i in top_class]

    return probs, classes
