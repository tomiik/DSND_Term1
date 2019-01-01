# Imports
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from collections import OrderedDict
from torch import nn
from torch import optim
import time
import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('--save_dir', type=str, default='checkpoint.pth')
parser.add_argument('--arch', type=str, default='vgg13')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() & args.gpu else 'cpu')
print('Device: {}'.format(device))

# Load data
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'
test_dir = args.data_dir + '/test'
print('Train dir: {}'.format(train_dir))
print('Valid dir: {}'.format(valid_dir))
print('Test dir: {}'.format(test_dir))

# Transforms
train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


valid_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load datasets
train_data = datasets.ImageFolder(train_dir, train_transform)
valid_data = datasets.ImageFolder(valid_dir, valid_transform)
test_data = datasets.ImageFolder(test_dir, test_transform)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# Map Label
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print(len(cat_to_name))

# Model
if args.arch == "vgg13":
    model = models.vgg13(pretrained=True)
elif args.arch == "vgg16":
    model = models.vgg16(pretrained=True)
elif args.arch == "vgg19":
    model = models.vgg19(pretrained=True)
else:
    model = models.vgg13(pretrained=True)
    
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc2', nn.Linear(4096, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.3)),
                          ('fc3', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Train
def train(model, train_loader, optimizer, criterion, epochs, print_every):
        steps = 0
        
        model.to(device)
        
        for epoch in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                probs = torch.exp(outputs).data
                equality = (labels.data == probs.max(1)[1])
                
                if steps % print_every == 0:
                    model.eval()
                    valid_loss, valid_accuracy = validate(model, valid_loader)
                    print("Epoch: {} / {} -".format(epoch+1, epochs),
                          "Running Loss: {:.3f}, ".format(running_loss/print_every),
                          "Validation(Loss: {:.3f}, ".format(valid_loss),
                          "Accuracy: {:.3f} %)".format(valid_accuracy)
                         )
                    running_loss = 0
                    model.train()
                    
# Validate
def validate(model, data_loader):
    model.to(device)
    correct = 0
    total = 0
    steps = 0
    valid_loss = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            steps += 1
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    return valid_loss/steps,100 * correct/total # loss, accuracy 

epochs = args.epochs
print_every = 5
train(model, train_loader, optimizer, criterion, epochs, print_every)
print("Done training.")

# Save the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint_dict = {
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'classifier': classifier,
    'class_to_idx': model.class_to_idx,
    'optimizer_state': optimizer.state_dict,
    'epochs': epochs
}

torch.save(checkpoint_dict, args.save_dir)
print("Saved {}".format(args.save_dir))