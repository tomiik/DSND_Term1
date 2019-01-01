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


# Get args
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('checkpoint', type=str)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() & args.gpu else 'cpu')
print('Device: {}'.format(device))


def load_checkpoint(path):
    checkpoint = torch.load(path)
    
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_img(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(img)

    width, height = img.size
    
    if width < height:
        img.thumbnail((256, 100000))
    elif width > height:
        img.thumbnail((100000, 256))
    else:
        img.thumbnail((256, 256))
        
    width, height = img.size
    
    left = (width - 224)/ 2
    bottom = (height - 224) / 2
    right = left + 224
    top = bottom + 224
    
    # Crop
    img = img.crop((left, bottom, right, top))
    
    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    img = img.transpose((2, 0, 1))

    return img

def predict(image_path, model, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_img(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    image_tensor = image_tensor[None]
    image_tensor = image_tensor.to(device)
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    ps = torch.exp(output)
    
    top_ps, top_labels = ps.topk(k)
    
    top_ps = top_ps.cpu().numpy().reshape(k,)
    top_idx = top_labels.cpu().numpy().reshape(k,)
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    probs = list(top_ps)
    ids = list(top_idx)
    labels = []
    flowers = []
    
    for idx in ids:
        labels.append(idx_to_class[idx])
        flowers.append(cat_to_name[idx_to_class[idx]])
    
    return probs, ids, flowers


# Load checkpoint
model = load_checkpoint(args.checkpoint)
model.eval()
model.train()

# Map Label
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
# Predict
probs, classes, flowers = predict(args.input, model, args.top_k)


print('probs: ', probs)
print('classes: ', classes)
print('flowers: ', flowers)