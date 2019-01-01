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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

# Get args
args = sys.argv
img_path = args[1]
checkpoint_name = args[2]


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
model = load_checkpoint(checkpoint_name)
model.eval()
model.train()

# Map Label
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Process input image
processed_img = process_img(img_path)

# Predict
probs, classes, flowers = predict(img_path, model, topk)


print(probs)
print(classes)
print(flowers)
print("Predicted class: {}, probability: {:.3f}".format(topk_class, topk_probs))