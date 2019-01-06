import torch
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

def cal_accuracy(model, dataloaders, dataset_sizes, phase):
    model.eval()
    model.to(device)
    
    running_corrects = 0
    test_acc = 0.0
    
    for idx, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs,1)
        
        running_corrects += torch.sum(preds==labels)
        
    test_acc = running_corrects.double()/dataset_sizes[phase]
    print("Test Accuracy : {:.4f}".format(test_acc))
    
        
        