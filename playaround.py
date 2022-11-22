import torch
import os
import pandas as pd
from Hindi_dataset import *
import matplotlib.pyplot as plt

#Root directory contains list of all 
rootdir = r"E:\Nikhil Spring 22\archive\Hindi-MNIST\train"

def generate_annotations(rootdir):
    df = pd.DataFrame(columns=['path', 'label'])
    for dir in os.listdir(rootdir):
         currLabel = int(str((dir)))
         for filename in os.listdir(os.path.join(rootdir,dir)):
             path = os.path.join(rootdir,dir,filename)
             temp_dict = {'path': path, 'label':currLabel}
             df = df.append(temp_dict, ignore_index=True)
    
    return df

df = generate_annotations(rootdir=rootdir)
df.to_csv('LabelMap.csv')

Hindi = Hindi_Digits('LabelMap.csv')

x, label = Hindi[10]
plt.imshow(x.squeeze(), cmap = 'gray')
print("Here")
