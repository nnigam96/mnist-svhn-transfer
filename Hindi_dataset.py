import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class Hindi_Digits(Dataset):
    def __init__(self, csv='LabelMap_only1.csv', transform=None):
        self.annotation = pd.read_csv(csv)
        #self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.annotation.iloc[idx].path
        image = Image.open(img_name)
        label = self.annotation.iloc[idx].label
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'label': label}
        image = transforms.PILToTensor()(image)
        #if self.transform:
            #sample = self.transform(sample)
        image = image.to(torch.float32)
        #label = label.to(torch.float32)
        return image, label