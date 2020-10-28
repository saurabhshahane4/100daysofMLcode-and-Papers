from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd 
import torchvision.transforms as transforms
import os

#fix torch random seed
torch.manual_seed(0)

class histoCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        #path to images
        path2data = os.path.join(data_dir, data_type)

        filenames = os.lisdir(path2data)

        self.full_filenames = [os.path.join(path2data, f) for f in filenames]
        csv_filename = data_type+"_labaels.csv"
        path2csvLabels=os.path.join(data_dir, csv_filename)
        labels_df=pd.read_csv(path2csvLabels)

        labels_df.set_index("id", inplace=True)

        self.labels = [labels_df.iloc[csv_filename[:-4]].values[0] for filename in filenames]

        self.transform = transform
    
    def __len__(self):
        return len(self.full_filenames)
    
    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]

    