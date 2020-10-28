import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor()])

data_dir = "./data/"
histo_dataset = histoCancerDataset(data_dir, data_transformer,"train")
print(len(histo_dataset))

img, label = histo_dataset[9]
print(img.shape, torch.min(img), torch.max(img))

torch.Size([3, 96, 96])

#splitting the dataset
from torch.utils.data import random_split

len_histo = len(histo_dataset)
len_train = int(0.8*len_histo)
len_val = len_histo - len_train
train_ds, val_ds = random_split(histo_dataset,[len_train, len_val])
print("train_dataset_length:", len(train_ds))
print('val dataset length:', len(val_ds))

for x,y in train_ds:
    print(x.shape,y)
    break

for x,y in val_ds:
    print(x.shape,y)
    break

from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt 
np.random.seed(0)

def show(img,y,color=False):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    if color==False:
        npimg_tr=npimg_tr[:,:,0]
        plt.imshow(npimg_tr, interpolation='nearest', cmap='gray')
    else:
        plt.imshow(npimg_tr, interpolation='nearesr')
    plt.title("label:"+str(y))

#grid of sample images
grid_size =4
rnd_inds = np.random.randint(0, len(train_ds), grid_size)
print('image indices:', rnd_inds)

x_grid_train=[train_ds[i][0] for i in rnd_inds]
y_grid_train = [train_ds[i][1] for i in rnd_inds]

x_grid_train = utils.make_grid(x_grid_train, nrow=4, padding=2)
print(x_grid_train.shape)
plt.rcparams['figure.figsize'] = (10.0, 5)
show(x_grid_train,y_grid_train)


#image transformation
train_transformer= transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(96, scale=(0.8,1.0), ratio=(1.0,1.0)),
    transforms.ToTensor()
])

val_transformer = transforms.Compose([transforms.ToTensor()])
train_ds.transform = train_transformer
val_ds.transform = val_transformer