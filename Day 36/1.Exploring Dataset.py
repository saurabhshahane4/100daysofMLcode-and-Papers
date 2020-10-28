#website =  https:/​/​www.​kaggle.​com/​c/histopathologic-​cancer-​detection/​data
import pandas as pd 
path2csv = "./data/train_labels.csv"
labels_df = pd.read_csv(path2csv)
labels_df.head()

print(labels_df['label'].value_counts())
labels_df['label'].hist();

import matplotlib.pylab as plt
from PIL import Image, ImageDraw
import numpy as np
import os

malignantIds = labels_df.iloc[labels_df['label']==1]['id'].values

path2train="./data/train/"

color=False
plt.rcParams['figure.figsize'] = (10.0 , 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
nrows, ncols=3,3

for i , id_ in enumerate(malignantIds[:nrows*ncols]):
    full_filenames = os.path.join(path2train, id_ + '.tif')
    #load image
    img = Image.open(full_filenames)

    draw = ImageDraw.Draw(img)
    draw.rectangle(((32,32),(64,64)), outline='green')
    plt.subplot(nrows, ncols, i+1)
    if color is True:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0],cmap="gray")
    plt.axis('off')

print("image shape:", np.array(img).shape)
print("pixel values range from %s to %s" %(np.min(img), np.max(img)))
